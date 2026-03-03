# Research Copilot — Academic Paper Assistant

A Retrieval-Augmented Generation (RAG) system for querying a corpus of 20 agroecology research papers. Built with OpenAI `gpt-4o`, ChromaDB, and Streamlit.

---

## Architecture

```
User question
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                    RAGPipeline                      │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │ Retriever│───▶│ Prompt   │───▶│  Generator   │  │
│  │(ChromaDB)│    │ Strategy │    │  (gpt-4o)    │  │
│  └──────────┘    └──────────┘    └──────────────┘  │
│                                         │           │
│                                         ▼           │
│                                   ┌──────────┐      │
│                                   │  APA     │      │
│                                   │ Citations│      │
│                                   └──────────┘      │
└─────────────────────────────────────────────────────┘
     │
     ▼
   Answer + Citations
```

### Key components

| Module | Path | Responsibility |
|---|---|---|
| PDF Extractor | `src/ingestion/pdf_extractor.py` | Extracts text and metadata from PDFs via PyMuPDF |
| Text Cleaner | `src/ingestion/text_cleaner.py` | Normalises whitespace, removes artefacts |
| Chunker | `src/chunking/chunker.py` | Token-based sliding window (tiktoken) |
| Embedder | `src/embedding/embedder.py` | `text-embedding-3-small` via OpenAI API |
| Vector Store | `src/vectorstore/chroma_store.py` | Persistent ChromaDB collections |
| Retriever | `src/retrieval/retriever.py` | Cosine similarity search → `RetrievedChunk` objects |
| Prompt Strategies | `src/generation/prompt_strategies.py` | 4 prompting techniques (see below) |
| Generator | `src/generation/generator.py` | Calls `gpt-4o` with chosen strategy |
| APA Formatter | `src/citation/apa_formatter.py` | Formats APA 7th-edition citations from chunk metadata |
| RAG Pipeline | `src/pipeline/rag_pipeline.py` | Orchestrates ingestion and query flows |
| Streamlit App | `app/streamlit_app.py` | Interactive web interface |

---

## Project structure

```
tarea 01/
├── app/
│   └── streamlit_app.py          # Web interface
├── evaluation/
│   └── results.csv               # 80-row evaluation (20Q × 4 strategies)
├── papers/
│   ├── papers_manifest.json      # Metadata for all 20 PDFs
│   └── *.pdf                     # Agroecology paper corpus
├── scripts/
│   ├── ingest_all_papers.py      # Index all papers into ChromaDB
│   └── evaluate.py               # Run evaluation across strategies
├── src/
│   ├── chunking/
│   ├── citation/
│   ├── embedding/
│   ├── generation/
│   ├── ingestion/
│   ├── pipeline/
│   ├── retrieval/
│   └── vectorstore/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Chunking configurations

Two collections are maintained in ChromaDB:

| Collection | Chunk size | Overlap | Approx. chunks (20 papers) |
|---|---|---|---|
| `research_copilot_256` | 256 tokens | 25 tokens | ~2,046 |
| `research_copilot_1024` | 1,024 tokens | 100 tokens | ~532 |

**Trade-offs:**

- **256 tokens** — finer granularity, higher recall for specific facts, lower risk of irrelevant context diluting an answer. Downside: a single concept may be split across chunks.
- **1,024 tokens** — preserves more argumentative context per chunk, better for synthesis questions. Downside: more noise per chunk and higher token cost for generation.

---

## Prompt strategies

Four strategies are implemented in `src/generation/prompt_strategies.py`:

### 1. Clear Delimiters (`clear_delimiters`)
Wraps context in `<context>` XML tags and the question in `<question>` tags. The system prompt instructs the model to answer only from within those delimiters. Reduces hallucination by giving the model clear boundaries.

```
<context>
[Source: Title | Author | Year]
...retrieved text...
</context>

<question>
{question}
</question>
```

### 2. JSON Output (`json_output`)
Forces structured output matching a fixed schema:
```json
{
  "answer": "...",
  "confidence": "high|medium|low",
  "key_concepts": ["..."],
  "citations": [{"author":"...","year":"...","title":"..."}],
  "limitations": "..."
}
```
Useful for downstream parsing and automated evaluation.

### 3. Few-Shot (`few_shot`)
Provides two example question-answer pairs in academic style before asking the real question. Guides the model toward the desired register and citation style without fine-tuning.

### 4. Chain-of-Thought (`chain_of_thought`)
Instructs the model to reason in four explicit steps before writing the final answer:
1. Identify relevant sources
2. Extract key claims
3. Synthesise into a coherent answer
4. Add inline APA citations

Produces more transparent reasoning; useful when verifiability matters.

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd "tarea 01"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and add your key:
# OPENAI_API_KEY=sk-...
```

### 3. Ingest all papers

```bash
python scripts/ingest_all_papers.py
```

This indexes all 20 PDFs into both ChromaDB collections. Already-indexed papers are skipped automatically.

### 4. Run the app

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## Evaluation

Run the full evaluation (20 questions × 4 strategies = 80 GPT calls):

```bash
python scripts/evaluate.py
```

Results are saved to `evaluation/results.csv` with columns: `question_id`, `question`, `strategy`, `answer`, `citations`, `n_chunks`.

---

## Limitations

- **Closed-domain**: The system answers only from the 20 indexed papers. Questions outside this corpus return "I cannot find this in the provided papers."
- **Embedding cost**: Every new query calls the OpenAI Embeddings API. Running the full evaluation (80 calls) and re-ingesting papers will incur API costs.
- **No re-ranking**: Retrieved chunks are ordered by cosine distance only. A cross-encoder re-ranker could improve precision.
- **APA metadata quality**: Author and year fields depend on PDF metadata or the manual manifest. Errors in the manifest propagate to citations.
- **JSON strategy fragility**: If `gpt-4o` wraps the JSON in markdown fences the response parser may fail.

---

## Prompt strategy comparison

The evaluation script ran all 4 strategies against 20 questions (80 total GPT calls). Key observations from `evaluation/results.csv`:

| Strategy | Strengths | Weaknesses |
|---|---|---|
| `clear_delimiters` | Stays strictly on-topic, rarely hallucinates, clean prose | Can be overly cautious; sometimes refuses to synthesise across papers |
| `json_output` | Structured and machine-parseable; includes confidence score | Verbose; occasionally wraps output in markdown fences breaking the JSON |
| `few_shot` | Consistent academic register; good citation style | Examples may bias the answer style regardless of question type |
| `chain_of_thought` | Most transparent reasoning; best for complex synthesis questions | Longer outputs; intermediate reasoning steps add token cost |

**Recommendation:** Use `clear_delimiters` for factual questions and `chain_of_thought` for synthesis questions. Use `json_output` when the answer needs to be processed programmatically.

---

## Future improvements

1. **Re-ranking**: Add a cross-encoder re-ranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) after retrieval to improve chunk relevance before generation.
2. **Hybrid search**: Combine dense vector search with BM25 keyword search for better recall on specific terms (author names, years, technical terminology).
3. **Larger corpus**: Extend ingestion to support dynamic paper uploads through the Streamlit interface, beyond the fixed 20-paper manifest.
4. **Streaming responses**: Use OpenAI's streaming API to display answers token-by-token in the chat interface for a better user experience.
5. **Evaluation metrics**: Add automated metrics (RAGAS faithfulness, answer relevancy) instead of manual inspection of the CSV.
6. **Multi-language support**: Several papers are in Spanish; add language detection and cross-lingual retrieval for mixed-language corpora.

---

## Video

> 📹 Video link: _add your YouTube/Google Drive link here before submitting_

---

## Papers corpus

20 peer-reviewed papers on agroecology (2020–2026), including works from *Agronomy for Sustainable Development*, *Sustainability*, *Global Environmental Change*, and *Agroecology and Sustainable Food Systems*. Full list in `papers/papers_manifest.json`.
