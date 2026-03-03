import streamlit as st
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from collections import Counter
from src.pipeline.rag_pipeline import RAGPipeline
from src.citation.apa_formatter import deduplicated_citations

@st.cache_resource
def get_pipeline(collection_name: str, chunk_size: int):
    return RAGPipeline(collection_name=collection_name, chunk_size=chunk_size)

@st.cache_data
def load_manifest():
    with open("papers/papers_manifest.json", encoding="utf-8") as f:
        return json.load(f)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Research Copilot", page_icon="🌱", layout="wide")
st.title("🌱 Research Copilot: Academic Paper Assistant")
st.caption("Agroecology Knowledge Base — RAG-powered Q&A with 20 indexed papers")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    chunk_config = st.selectbox(
        "Chunk Configuration",
        ["256 tokens (precise)", "1024 tokens (broad context)"]
    )
    collection = "research_copilot_256" if "256" in chunk_config else "research_copilot_1024"
    chunk_size = 256 if "256" in chunk_config else 1024

    strategy = st.selectbox(
        "Prompt Strategy",
        ["clear_delimiters", "json_output", "few_shot", "chain_of_thought"],
        help="Strategy 1: Clear delimiters | 2: JSON output | 3: Few-shot | 4: Chain-of-thought"
    )

    papers = load_manifest()
    paper_options = ["All papers"] + [f"{p['paper_id']}: {p['title'][:45]}..." for p in papers]
    paper_filter = st.selectbox("Filter by paper", paper_options)
    selected_paper_id = None if paper_filter == "All papers" else paper_filter.split(":")[0].strip()

    top_k = st.slider("Top-K chunks retrieved", 3, 10, 5)

    st.divider()
    st.caption(f"Collection: `{collection}`")
    st.caption(f"Strategy: `{strategy}`")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Paper Browser", "📊 Dashboard"])

# ─────────────────────────────────────────────────────────────
# TAB 1 — CHAT
# ─────────────────────────────────────────────────────────────
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📎 Sources ({len(msg['citations'])} papers cited)"):
                    for cite in msg["citations"]:
                        st.markdown(f"- {cite}")

    # New input
    if prompt := st.chat_input("Ask a question about agroecology..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching papers and generating answer..."):
                pipeline = get_pipeline(collection, chunk_size)
                result   = pipeline.query(
                    question=prompt,
                    top_k=top_k,
                    paper_id=selected_paper_id,
                    strategy=strategy,
                )
                answer    = result["answer"]
                citations = deduplicated_citations(result["chunks"])

            st.markdown(answer)

            if citations:
                with st.expander(f"📎 Sources ({len(citations)} papers cited)"):
                    for cite in citations:
                        st.markdown(f"- {cite}")

            st.caption(f"Strategy: `{strategy}` | Chunks: `{chunk_config}` | Top-K: `{top_k}`")

        st.session_state.messages.append({
            "role":      "assistant",
            "content":   answer,
            "citations": citations,
        })

# ─────────────────────────────────────────────────────────────
# TAB 2 — PAPER BROWSER
# ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"📚 Indexed Papers ({len(papers)} total)")
    search_term = st.text_input("Search by title or author", "")

    filtered = [
        p for p in papers
        if search_term.lower() in p["title"].lower()
        or search_term.lower() in p["author"].lower()
    ] if search_term else papers

    for p in filtered:
        with st.expander(f"**{p['paper_id']}** — {p['title'][:80]}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Author:** {p['author']}")
                st.markdown(f"**Year:** {p['year']}")
            with col2:
                st.markdown(f"**Journal:** {p.get('journal', '—')}")
                st.markdown(f"**File:** `{p['filename']}`")

# ─────────────────────────────────────────────────────────────
# TAB 3 — DASHBOARD
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📊 Corpus Statistics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Papers", len(papers))
    col2.metric("Chunks (256-token)", "~2,098")
    col3.metric("Chunks (1024-token)", "~545")
    col4.metric("Prompt Strategies", "4")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Papers by Journal")
        journals = Counter(p.get("journal", "Unknown") for p in papers if p.get("journal"))
        st.bar_chart(dict(journals.most_common(8)))

    with col_b:
        st.subheader("Papers by Year")
        years = Counter(p.get("year", "n.d.") for p in papers)
        st.bar_chart(dict(sorted(years.items())))

    st.divider()
    st.subheader("Chunking Configuration Comparison")
    st.markdown("""
| Config | Chunk Size | Overlap | Total Chunks | Best For |
|--------|-----------|---------|-------------|----------|
| Small  | 256 tokens | 25 tokens | ~2,098 | Precise factual queries |
| Large  | 1,024 tokens | 100 tokens | ~545 | Broad synthesis questions |
    """)

    st.subheader("Prompt Strategy Comparison")
    st.markdown("""
| Strategy | Technique | Output Format | Best For |
|----------|-----------|--------------|----------|
| Clear Delimiters | XML `<context>` tags | Academic prose | General Q&A |
| JSON Output | Structured schema | Machine-readable JSON | Data extraction |
| Few-Shot | 2 example Q&A pairs | Academic prose | Consistent citation style |
| Chain-of-Thought | Step-by-step reasoning | Reasoned prose | Complex synthesis |
    """)
