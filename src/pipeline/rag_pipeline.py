from dotenv import load_dotenv
load_dotenv()

from src.ingestion.pdf_extractor import extract_text_from_pdf
from src.ingestion.text_cleaner import clean_extracted_text
from src.chunking.chunker import TokenChunker
from src.embedding.embedder import OpenAIEmbedder
from src.vectorstore.chroma_store import ChromaVectorStore
from src.retrieval.retriever import Retriever, RetrievedChunk
from src.generation.generator import OpenAIGenerator


class RAGPipeline:
    """
    Orchestrates the full RAG flow: ingestion and query.

    Parameters
    ----------
    collection_name : str
        ChromaDB collection to use (e.g. 'research_copilot_256').
    chunk_size : int
        Token size for chunking (256 or 1024 for the two configurations).
    chunk_overlap : int
        Token overlap between consecutive chunks.
    """

    def __init__(
        self,
        collection_name: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.collection_name = collection_name
        self.chunk_size      = chunk_size
        self.chunk_overlap   = chunk_overlap

        self.embedder  = OpenAIEmbedder()
        self.store     = ChromaVectorStore()
        self.store.create_collection(collection_name)
        self.chunker   = TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retriever = Retriever(collection_name=collection_name)
        self.generator = OpenAIGenerator()

    # ------------------------------------------------------------------
    # INGESTION
    # ------------------------------------------------------------------

    def ingest_paper(
        self,
        pdf_path: str,
        paper_id: str,
        override_metadata: dict = None,
    ) -> int:
        """
        Extract, chunk, embed and store one PDF.

        override_metadata : dict, optional
            Manual fields (title, author, year, journal) that take priority
            over whatever PyMuPDF extracts. Use this for the ingestion manifest.

        Returns the number of chunks stored.
        """
        # --- Skip if already indexed ---
        existing = self.store.collection.get(ids=[f"{paper_id}_chunk_0"])
        if existing["ids"]:
            print(f"  [skip] {paper_id} already indexed.")
            return 0

        # --- Extract ---
        result  = extract_text_from_pdf(pdf_path)
        cleaned = clean_extracted_text(result["text"])

        # --- Build metadata (override wins over PDF metadata) ---
        pdf_meta = result["metadata"]
        base_metadata = {
            "paper_id":   paper_id,
            "title":      (override_metadata or {}).get("title")  or pdf_meta.get("title")  or "Untitled",
            "author":     (override_metadata or {}).get("author") or pdf_meta.get("author") or "Unknown",
            "year":       (override_metadata or {}).get("year")   or pdf_meta.get("year")   or "n.d.",
            "journal":    (override_metadata or {}).get("journal", ""),
            "chunk_size": self.chunk_size,
        }

        # --- Chunk ---
        chunks = self.chunker.chunk_text(cleaned, metadata=base_metadata)

        # --- Embed ---
        texts      = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # --- Store ---
        ids       = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [base_metadata for _ in chunks]
        self.store.add_documents(ids, texts, embeddings, metadatas)

        return len(chunks)

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 5,
        paper_id: str = None,
        strategy: str = "default",
    ) -> dict:
        """
        Retrieve relevant chunks and generate an answer.

        Returns a dict with:
          - answer   : str
          - chunks   : list[RetrievedChunk]
          - strategy : str
        """
        chunks = self.retriever.retrieve(question, paper_id=paper_id)

        answer = self.generator.generate_answer(
            question=question,
            chunks=chunks,
            strategy=strategy,
        )

        return {
            "answer":   answer,
            "chunks":   chunks,
            "strategy": strategy,
        }
