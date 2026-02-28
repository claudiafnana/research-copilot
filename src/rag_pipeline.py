from src.ingestion.pdf_extractor import extract_text_from_pdf
from src.ingestion.text_cleaner import clean_extracted_text
from src.chunking.chunker import TokenChunker
from src.embedding.embedder import OpenAIEmbedder
from src.vectorstore.chroma_store import ChromaVectorStore


if __name__ == "__main__":
    pdf_path = "papers/agroecologia_e_innovacion.pdf"

    print("📄 Extrayendo PDF...")
    result = extract_text_from_pdf(pdf_path)
    cleaned = clean_extracted_text(result["text"])

    print("Chunking...")
    chunker = TokenChunker(chunk_size=256, chunk_overlap=25)
    chunks = chunker.chunk_text(cleaned, metadata={"paper_id": "paper_001"})

    test_chunks = chunks[:3]

    texts = [c["text"] for c in test_chunks]
    ids = [f"paper_001_chunk_{i}" for i in range(len(test_chunks))]
    metadatas = [c["metadata"] for c in test_chunks]

    print("Generando embeddings...")
    embedder = OpenAIEmbedder()
    embeddings = embedder.embed_texts(texts)

    print("Guardando en Chroma...")
    store = ChromaVectorStore(persist_directory="./chroma_db")
    store.create_collection("papers_test")
    store.add_documents(ids, texts, embeddings, metadatas)

    print("Probando búsqueda...")
    query_embedding = embedder.embed_query("¿Qué es el desarrollo territorial?")
    results = store.query(query_embedding, n_results=2)

    print("\nResultados encontrados:\n")
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        print("Distancia:", dist)
        print(doc[:300])
        print("----")