from src.ingestion.pdf_extractor import extract_text_from_pdf
from src.ingestion.text_cleaner import clean_extracted_text
from src.chunking.chunker import TokenChunker
from src.embedding.embedder import OpenAIEmbedder
from src.vectorstore.chroma_store import ChromaVectorStore
from src.generation.generator import OpenAIGenerator


if __name__ == "__main__":

    # --- Config ---
    pdf_path   = "papers/agroecologia_e_innovacion.pdf"
    paper_id   = "paper_001"
    chunk_size = 512
    chunk_overlap = 50
    collection_name = "research_copilot_512"

    # --- Ingestion ---
    print("Extracting PDF...")
    result  = extract_text_from_pdf(pdf_path)
    cleaned = clean_extracted_text(result["text"])

    # Build flat metadata for Chroma (no nested dicts)
    pdf_meta = result["metadata"]
    base_metadata = {
        "paper_id":   paper_id,
        "title":      pdf_meta.get("title", "") or "Untitled",
        "author":     pdf_meta.get("author", "") or "Unknown",
        "year":       pdf_meta.get("year", "n.d."),
        "chunk_size": chunk_size,
    }

    # --- Chunking ---
    print("Chunking...")
    chunker = TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks  = chunker.chunk_text(cleaned, metadata=base_metadata)

    # --- Embeddings ---
    print("Generating embeddings...")
    embedder   = OpenAIEmbedder()
    texts      = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_texts(texts)

    # --- Storage ---
    print("Storing in Chroma...")
    store = ChromaVectorStore()
    store.create_collection(collection_name)

    # IDs are now globally unique: {paper_id}_chunk_{i}
    ids       = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [base_metadata for _ in chunks]

    store.add_documents(ids, texts, embeddings, metadatas)
    print(f"Stored {len(chunks)} chunks with IDs: {ids[0]} ... {ids[-1]}")

    # --- Retrieval test ---
    print("Testing retrieval...")
    query           = "What is agroecology?"
    query_embedding = embedder.embed_query(query)
    results         = store.query(query_embedding, n_results=3)

    retrieved_texts  = results["documents"][0]
    retrieved_metas  = results["metadatas"][0]
    retrieved_dists  = results["distances"][0]

    # Context now includes source metadata so GPT can cite properly
    context_parts = []
    for text, meta, dist in zip(retrieved_texts, retrieved_metas, retrieved_dists):
        header = f"[Source: {meta['title']} | {meta['author']} | {meta['year']}]"
        context_parts.append(f"{header}\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    # --- Generation ---
    print("Generating answer with GPT...")
    generator = OpenAIGenerator()
    answer    = generator.generate_answer(query, context)

    print("\n=============================")
    print("QUESTION:", query)
    print("\nANSWER:\n")
    print(answer)
    print("\nSOURCES RETRIEVED:")
    for meta, dist in zip(retrieved_metas, retrieved_dists):
        print(f"  - {meta['title']} ({meta['year']}) | distance: {dist:.4f}")
