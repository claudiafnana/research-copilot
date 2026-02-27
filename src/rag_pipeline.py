from src.ingestion.pdf_extractor import extract_text_from_pdf
from src.ingestion.text_cleaner import clean_extracted_text
from src.chunking.chunker import TokenChunker


if __name__ == "__main__":
    result = extract_text_from_pdf("papers/agroecologia_e_innovacion.pdf")
    cleaned = clean_extracted_text(result["text"])

    # Configuración 1: chunks pequeños
    chunker_small = TokenChunker(chunk_size=256, chunk_overlap=25)
    chunks_small = chunker_small.chunk_text(
        cleaned,
        metadata={"paper_id": "paper_001"}
    )

    # Configuración 2: chunks grandes
    chunker_large = TokenChunker(chunk_size=1024, chunk_overlap=100)
    chunks_large = chunker_large.chunk_text(
        cleaned,
        metadata={"paper_id": "paper_001"}
    )

    print("Total páginas:", result["total_pages"])
    print("Chunks SMALL (256 tokens):", len(chunks_small))
    print("Chunks LARGE (1024 tokens):", len(chunks_large))

    print("\n--- Ejemplo chunk SMALL ---\n")
    print(chunks_small[0]["text"][:400])
    print("\nTokens en ese chunk:", chunks_small[0]["token_count"])