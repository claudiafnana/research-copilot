from src.retrieval.retriever import RetrievedChunk


def format_apa(chunk: RetrievedChunk) -> str:
    """
    Generate an APA 7th edition citation string from a RetrievedChunk.
    Format: Author (Year). Title. Journal.
    """
    author  = chunk.author  or "Unknown Author"
    year    = chunk.year    or "n.d."
    title   = chunk.title   or "Untitled"
    journal = getattr(chunk, "journal", "") or ""

    citation = f"{author} ({year}). *{title}*."
    if journal:
        citation += f" {journal}."
    return citation


def deduplicated_citations(chunks: list[RetrievedChunk]) -> list[str]:
    """Return one APA citation per unique paper_id, sorted alphabetically."""
    seen = {}
    for c in chunks:
        if c.paper_id not in seen:
            seen[c.paper_id] = format_apa(c)
    return sorted(seen.values())
