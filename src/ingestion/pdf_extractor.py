import fitz
import re
from datetime import datetime


def _parse_year(creation_date: str) -> str:
    """
    PyMuPDF returns dates in format: 'D:20210315120000+00'00''
    We extract the 4-digit year from position 2:6.
    """
    if not creation_date:
        return "n.d."
    match = re.search(r"D:(\d{4})", creation_date)
    if match:
        return match.group(1)
    return "n.d."


def _clean_metadata_field(value: str) -> str:
    """Strip null bytes and extra whitespace that PyMuPDF sometimes includes."""
    if not value:
        return ""
    return value.replace("\x00", "").strip()


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract text and metadata from a PDF file.
    Returns a structured dictionary with enriched metadata.
    """
    doc = fitz.open(pdf_path)

    full_text = ""
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page_number": page_num + 1,
            "text": text,
            "char_count": len(text)
        })
        full_text += f"\n[PAGE {page_num + 1}]\n{text}"

    raw_meta = doc.metadata

    # Build a clean, structured metadata dict for downstream use
    structured_metadata = {
        "title":    _clean_metadata_field(raw_meta.get("title", "")),
        "author":   _clean_metadata_field(raw_meta.get("author", "")),
        "subject":  _clean_metadata_field(raw_meta.get("subject", "")),
        "year":     _parse_year(raw_meta.get("creationDate", "")),
        "producer": _clean_metadata_field(raw_meta.get("producer", "")),
        "raw":      raw_meta,
    }

    return {
        "text": full_text,
        "metadata": structured_metadata,
        "pages": pages,
        "total_pages": len(doc)
    }
