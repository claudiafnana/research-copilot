import fitz

def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract text and metadata from a PDF file.
    Returns structured dictionary.
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

    metadata = doc.metadata

    return {
        "text": full_text,
        "metadata": metadata,
        "pages": pages,
        "total_pages": len(doc)
    }