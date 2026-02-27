import re

def clean_extracted_text(text: str) -> str:
    """
    Clean and normalize extracted PDF text:
    - collapse whitespace
    - fix hyphenation across line breaks
    - remove repeated page markers spacing issues
    """
    # Fix hyphenated words at line breaks: "infor-\n mation" -> "information"
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    # Normalize multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)

    # Keep [PAGE X] markers readable (optional)
    text = re.sub(r"\s*\[PAGE\s+(\d+)\]\s*", r"\n[PAGE \1]\n", text)

    return text.strip()