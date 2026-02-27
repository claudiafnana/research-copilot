from PyPDF2 import PdfReader

def extract_pages(path: str):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page_number": i, "text": text})
    return pages

if __name__ == "__main__":
    pdf_path = "papers/agroecologia_e_innovacion.pdf"
    pages = extract_pages(pdf_path)

    print("Total páginas:", len(pages))
    print("\n--- PÁGINA 1 (primeros 600 chars) ---\n")
    print(pages[0]["text"][:600])