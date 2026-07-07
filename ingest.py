#Ingests PDF files and extracts raw text with page-level metadata


import fitz  # PyMuPDF
from pathlib import Path
from typing import Generator

def extract_pages(pdf_path: str | Path) -> Generator[dict, None, None]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    doc_name = pdf_path.name

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            yield {
                "doc_name": doc_name,
                "page_number": page_num,
                "text": text,
            }

    doc.close()


def ingest_directory(pdf_dir: str | Path) -> list[dict]:
    pages = []
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        pages.extend(extract_pages(pdf_path))
    return pages
