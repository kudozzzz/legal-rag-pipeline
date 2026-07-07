# Use overlap so clauses split across chunk boundaries remain retrievable

from typing import Generator


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 100,
) -> Generator[tuple[str, int], None, None]:

    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end], idx
        start += chunk_size - overlap
        idx += 1


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 512,
    overlap: int = 100,
) -> list[dict]:

    chunks = []
    for page in pages:
        for chunk_text_value, chunk_idx in chunk_text(
            page["text"], chunk_size, overlap
        ):
            chunks.append(
                {
                    "doc_name": page["doc_name"],
                    "page_number": page["page_number"],
                    "chunk_index": chunk_idx,
                    "text": chunk_text_value,
                }
            )
    return chunks
