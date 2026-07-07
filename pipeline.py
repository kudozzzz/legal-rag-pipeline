#End-to-end RAG pipeline: ingest -> chunk -> embed -> index -> query

from pathlib import Path

import numpy as np

from ingest import ingest_directory
from chunking import chunk_pages
from embeddings import load_model, embed_chunks
from vector_store import VectorStore

_LOW_CONFIDENCE_THRESHOLD = 0.30
_LOW_CONFIDENCE_ANSWER = (
    "The available documents do not contain sufficiently relevant information "
    "to answer this question with confidence."
)


class RAGPipeline:
    def __init__(
        self,
        pdf_dir: str | Path,
        index_path: str | Path | None = None,
        top_k: int = 3,
    ):
        self.top_k = top_k
        self.model = load_model()

        if index_path and (Path(str(index_path) + ".faiss")).exists():
            self.store = VectorStore.load(index_path)
        else:
            self.store = self._build(pdf_dir, index_path)

    def _build(
        self, pdf_dir: str | Path, index_path: str | Path | None
    ) -> VectorStore:
        pages = ingest_directory(pdf_dir)
        if not pages:
            raise ValueError(f"No PDFs found in {pdf_dir}")

        chunks = chunk_pages(pages)
        chunks = embed_chunks(chunks, self.model)

        dim = len(chunks[0]["embedding"])
        store = VectorStore(dim=dim)
        store.add(chunks)

        if index_path:
            store.save(index_path)

        return store

    def query(self, question: str) -> dict:
        q_vec = self.model.encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        )[0]

        results = self.store.search(q_vec, top_k=self.top_k)

        if not results:
            return {"answer": _LOW_CONFIDENCE_ANSWER, "sources": [], "confidence": 0.0}


        scores = [r["score"] for r in results]
        confidence = float(np.clip(np.mean(scores), 0.0, 1.0))

        if confidence < _LOW_CONFIDENCE_THRESHOLD:
            return {
                "answer": _LOW_CONFIDENCE_ANSWER,
                "sources": _format_sources(results),
                "confidence": round(confidence, 4),
            }


        answer = "\n\n".join(r["meta"]["text"] for r in results)

        return {
            "answer": answer,
            "sources": _format_sources(results),
            "confidence": round(confidence, 4),
        }


def _format_sources(results: list[dict]) -> list[dict]:
    return [
        {
            "document": r["meta"]["doc_name"],
            "page": r["meta"]["page_number"],
            "chunk": r["meta"]["text"],
        }
        for r in results
    ]
