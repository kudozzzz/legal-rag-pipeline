#FAISS-backed vector store with parallel metadata list

import pickle
from pathlib import Path

import faiss
import numpy as np


_FAISS_SUFFIX = ".faiss"
_META_SUFFIX = ".meta"


class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []

    def add(self, chunks: list[dict]) -> None:
        """Add embedded chunks. Each chunk must have an 'embedding' key."""
        vecs = np.stack([c["embedding"] for c in chunks]).astype("float32")
        self.index.add(vecs)
        for c in chunks:
            self.metadata.append(
                {
                    "doc_name": c["doc_name"],
                    "page_number": c["page_number"],
                    "chunk_index": c["chunk_index"],
                    "text": c["text"],
                }
            )

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """Return top-k results as list of {meta, score} dicts."""
        vec = query_embedding.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"meta": self.metadata[idx], "score": float(score)})
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        faiss.write_index(self.index, str(path) + _FAISS_SUFFIX)
        with open(str(path) + _META_SUFFIX, "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        index = faiss.read_index(str(path) + _FAISS_SUFFIX)
        with open(str(path) + _META_SUFFIX, "rb") as f:
            metadata = pickle.load(f)

        store = cls.__new__(cls)
        store.index = index
        store.metadata = metadata
        return store
