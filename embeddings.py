#Generates sentence embeddings for each chunk using sentence transformers


import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "BAAI/bge-small-en-v1.5" 


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> list[dict]:

    texts = [c["text"] for c in chunks]

    all_embeddings: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(vecs)

    for chunk, vec in zip(chunks, all_embeddings):
        chunk["embedding"] = vec

    return chunks
