import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("semantic_cache.embeddings")


class EmbeddingsProvider:
    """Wraps SentenceTransformer to produce L2-normalized float32 embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L12-v2") -> None:
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        if vec.ndim == 1:
            vec = vec[None, :]
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vec = (vec / norms).astype("float32")
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        # print("Norms before:", np.linalg.norm(vecs, axis=1))
        # print("Norms after:", np.linalg.norm(vecs / norms, axis=1))
        vecs = (vecs / norms).astype("float32")
        # print("Final vecs:", vecs)
        return vecs
