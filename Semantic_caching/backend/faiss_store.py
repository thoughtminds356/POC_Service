import logging
from typing import Optional, Tuple

import numpy as np
import faiss

logger = logging.getLogger("semantic_cache.faiss_store")


class FaissStore:
    """FAISS store using IndexIDMap over IndexFlatIP with explicit int64 ids."""

    def __init__(self) -> None:
        self.index: Optional[faiss.IndexIDMap] = None
        self.dim: Optional[int] = None
        self.next_id: int = 0

    def ensure_index(self, dim: int) -> None:
        if self.index is None:
            logger.info(f"Initializing FAISS IndexIDMap(FlatIP) with dim={dim}")
            base = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIDMap(base)
            self.dim = dim

    def add_vectors(self, vecs: np.ndarray, ids: Optional[np.ndarray] = None) -> np.ndarray:
        assert vecs.ndim == 2, "vecs must be (n, d)"
        self.ensure_index(vecs.shape[1])
        if ids is None:
            ids = np.arange(self.next_id, self.next_id + vecs.shape[0], dtype=np.int64)
        else:
            ids = ids.astype(np.int64)
        self.index.add_with_ids(vecs, ids)
        self.next_id = int(ids.max()) + 1
        return ids

    def search(self, vec: np.ndarray, top_k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        assert self.index is not None, "Index is not initialized"
        return self.index.search(vec, top_k)