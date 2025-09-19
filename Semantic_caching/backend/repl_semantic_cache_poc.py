"""
Semantic Cache POC (Terminal REPL) - Simple version

- Uses SentenceTransformer (all-MiniLM-L12-v2) to embed text queries
- Stores embeddings in FAISS (IndexFlatIP) using cosine similarity (via L2-normalized vectors)
- Seeds with a few dummy query/response pairs
- On each user query:
  - Search FAISS for nearest neighbor
  - If similarity > 0.85: cache hit -> return cached response
  - Else: cache miss -> simulate Pinecone/LLM backend, create dummy response, insert into cache
- Runs in a simple REPL loop (type 'exit' or 'quit' to stop)

This file is intentionally minimal and self-contained with clear comments.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Dict
import json
import os

import numpy as np
import faiss  # provided by the 'faiss-cpu' pip package
from sentence_transformers import SentenceTransformer


# ----- Logging setup -----
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("semantic_cache_poc")


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


class SemanticCache:
    """Semantic cache orchestrator tying embeddings and FAISS store together."""

    def __init__(self, embedder: EmbeddingsProvider, store: FaissStore, similarity_threshold: float = 0.85, persist_path: Optional[str] = None) -> None:
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        self.embedder = embedder
        self.store = store
        self.similarity_threshold = similarity_threshold
        self.metadata: Dict[int, Tuple[str, str]] = {}
        # Persist pairs (query/response) to JSON so cache grows across runs
        self.persist_path = persist_path or os.path.join(os.path.dirname(__file__), "semantic_cache_pairs.json")

    @staticmethod
    def default_seed_pairs() -> List[Tuple[str, str]]:
        return [
            ("What is the capital of France?", "The capital of France is Paris."),
            ("Explain cosine similarity.", "Cosine similarity measures the cosine of the angle between two vectors, indicating orientation similarity."),
            ("Best way to boil eggs?", "Place eggs in cold water, bring to a boil, then simmer 9-12 minutes and cool in ice bath."),
            ("What is FAISS?", "FAISS is a library by Facebook AI for efficient similarity search and clustering of dense vectors."),
            ("Who's Ada Lovelace?", "Ada Lovelace was a 19th-century mathematician often considered the first computer programmer."),
        ]

    def seed(self, pairs: Optional[List[Tuple[str, str]]] = None) -> None:
        # Try to load previously persisted pairs; fall back to provided/default pairs
        persisted = self._load_persisted_pairs()
        pairs = persisted or (pairs or self.default_seed_pairs())
        if not pairs:
            return
        vecs = self.embedder.embed_batch([q for q, _ in pairs])
        ids = self.store.add_vectors(vecs)
        for i, (q, r) in enumerate(pairs):
            self.metadata[int(ids[i])] = (q, r)
        logger.info(f"Seeded cache with {len(pairs)} entries.")
        # Ensure current state is saved (creates file on first run)
        self._save_all_pairs()

    def insert(self, query: str, response: str, pre_vec: Optional[np.ndarray] = None) -> int:
        vec = pre_vec if pre_vec is not None else self.embedder.embed(query)
        ids = self.store.add_vectors(vec)
        new_id = int(ids[0])
        self.metadata[new_id] = (query, response)
        return new_id

    def search(self, query: str, top_k: int = 1) -> Tuple[bool, float, Optional[str], str]:
        if not query or not query.strip():
            return (False, 0.0, None, "Query is empty.")
        q_vec = self.embedder.embed(query)
        D, I = self.store.search(q_vec, top_k)
        best_sim = float(D[0, 0])
        best_id = int(I[0, 0])
        if best_id >= 0 and best_sim > self.similarity_threshold:
            cached_query, cached_response = self.metadata.get(best_id, (None, None))
            return (True, best_sim, cached_query, cached_response)
        # Miss: simulate backend and insert
        logger.info("Cache miss -> Searching Pinecone/LLM backend...")
        simulated_response = f"[BACKEND] Dummy response for: '{query}'"
        self.insert(query, simulated_response, pre_vec=q_vec)
        # Save updated pairs so they persist across runs
        self._save_all_pairs()
        return (False, best_sim, None, simulated_response)

    def print_cache(self) -> None:
        if not self.metadata:
            print("No cached entries.")
            return
        print("\n--- Cached Entries ---")
        for i in sorted(self.metadata.keys()):
            q, r = self.metadata[i]
            print(f"[{i}] Q: {q}")
            print(f"    A: {r}\n")
        print("----------------------\n")

    # ----- Persistence helpers -----
    def _load_persisted_pairs(self) -> List[Tuple[str, str]]:
        path = self.persist_path
        try:
            if not path or not os.path.isfile(path):
                return []
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pairs: List[Tuple[str, str]] = []
            for item in data if isinstance(data, list) else []:
                q = item.get("query") if isinstance(item, dict) else None
                r = item.get("response") if isinstance(item, dict) else None
                if isinstance(q, str) and isinstance(r, str):
                    pairs.append((q, r))
            return pairs
        except Exception as e:
            logger.error(f"Failed to load persisted pairs from {path}: {e}")
            return []

    def _save_all_pairs(self) -> None:
        path = self.persist_path
        try:
            if not path:
                return
            items = [
                {"query": self.metadata[i][0], "response": self.metadata[i][1]}
                for i in sorted(self.metadata.keys())
            ]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save pairs to {path}: {e}")


def run_repl() -> None:
    embedder = EmbeddingsProvider()
    store = FaissStore()
    cache = SemanticCache(embedder, store, similarity_threshold=0.85)
    cache.seed()

    # Show current cache at startup
    cache.print_cache()

    print("Type your query (or 'exit'/'quit' to stop).\n")

    while True:
        try:
            user_query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_query:
            continue

        hit, sim, cached_q, response = cache.search(user_query, top_k=1)
        if hit:
            print("\n[HIT] Similarity: {:.4f} (> {:.2f})".format(sim, cache.similarity_threshold))
            print(f"Cached Query   : {cached_q}")
            print(f"Your Query     : {user_query}")
            print(f"Cached Response: {response}\n")
        else:
            print("\n[MISS] Best similarity: {:.4f} (<= {:.2f})".format(sim, cache.similarity_threshold))
            print("Searching Pinecone... (simulated)")
            print(f"New Query      : {user_query}")
            print(f"New Response   : {response}\n")

        # Optionally show updated cache after each query
        cache.print_cache()


if __name__ == "__main__":
    run_repl()
