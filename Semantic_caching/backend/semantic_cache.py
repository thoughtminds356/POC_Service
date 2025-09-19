import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import json
import os
from embeddings_provider import EmbeddingsProvider
from faiss_store import FaissStore

logger = logging.getLogger("semantic_cache.core")


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
        print("Search D:", D)
        print("Search I:", I)
        best_sim = float(D[0, 0])
        best_id = int(I[0, 0])
        if best_id >= 0 and best_sim > self.similarity_threshold:
            cached_query, cached_response = self.metadata.get(best_id, (None, None))
            return (True, best_sim, cached_query, cached_response)
        # Miss: simulate backend and insert
        logger.info("Cache miss -> Searching Pinecone/LLM backend...")
        simulated_response = f"[BACKEND] Dummy response for: '{query}'" # Placeholder for response from pinecone
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
