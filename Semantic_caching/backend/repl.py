import logging

from embeddings_provider import EmbeddingsProvider
from faiss_store import FaissStore
from semantic_cache import SemanticCache


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("semantic_cache.repl")


def main() -> None:
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
    main()
