"""Step 5: semantic search + optional cross-encoder reranking (v1).

Flow when config.USE_RERANKER is True:
    1. Fetch FETCH_K candidates from Chroma (bi-encoder, fast but coarse).
    2. Score every (query, chunk) pair with a cross-encoder (slow but accurate).
    3. Keep the best TOP_K by rerank score.

When USE_RERANKER is False this behaves exactly like v0 (plain semantic search).
"""
from functools import lru_cache

import config
from src.vectorstore import load_vectorstore


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    # Same offline-friendly pattern as src/embeddings.py: prefer a complete
    # local snapshot, otherwise download by model ID on first run.
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            model_name = snapshot_download(
                repo_id=config.RERANKER_MODEL,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            model_name = config.RERANKER_MODEL
    except ImportError:
        model_name = config.RERANKER_MODEL

    return CrossEncoder(model_name, max_length=512)


def rerank(query: str, results, top_k: int):
    """results: list of (Document, distance). Returns top_k of (Document, rerank_score)."""
    if not results:
        return []
    docs = [doc for doc, _ in results]
    pairs = [(query, doc.page_content) for doc in docs]
    scores = get_reranker().predict(pairs)  # higher = more relevant
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:top_k]]


def retrieve(query: str, k: int = config.TOP_K):
    """Return a list of (Document, score).

    v0 (no reranker): score is cosine distance, lower = more similar.
    v1 (reranker):    score is rerank relevance, higher = more relevant.
    """
    store = load_vectorstore()
    if not config.USE_RERANKER:
        return store.similarity_search_with_score(query, k=k)

    candidates = store.similarity_search_with_score(query, k=max(config.FETCH_K, k))
    return rerank(query, candidates, top_k=k)
