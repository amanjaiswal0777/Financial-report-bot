"""Step 5: retrieval = semantic search + optional BM25 hybrid + optional reranking.

Flow when config.USE_HYBRID and config.USE_RERANKER are both True (v3):
    1. Semantic search in Chroma           -> SEMANTIC_K candidates (ranked list A)
    2. BM25 keyword search over all chunks -> BM25_K candidates     (ranked list B)
    3. Reciprocal Rank Fusion merges A and B, keep the best FETCH_K
    4. Cross-encoder reranks that pool, keep the best TOP_K

Other switches:
    USE_HYBRID=False                 -> exactly v0 / v2 behaviour (reproducible)
    USE_HYBRID=True, USE_RERANKER=False -> top K by RRF score (ablation)

Each hybrid candidate carries `semantic_rank` and `bm25_rank` in its metadata
(None = not found by that retriever) so you can see who found what.
"""
import re
from functools import lru_cache

from langchain_core.documents import Document

import config
from src.vectorstore import load_vectorstore

# ---------------------------------------------------------------- tokenizer
# Keeps numbers intact ("35,620", "18.96"), drops brackets/punctuation/currency
# symbols, no stemming (so "tax" and "taxes" stay distinct line items).
_TOKEN_RE = re.compile(r"\d+(?:,\d+)*(?:\.\d+)?|[a-z][a-z0-9]*")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "how", "in", "is", "it", "its", "of", "on",
    "or", "s", "that", "the", "to", "was", "were", "what", "which", "who",
    "why", "with",
}


def tokenize(text: str) -> list:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


# ---------------------------------------------------------------- BM25 leg
@lru_cache(maxsize=1)
def get_bm25():
    """Build a BM25 index from the exact chunks stored in Chroma (cached per process)."""
    from rank_bm25 import BM25Okapi

    data = load_vectorstore().get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=dict(meta or {}))
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
    if not docs:
        raise RuntimeError("Chroma index is empty. Run: python ingest.py first.")
    return BM25Okapi([tokenize(d.page_content) for d in docs]), docs


def bm25_search(query: str, k: int):
    """Return up to k of (Document, bm25_score), best first. Zero-score chunks are dropped."""
    index, docs = get_bm25()
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = index.get_scores(tokens)
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
    return [(docs[i], float(scores[i])) for i in order if scores[i] > 0]


# ---------------------------------------------------------------- fusion
def _key(doc):
    return (doc.metadata.get("page"), doc.page_content)


def reciprocal_rank_fusion(semantic, keyword, rrf_k: int):
    """Merge two ranked lists of (Document, score) using ranks only.

    Each chunk gets 1/(rrf_k + rank) from every list it appears in; scores are
    summed. Chunks are deduplicated by (page, text). Returns (Document, rrf_score)
    sorted best first, with semantic_rank / bm25_rank added to the metadata.
    """
    fused = {}
    for tag, ranked in (("semantic_rank", semantic), ("bm25_rank", keyword)):
        for rank, (doc, _) in enumerate(ranked, start=1):
            entry = fused.setdefault(_key(doc), {"doc": doc, "score": 0.0, "ranks": {}})
            entry["score"] += 1.0 / (rrf_k + rank)
            entry["ranks"][tag] = rank

    merged = []
    for entry in sorted(fused.values(), key=lambda e: e["score"], reverse=True):
        base = entry["doc"]
        meta = dict(base.metadata)
        meta["semantic_rank"] = entry["ranks"].get("semantic_rank")
        meta["bm25_rank"] = entry["ranks"].get("bm25_rank")
        merged.append((Document(page_content=base.page_content, metadata=meta), entry["score"]))
    return merged


def hybrid_candidates(store, query: str, pool: int):
    semantic = store.similarity_search_with_score(query, k=config.SEMANTIC_K)
    keyword = bm25_search(query, k=config.BM25_K)
    return reciprocal_rank_fusion(semantic, keyword, config.RRF_K)[:pool]


# ---------------------------------------------------------------- reranker
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
    """results: list of (Document, score). Returns top_k of (Document, rerank_score)."""
    if not results:
        return []
    docs = [doc for doc, _ in results]
    pairs = [(query, doc.page_content) for doc in docs]
    scores = get_reranker().predict(pairs)  # higher = more relevant
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:top_k]]


# ---------------------------------------------------------------- entry point
def retrieve(query: str, k: int = config.TOP_K):
    """Return a list of (Document, score).

    Score meaning depends on the mode:
        plain semantic            : cosine distance, lower = more similar
        hybrid, no reranker       : RRF score, higher = better
        any mode with reranker    : cross-encoder score, higher = better
    """
    store = load_vectorstore()
    pool = max(config.FETCH_K, k)

    if config.USE_HYBRID:
        candidates = hybrid_candidates(store, query, pool)
        if config.USE_RERANKER:
            return rerank(query, candidates, top_k=k)
        return candidates[:k]

    if not config.USE_RERANKER:
        return store.similarity_search_with_score(query, k=k)

    candidates = store.similarity_search_with_score(query, k=pool)
    return rerank(query, candidates, top_k=k)
