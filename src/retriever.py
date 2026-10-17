"""Step 5: plain semantic search. No reranker, no hybrid, no query rewriting.

Later versions will replace/extend this file only.
"""
import config
from src.vectorstore import load_vectorstore


def retrieve(query: str, k: int = config.TOP_K):
    """Return a list of (Document, distance). Lower distance = more similar."""
    return load_vectorstore().similarity_search_with_score(query, k=k)
