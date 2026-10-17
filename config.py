"""All settings live here so every experiment is one-line change.

Baseline (v0) = naive RAG: fixed-size chunks + semantic search only.
Keep these values frozen for the baseline, then change them in later versions.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "naive_rag"

# Embeddings (runs locally, free)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 8

# Reranking (v1): fetch a wide candidate pool, rerank, keep TOP_K.
# Set USE_RERANKER = False to reproduce the v0 naive baseline exactly.
USE_RERANKER = True
RERANKER_MODEL = "BAAI/bge-reranker-base"
FETCH_K = 30

# LLM (set in .env)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 0
