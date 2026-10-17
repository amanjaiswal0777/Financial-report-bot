"""Step 4: store / load chunks in a persistent Chroma vector database."""
import shutil

from langchain_chroma import Chroma

import config
from src.embeddings import get_embeddings

_COSINE = {"hnsw:space": "cosine"}


def build_vectorstore(chunks):
    """Create a fresh index (wipes the old one so re-ingesting never duplicates)."""
    embeddings = get_embeddings()
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata=_COSINE,
    )


def load_vectorstore():
    if not config.CHROMA_DIR.exists():
        raise FileNotFoundError("No index found. Run: python ingest.py first.")
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata=_COSINE,
    )
