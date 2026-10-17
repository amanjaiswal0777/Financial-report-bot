"""Step 3: embedding model (local sentence-transformers, no API key needed)."""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

import config


@lru_cache(maxsize=1)
def get_embeddings():
    # Prefer a complete local snapshot when available. Passing the resolved
    # directory prevents Transformers from probing the Hub for optional files
    # (which can fail in otherwise fully offline runs).
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        model_name = snapshot_download(
            repo_id=config.EMBEDDING_MODEL,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        # Preserve the normal first-run behavior: download by model ID.
        model_name = config.EMBEDDING_MODEL

    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
