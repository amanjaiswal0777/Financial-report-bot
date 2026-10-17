"""Step 2: split pages into fixed-size overlapping chunks (naive chunking)."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)
