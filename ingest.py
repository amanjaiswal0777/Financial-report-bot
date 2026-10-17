"""Build the vector index from a PDF.

Usage:
    python ingest.py data/your_report.pdf
    python ingest.py            # uses the first PDF found in data/
"""
import argparse
from pathlib import Path

import config
from src.chunker import split_documents
from src.loader import load_pdf
from src.vectorstore import build_vectorstore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", help="path to the PDF")
    args = parser.parse_args()

    if args.pdf:
        pdf = Path(args.pdf)
    else:
        pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
        if not pdfs:
            raise SystemExit("Put a PDF in the data/ folder first.")
        pdf = pdfs[0]

    print(f"Loading {pdf.name} ...")
    pages = load_pdf(pdf)
    print(f"  {len(pages)} pages")

    chunks = split_documents(pages)
    print(f"  {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    print("Embedding and indexing (first run downloads the embedding model) ...")
    build_vectorstore(chunks)
    print(f"Done. Index saved in {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
