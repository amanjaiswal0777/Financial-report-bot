"""Ask questions about the indexed PDF.

Usage:
    python ask.py "What was the total revenue in FY24?"
    python ask.py "..." --show-chunks     # also print retrieved chunks (for debugging)
    python ask.py                         # interactive mode
"""
import argparse

from src.pipeline import answer


def run(question: str, show_chunks: bool):
    result = answer(question)
    print("\nANSWER:\n" + result["answer"])
    pages = sorted({c["page"] for c in result["chunks"]})
    print(f"\nRetrieved from pages: {pages}")
    if show_chunks:
        for i, c in enumerate(result["chunks"], 1):
            print(f"\n--- chunk {i} | page {c['page']} | score {c['distance']} "
      f"| semantic rank {c.get('semantic_rank')} | bm25 rank {c.get('bm25_rank')} ---")
            print(c["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--show-chunks", action="store_true")
    args = parser.parse_args()

    if args.question:
        run(args.question, args.show_chunks)
        return

    print("Interactive mode. Type 'exit' to quit.")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        run(q, args.show_chunks)


if __name__ == "__main__":
    main()
