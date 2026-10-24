"""Glue: question -> retrieve -> generate. The eval script will call this later."""
from src.generator import generate_answer
from src.retriever import retrieve


def answer(question: str) -> dict:
    results = retrieve(question)
    docs = [doc for doc, _ in results]
    return {
        "question": question,
        "answer": generate_answer(question, docs),
        "chunks": [
            {
                 "page": doc.metadata.get("page"),
                "source": doc.metadata.get("source"),
                "distance": round(float(score), 4),
                "semantic_rank": doc.metadata.get("semantic_rank"),
                "bm25_rank": doc.metadata.get("bm25_rank"),
                "text": doc.page_content,
            }
            for doc, score in results
        ],
    }
