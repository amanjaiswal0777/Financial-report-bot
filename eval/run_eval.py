"""Run the golden set through the RAG pipeline and score it with DeepEval.

Usage (from project root):
    python eval/run_eval.py --version v0_naive --limit 3     # cheap smoke test
    python eval/run_eval.py --version v0_naive               # full run
    python eval/run_eval.py --version v0_naive --reasons     # also ask judge for reasons
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import config  # noqa: E402  (also loads .env)
from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase  # noqa: E402
from src.pipeline import answer  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "results"
METRIC_NAMES = [
    "contextual_recall",
    "contextual_precision",
    "faithfulness",
    "answer_relevancy",
]


def build_metrics(judge: str, threshold: float, reasons: bool) -> dict:
    kw = dict(threshold=threshold, model=judge, include_reason=reasons)
    return {
        "contextual_recall": ContextualRecallMetric(**kw),
        "contextual_precision": ContextualPrecisionMetric(**kw),
        "faithfulness": FaithfulnessMetric(**kw),
        "answer_relevancy": AnswerRelevancyMetric(**kw),
    }


def expected_answer(item: dict) -> str:
    return (
        item.get("ground_truth")
        or item.get("answer")
        or item.get("expected_output")
        or ""
    )


def question_text(item: dict) -> str:
    question = item.get("question") or item.get("input")
    if not question:
        raise ValueError("Each golden-set item must include 'question' or 'input'.")
    return question


def expected_pages(item: dict) -> list:
    if "source_pages" in item:
        return list(item["source_pages"])
    page = item.get("source_page")
    if page is None:
        metadata = item.get("additional_metadata") or {}
        if "source_pages" in metadata:
            return list(metadata["source_pages"])
        page = metadata.get("source_page")
    return [page] if page is not None else []


def summarize(rows: list) -> dict:
    summary = {}
    for name in METRIC_NAMES:
        scores = [r["metrics"][name]["score"] for r in rows if r["metrics"][name].get("score") is not None]
        passed = [r["metrics"][name]["passed"] for r in rows if r["metrics"][name].get("passed") is not None]
        summary[name] = {
            "mean": round(mean(scores), 3) if scores else None,
            "pass_rate": round(sum(passed) / len(passed), 3) if passed else None,
        }
    hits = [r["page_hit"] for r in rows if r["page_hit"] is not None]
    summary["page_hit_rate"] = round(sum(hits) / len(hits), 3) if hits else None
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="label, e.g. v0_naive, v1_hybrid")
    parser.add_argument("--golden", default=str(ROOT / "eval" / "golden_dataset.json"))
    parser.add_argument("--judge", default=os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None, help="only first N questions")
    parser.add_argument("--reasons", action="store_true", help="judge explains each score (more tokens)")
    args = parser.parse_args()

    with open(args.golden, encoding="utf-8") as golden_file:
        golden = json.load(golden_file)
    if args.limit:
        golden = golden[: args.limit]

    rows = []
    for i, item in enumerate(golden, 1):
        t0 = time.time()
        question = question_text(item)
        out = answer(question)
        retrieval_context = [c["text"] for c in out["chunks"]]
        retrieved_pages = sorted({c["page"] for c in out["chunks"]})

        test_case = LLMTestCase(
            input=question,
            actual_output=out["answer"],
            expected_output=expected_answer(item),
            retrieval_context=retrieval_context,
        )

        metrics = {}
        for name, metric in build_metrics(args.judge, args.threshold, args.reasons).items():
            try:
                metric.measure(test_case)
                metrics[name] = {
                    "score": metric.score,
                    "passed": metric.is_successful(),
                    "reason": metric.reason,
                }
            except Exception as e:  # keep going if one metric fails
                metrics[name] = {"score": None, "passed": None, "error": str(e)}

        pages = expected_pages(item)
        page_hit = any(p in retrieved_pages for p in pages) if pages else None

        rows.append(
            {
                "id": item.get("id", (item.get("additional_metadata") or {}).get("id")),
                "question": question,
                "expected": expected_answer(item),
                "actual": out["answer"],
                "expected_pages": pages,
                "retrieved_pages": retrieved_pages,
                "page_hit": page_hit,
                "metrics": metrics,
            }
        )
        short = {k: (round(v["score"], 2) if v.get("score") is not None else "ERR") for k, v in metrics.items()}
        print(f"[{i}/{len(golden)}] {time.time() - t0:.0f}s page_hit={page_hit} {short}")

    result = {
        "version": args.version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": config.TOP_K,
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model": config.LLM_MODEL,
            "judge_model": args.judge,
            "threshold": args.threshold,
            "n_questions": len(rows),
        },
        "summary": summarize(rows),
        "results": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.version}.json"
    json.dump(result, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\n=== {args.version} ({len(rows)} questions, judge={args.judge}) ===")
    print(f"{'metric':<22}{'mean':>8}{'pass rate':>12}")
    for name in METRIC_NAMES:
        s = result["summary"][name]
        print(f"{name:<22}{str(s['mean']):>8}{str(s['pass_rate']):>12}")
    print(f"{'page_hit_rate':<22}{str(result['summary']['page_hit_rate']):>8}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
