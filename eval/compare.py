"""Print a markdown table comparing all saved versions (paste into README).

Usage: python eval/compare.py
"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ROWS = ["contextual_recall", "contextual_precision", "faithfulness", "answer_relevancy", "page_hit_rate"]


def main():
    runs = [json.load(open(p, encoding="utf-8")) for p in RESULTS_DIR.glob("*.json")]
    runs.sort(key=lambda r: r["timestamp"])
    if not runs:
        raise SystemExit("No results yet. Run eval/run_eval.py first.")

    print("| Metric | " + " | ".join(r["version"] for r in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for name in ROWS:
        cells = []
        for r in runs:
            v = r["summary"][name]
            cells.append(str(v["mean"] if isinstance(v, dict) else v))
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
