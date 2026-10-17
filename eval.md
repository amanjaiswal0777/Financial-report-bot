# Evaluation Log: Financial-report-bot

## What this file holds

This file is the **growth log of the RAG application**. For every version we record:

1. **What changed** (the configuration / pipeline at that version)
2. **What the results were** (DeepEval scores on the same golden set)
3. **What we learned** and what to try next

Every version is scored on the **same golden dataset** (`eval/golden_dataset.json`, 15 questions) with the **same judge model**, so numbers are directly comparable across versions.

Raw per-question results live in `eval/results/<version>.json`. This file is the human-readable summary.

---

## Version overview

| Version | Idea | Status | Recall | Precision | Faithfulness | Answer Relevancy | Page Hit |
|---|---|---|---|---|---|---|---|
| `v0_naive` | Fixed-size chunks + plain semantic search | Done | 0.800 | 0.460 | 0.933 | 0.467 | 0.467 |
| `v1_...` | _next experiment_ | Planned | | | | | |

> Regenerate this table at any time with `python eval/compare.py`.

---

## v0_naive (baseline)

**Run date:** 2026-10-16 10:43
**Document:** HCLTech quarterly results PDF (quarter ended September 30, 2026)
**Goal:** a frozen baseline. Nothing is tuned. Every later version must beat these numbers.

### Configuration

| Area | Setting | Value |
|---|---|---|
| Loader | PDF reader | `PyPDFLoader` (page-level, 1-based page numbers) |
| Loader | OCR fallback | RapidOCR when a page has < 80 characters of native text (zoom 2.5) |
| Chunking | Splitter | `RecursiveCharacterTextSplitter` |
| Chunking | `CHUNK_SIZE` | 1000 characters |
| Chunking | `CHUNK_OVERLAP` | 200 characters |
| Embeddings | Model | `BAAI/bge-small-en-v1.5` (local, normalized) |
| Vector DB | Store | Chroma, collection `naive_rag`, cosine distance |
| Retrieval | Method | Plain semantic similarity search |
| Retrieval | `TOP_K` | 8 |
| Retrieval | Extras | None (no reranker, no hybrid/BM25, no query rewriting) |
| Generation | LLM | `gpt-4o-mini`, temperature 0 |
| Generation | Prompt rules | Use only context, max two short sentences, cite page numbers, say "I couldn't find this in the retrieved document text." if unsupported |
| Evaluation | Judge model | `gpt-4o-mini` |
| Evaluation | Pass threshold | 0.5 |
| Evaluation | Questions | 15 |

### Results

| Metric | Mean | Pass rate | What it measures |
|---|---|---|---|
| Contextual Recall | 0.800 | 0.800 | Did the retrieved chunks contain the info needed for the expected answer? |
| Contextual Precision | 0.460 | 0.467 | Are the relevant chunks ranked near the top, with little noise? |
| Faithfulness | 0.933 | 1.000 | Is the answer supported by the retrieved context (no hallucination)? |
| Answer Relevancy | 0.467 | 0.600 | Does the answer actually address the question? |
| Page Hit Rate | 0.467 | n/a | Did retrieval return at least one expected source page? (7 of 15) |

**Reading the numbers:** the model is honest (faithfulness is high) but often has nothing useful to work with. Retrieval is the bottleneck, not generation.

### Answer outcomes (15 questions)

| Outcome | Count | Question IDs |
|---|---|---|
| Fully correct | 5 | 2, 5, 8, 11, 15 |
| Partially correct (missing part of the answer) | 4 | 1, 3, 6, 12 |
| "Couldn't find this" | 6 | 4, 7, 9, 10, 13, 14 |

### Per-question breakdown

| ID | Type | Expected page | Page hit | Recall | Precision | Faithful | Relevancy | Outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | factual | 2 | Yes | 1.00 | 0.82 | 1.0 | 0.5 | Partial (revenue found, profit missing) |
| 2 | calculation | 2 | No | 0.00 | 0.27 | 1.0 | 0.5 | Correct |
| 3 | subjective | 2 | No | 1.00 | 1.00 | 0.5 | 1.0 | Partial (only 2 expense items) |
| 4 | comparison + calc | 2 | No | 1.00 | 0.00 | 1.0 | 0.0 | Not found |
| 5 | subjective | 1 | Yes | 1.00 | 1.00 | 1.0 | 1.0 | Correct |
| 6 | factual | 1 | Yes | 1.00 | 0.70 | 0.5 | 0.5 | Partial (record date missing) |
| 7 | calculation | 2 | No | 0.00 | 0.00 | 1.0 | 0.0 | Not found |
| 8 | comparison | 3 | Yes | 1.00 | 0.73 | 1.0 | 1.0 | Correct |
| 9 | factual | 2 | No | 1.00 | 0.00 | 1.0 | 0.0 | Not found |
| 10 | factual | 2 | No | 1.00 | 0.00 | 1.0 | 0.0 | Not found |
| 11 | subjective | 10 | Yes | 1.00 | 1.00 | 1.0 | 1.0 | Correct |
| 12 | subjective | 10 | No | 1.00 | 0.00 | 1.0 | 0.5 | Partial (only "Ind AS") |
| 13 | subjective | 17 | No | 0.00 | 0.13 | 1.0 | 0.0 | Not found |
| 14 | factual | 2 | Yes | 1.00 | 0.27 | 1.0 | 0.0 | Not found (page 2 retrieved but still failed) |
| 15 | calculation | 1 | Yes | 1.00 | 1.00 | 1.0 | 1.0 | Correct |

### Findings

- **Page 2 is almost never retrieved.** Page 2 holds the income statement (profit, tax, EPS, comprehensive income, total expenses). 8 questions expect it, but it was retrieved for only 2 of them (Q1, Q14). That alone explains most of the failures.
- **The same pages keep coming back** (3, 4, 6, 8, 21, 22, 24, 26) regardless of the question. Pure semantic search over similar-looking financial table text returns near-duplicate numeric chunks instead of the right one.
- **Q14 shows a chunking problem too.** Page 2 was retrieved, yet the answer was still "couldn't find". The statement table is probably split across chunks so the label and value no longer sit together.
- **Faithfulness is high (0.933).** The prompt's "do not guess" rule works. The system fails safe rather than hallucinating.
- **Metric noise exists.** Q2 scored recall 0.0 but the answer was correct (the numbers are inside the question itself). Q4, Q9, Q10 scored recall 1.0 even though the right page was missed. Treat per-question scores as indicative and trust the aggregate trend plus page hit rate.
- **Page hit rate is the most reliable retrieval signal here**, because it is a direct check against `source_page` and does not depend on an LLM judge.
- **Golden set note:** question type labels (e.g. Q3, Q5 marked `subjective`) are loose. Worth cleaning up before the set is used for deeper analysis.

### Targets for the next versions

| Metric | v0 baseline | Suggested target |
|---|---|---|
| Page Hit Rate | 0.467 | 0.80 or higher |
| Contextual Precision | 0.460 | 0.70 or higher |
| Answer Relevancy | 0.467 | 0.75 or higher |
| Faithfulness | 0.933 | Keep at 0.90 or higher |

---

## Ideas for upcoming versions

These are candidates, not commitments. Change **one thing per version** so the effect is clear.

| Version | Change | Why |
|---|---|---|
| `v1_hybrid` | Add BM25 keyword search alongside semantic search | Exact terms like "Profit before tax" and "Basic EPS" should match lexically |
| `v2_rerank` | Retrieve more (e.g. 20) then rerank down to 8 | Pushes the right chunk to the top, should lift precision |
| `v3_table_chunking` | Table-aware chunking, or keep each page's statement together | Fixes label/value splits like Q14 |
| `v4_query_rewrite` | Rewrite the question before retrieval | Helps long, calculation-style questions |

---

## How to run and record a new version

```bash
# 1. Change config.py / the relevant src/ file for the new idea
# 2. Rebuild the index if chunking or embeddings changed
python ingest.py data/your_report.pdf

# 3. Smoke test (3 questions), then full run
python eval/run_eval.py --version v1_hybrid --limit 3
python eval/run_eval.py --version v1_hybrid

# 4. Print the comparison table across all versions
python eval/compare.py
```

Then copy the section template below into this file.

### Template for a new version

```markdown
## vN_name

**Run date:** YYYY-MM-DD HH:MM
**What changed vs previous version:** one sentence

### Configuration
(table of settings, mark changed rows with **bold**)

### Results
| Metric | Mean | Pass rate | Change vs previous |
|---|---|---|---|

### Findings
- what improved
- what got worse
- what to try next
```
