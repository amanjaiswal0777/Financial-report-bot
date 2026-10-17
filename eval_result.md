# Evaluation Log: Financial-report-bot

## What this file holds

This file is the **growth log of the RAG application**. For every version we record:

1. **What changed** (the configuration / pipeline at that version)
2. **What the results were** (DeepEval scores on the same golden set)
3. **What we learned** and what to try next

Every version is scored on the **same golden dataset** (`eval/golden_dataset.json`, 15 questions) with the **same judge model** (`gpt-4o-mini`), so numbers are directly comparable across versions.

Raw per-question results live in `eval/results/<version>.json`. This file is the human-readable summary.

**Versions covered so far:** `v0_naive`, `version1_chunk_change`, `version_2_re_ranker`.

> **Important:** `version1_chunk_change` and `version_2_re_ranker` are each **one change on top of the v0 baseline**, not stacked on each other. The v2 result file shows chunking back at 1000 / 200, so v2 = v0 + reranker. Compare both against v0.

---

## Version overview

| Version | Idea | Chunk size | Overlap | Top-K | Recall | Precision | Faithfulness | Answer Relevancy | Page Hit |
|---|---|---|---|---|---|---|---|---|---|
| `v0_naive` | Fixed-size chunks + plain semantic search (baseline) | 1000 | 200 | 8 | 0.800 | 0.460 | 0.933 | 0.467 | 0.467 |
| `version1_chunk_change` | Bigger chunks, more overlap | **1300** | **500** | 8 | 0.833 | 0.508 | 0.800 | 0.733 | 0.667 |
| `version_2_re_ranker` | Cross-encoder reranker (fetch 30, keep best 8) | 1000 | 200 | 8 | 0.800 | 0.718 | 0.786 * | 0.678 | 0.600 |
| `v3_...` | _next experiment_ | | | | | | | | |

\* v2 faithfulness is averaged over 14 questions: the judge call for Q12 timed out (`RetryError ... TimeoutError`) and has no score.

Run dates: v0 = 2026-10-16 10:43, version1 = 2026-10-17 12:50, version2 = 2026-10-17 16:30.

> Regenerate the metric columns at any time with `python eval/compare.py`.

### Change vs v0 baseline

| Metric | v0 | version1 | Δ vs v0 | version2 | Δ vs v0 |
|---|---|---|---|---|---|
| Contextual Recall | 0.800 | 0.833 | +0.033 | 0.800 | 0.000 |
| Contextual Precision | 0.460 | 0.508 | +0.048 | 0.718 | **+0.258** |
| Faithfulness | 0.933 | 0.800 | -0.133 | 0.786 * | -0.147 |
| Answer Relevancy | 0.467 | 0.733 | **+0.266** | 0.678 | +0.211 |
| Page Hit Rate | 0.467 (7/15) | 0.667 (10/15) | **+0.200** | 0.600 (9/15) | +0.133 |

### Pass rates (threshold 0.5)

| Metric | v0 | version1 | version2 |
|---|---|---|---|
| Contextual Recall | 0.800 | 0.867 | 0.800 |
| Contextual Precision | 0.467 | 0.533 | 0.733 |
| Faithfulness | 1.000 | 0.800 | 0.857 * |
| Answer Relevancy | 0.600 | 0.733 | 0.733 |

### Progress against targets

| Metric | Target | v0 | version1 | version2 |
|---|---|---|---|---|
| Page Hit Rate | 0.80 or higher | 0.467 | 0.667 | 0.600 |
| Contextual Precision | 0.70 or higher | 0.460 | 0.508 | **0.718 (met)** |
| Answer Relevancy | 0.75 or higher | 0.467 | 0.733 (close) | 0.678 |
| Faithfulness | keep at 0.90 or higher | 0.933 | 0.800 | 0.786 * |

Only one target is met so far (precision in v2), and faithfulness has slipped below its floor in both new versions.

---

## Answer outcomes (manual check of `actual` vs `expected`)

The four DeepEval metrics never compare the answer to the expected answer, so this table was built by reading each `actual` against its `expected` in the result files.

- **Correct:** the main answer matches the expected answer
- **Partial:** right but incomplete, or one part missing or wrong
- **Wrong:** confidently gives a different figure or answer
- **Not found:** replies "I couldn't find this in the retrieved document text."

| Outcome | v0 | version1 | version2 |
|---|---|---|---|
| Correct | 5 (33%) | 6 (40%) | 7 (47%) |
| Partial | 4 | 3 | 0 |
| Wrong | 0 | 2 | 4 |
| Not found | 6 | 4 | 4 |

### Per-question comparison

| ID | Question (short) | Exp. page | Hit v0 | Hit v1 | Hit v2 | Outcome v0 | Outcome v1 | Outcome v2 |
|---|---|---|---|---|---|---|---|---|
| 1 | Revenue and profit for the period | 2 | Yes | Yes | Yes | Partial | Partial | Wrong |
| 2 | Revenue increase (calculation) | 2 | No | No | No | Correct | Correct | Correct |
| 3 | Main expense components | 2 | No | Yes | No | Partial | Partial | Wrong |
| 4 | Profit change year on year | 2 | No | Yes | No | Not found | Wrong | Wrong |
| 5 | Interim dividend decision | 1 | Yes | Yes | Yes | Correct | Correct | Correct |
| 6 | Record and payment dates | 1 | Yes | Yes | Yes | Partial | Correct | Correct |
| 7 | Basic vs diluted EPS | 2 | No | No | No | Not found | Not found | Not found |
| 8 | Top external-revenue segment | 3 | Yes | Yes | Yes | Correct | Correct | Correct |
| 9 | Tax expense and components | 2 | No | Yes | Yes | Not found | Not found | Not found |
| 10 | Total comprehensive income | 2 | No | No | Yes | Not found | Not found | Not found |
| 11 | Why no audit opinion | 10 | Yes | Yes | Yes | Correct | Correct | Correct |
| 12 | Accounting standard used | 10 | No | No | Yes | Partial | Partial | Correct |
| 13 | New group entities | 17 | No | No | No | Not found | Not found | Not found |
| 14 | Profit before tax | 2 | Yes | Yes | No | Not found | Wrong | Wrong |
| 15 | Board meeting duration | 1 | Yes | Yes | Yes | Correct | Correct | Correct |

Page 2 (the income statement) is expected by 8 questions (Q1, 2, 3, 4, 7, 9, 10, 14). It was retrieved for **2 of 8 in v0, 5 of 8 in version1, 3 of 8 in version2**.

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

### Findings

- **Page 2 is almost never retrieved** (2 of the 8 questions that need it). It holds the income statement: profit, tax, EPS, comprehensive income, total expenses.
- **The same pages keep coming back** (3, 4, 6, 8, 21, 22, 24, 26) whatever the question. Similar-looking numeric tables crowd out the right chunk.
- **Faithfulness is high (0.933).** The "do not guess" rule works: the system fails safe. 6 of 15 answers were "couldn't find".
- **Q14 hints at a chunking or extraction problem.** Page 2 was retrieved but the answer was still "couldn't find".
- **Metric noise exists.** Q2 scored recall 0.0 although the answer was correct (the numbers are inside the question).

---

## version1_chunk_change

**Run date:** 2026-10-17 12:50
**What changed vs v0:** chunk size 1000 to 1300 and overlap 200 to 500. Nothing else.

### Configuration (changed rows only; everything else identical to v0)

| Area | Setting | v0 | version1 |
|---|---|---|---|
| Chunking | `CHUNK_SIZE` | 1000 | **1300** |
| Chunking | `CHUNK_OVERLAP` | 200 | **500** |

The index must be rebuilt with `python ingest.py` after changing chunking. Retrieval is assumed to be plain semantic search with the reranker off (the result file does not store this).

### Results

| Metric | Mean | Pass rate | Change vs v0 (mean) |
|---|---|---|---|
| Contextual Recall | 0.833 | 0.867 | +0.033 |
| Contextual Precision | 0.508 | 0.533 | +0.048 |
| Faithfulness | 0.800 | 0.800 | -0.133 |
| Answer Relevancy | 0.733 | 0.733 | +0.266 |
| Page Hit Rate | 0.667 | n/a | +0.200 (10 of 15) |

### Findings

- **Biggest page-hit gain of the three versions.** Page 2 now shows up for 5 of 8 questions (Q1, 3, 4, 9, 14) instead of 2 of 8. Q3, Q4 and Q9 flipped from page miss to page hit.
- **Precision barely moved (+0.048).** The same crowd of pages (3, 4, 6, 20, 21, 22, 24) still appears for most questions. With overlap at 500, neighbouring chunks repeat text, so top-8 can be spent on near-duplicates (Q2 and Q15: 8 chunks came from only 4 pages).
- **Answer relevancy jumped (0.467 to 0.733)**, but mostly because fewer answers say "couldn't find" (6 down to 4). It does not mean the answers are right.
- **New failure mode: wrong answers.** Q4 and Q14 were "not found" in v0 and are now confident wrong figures (profit before exceptional items and tax of ₹5,065 crore from page 6, instead of the expected ₹5,132 crore profit and ₹6,467 crore PBT). Q1 also swapped the right profit for ₹5,065 crore.
- **Q6 fixed:** the record date is now found.
- **Faithfulness fell to 0.800.** Q1 is a real drop (the profit figure does not match the expected). Q8 and Q10 scored 0.0 but look like judge noise (Q8's answer is correct and its context scored recall 1.0 and precision 1.0).
- **Q9 still fails even with page 2 retrieved** (recall 1.0, precision 0.0, "couldn't find"), which points at how the table text is chunked or read, not only at ranking.

---

## version_2_re_ranker

**Run date:** 2026-10-17 16:30
**What changed vs v0:** fetch 30 candidates by semantic search, rerank with a cross-encoder, keep the best 8. Chunking is back to 1000 / 200 (taken from the result file).

### Configuration (changed rows only; everything else identical to v0)

| Area | Setting | v0 | version2 |
|---|---|---|---|
| Retrieval | Method | Semantic search, top 8 | **Semantic search, fetch 30, then cross-encoder rerank, keep 8** |
| Retrieval | `USE_RERANKER` | off | **True** |
| Retrieval | `RERANKER_MODEL` | none | **`BAAI/bge-reranker-base`** (max length 512) |
| Retrieval | `FETCH_K` | n/a | **30** |
| Retrieval | `TOP_K` | 8 | 8 |
| Chunking | Size / overlap | 1000 / 200 | 1000 / 200 |

The reranker settings come from `config.py` and `src/retriever.py`. The result JSON does not record them, so its config block looks identical to v0 (see "Housekeeping" below).

### Results

| Metric | Mean | Pass rate | Change vs v0 (mean) |
|---|---|---|---|
| Contextual Recall | 0.800 | 0.800 | 0.000 |
| Contextual Precision | 0.718 | 0.733 | **+0.258** |
| Faithfulness | 0.786 * | 0.857 * | -0.147 |
| Answer Relevancy | 0.678 | 0.733 | +0.211 |
| Page Hit Rate | 0.600 | n/a | +0.133 (9 of 15) |

\* 14 of 15 questions scored (Q12 judge timeout).

### Findings

- **Best precision (0.460 to 0.718).** The reranker does what it should: relevant chunks sit nearer the top. Recall is unchanged, so it reorders what is found rather than finding more.
- **Page hit is lower than version1 (0.600 vs 0.667).** Page 2 appears for only 3 of 8 questions (Q1, Q9, Q10). If the page 2 chunk is not among the 30 semantic candidates, the reranker cannot bring it back. Checking whether page 2 is in the candidate pool is worth doing before the next run.
- **Most correct answers so far (7 of 15).** Q6 and Q12 are now fully correct, and Q12 names Ind AS 34.
- **Four confident wrong answers (Q1, Q3, Q4, Q14).** Q1 gives revenue of ₹15,445 crore against the expected ₹35,620 crore. Q3 lists depreciation, interest and share-based payments from page 8 instead of the expense lines. Q4 and Q14 repeat the ₹5,065 crore figure. These numbers look like they belong to a different table (possibly standalone rather than consolidated figures on page 6). **Verify this by opening page 6 of the PDF.**
- **Q9 and Q10: page 2 retrieved, judge scored recall 1.0 and precision 1.0, yet the answer is "couldn't find".** The judge says the information was in the context, but the model could not use it. This points at table text where row labels and values are separated, not at ranking.
- **Faithfulness 0.786 is partly noise.** Q2 scored 0.0 although the answer is correct (the numbers are in the question). Q12 has no score, so rerun it.

---

## Cross-version analysis

1. **Retrieval is better, answers only modestly.** Correct answers went 5, 6, 7 of 15 (33%, 40%, 47%) while page hit went 0.467, 0.667, 0.600. Each fix moved one or two questions.
2. **Four questions fail in every version: Q7, Q9, Q10, Q13.** Q7 (EPS), Q9 (tax) and Q10 (comprehensive income) are all rows of the page 2 income statement. Better ranking and bigger chunks have not fixed them, so look at how page 2 is extracted and chunked. Q13 expects page 17 (the list of group entities), which no version ever retrieved. That is a separate retrieval problem.
3. **Better retrieval created wrong answers (0, then 2, then 4).** In v0 the bot said "couldn't find"; later versions pull a plausible-looking chunk and answer with a different figure. Faithfulness does not catch this because the answer is faithful to the wrong chunk.
4. **The metrics miss wrong answers.** Answer relevancy is 1.0 for several wrong answers (version2 Q1, Q3, Q4, Q14). None of the four DeepEval metrics checks the actual answer against `expected_output`, so a correctness metric is the biggest gap in the eval.
5. **The sample is small.** One question is worth 6.7 points. version1 vs version2 page hit differs by one question, so do not read it as a firm ranking.
6. **Page-hit may be too strict.** Some content appears on more than one page: Q11 and Q12 expect page 10 but answers cite page 19, and Q8 expects page 3 but v2 cites page 21. Allow several acceptable pages per question.

### Housekeeping noted while reviewing

- `eval/run_eval.py` saves only chunking, top-K, models and threshold in `config`. It does not save `use_reranker`, `reranker_model` or `fetch_k`, so version2's saved config looks the same as v0's. Add these fields.
- Version labels are inconsistent (`v0_naive`, `version1_chunk_change`, `version_2_re_ranker`). Pick one scheme, for example `v1_chunking`, `v2_rerank`.
- The golden set `type` labels are loose (Q3 and Q5 are marked `subjective`). Clean them up before deeper analysis.
- `golden_dataset.json` accepts `source_pages` (a list) inside `additional_metadata`. `run_eval.py` already supports it, so use it for the duplicate-content questions.

---

## Ideas for upcoming versions

These are candidates, not commitments. Change **one thing per version** so the effect is clear, except where a combination is stated.

| Version | Change | Why |
|---|---|---|
| `v3_table_extraction` | Read page 2 so each row keeps its label and values together (for example PyMuPDF `find_tables()`, or one chunk per statement), and check the text with `python ask.py "..." --show-chunks` | Q7, Q9, Q10 fail in every version even when page 2 is retrieved |
| `v4_hybrid` | Add BM25 keyword search next to semantic search, feed both into the reranker | Exact terms such as "Basic EPS" and "Total comprehensive income" should match lexically; also widens the candidate pool |
| `v5_chunk_plus_rerank` | Combine 1300 / 500 chunking with the reranker | Chunking got page 2 in front more often; the reranker improved ordering. Worth testing, not guaranteed to stack |
| `v6_scope_metadata` | Tag chunks as consolidated or standalone and prefer consolidated | Only if page 6 is confirmed to hold standalone figures; targets the wrong-answer cases |
| (eval upgrade) | Add an answer-correctness metric (for example DeepEval `GEval` against `expected_output`), multi-page `source_pages`, and the reranker fields in saved config | Makes wrong answers visible in the scores |

---

## How to run and record a new version

```bash
# 1. Change config.py / the relevant src/ file for the new idea
# 2. Rebuild the index if chunking or embeddings changed
python ingest.py data/your_report.pdf

# 3. Smoke test (3 questions), then full run
python eval/run_eval.py --version v3_table_extraction --limit 3
python eval/run_eval.py --version v3_table_extraction

# 4. Print the comparison table across all versions
python eval/compare.py
```

Then copy the section template below into this file.

### Template for a new version

```markdown
## vN_name

**Run date:** YYYY-MM-DD HH:MM
**What changed vs baseline:** one sentence

### Configuration (changed rows only)
| Area | Setting | v0 | this version |
|---|---|---|---|

### Results
| Metric | Mean | Pass rate | Change vs v0 |
|---|---|---|---|

### Findings
- what improved
- what got worse
- what to try next
```
