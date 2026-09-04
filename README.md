# ViFinQA table retrieval

Finds the financial-statement tables that answer a Vietnamese question, and
packages them as a submission.

```
question
  → normalize into report/year/line-item/operation slots
  → document gate: ticker, year, scope
  → BM25 + same-triple dense table candidates
  → Qwen3-Reranker-8B ordering and report-aware budget
  → source-derived CSV evidence
  → deterministic lookup or validated Pandas program
  → execution replay and submission.zip
```

Everything is derived from the organizers' public data. The organizer corpus is
an external input under `data/raw/vifinqa/`; it is deliberately not versioned.

## Where it stands

Current published artifact (`output/final_slots_fast/adaptive_gap3_final_current`):

| | |
|---|---:|
| Documents F2 | 0.9711 |
| Tables F2 | 0.5495 |
| Table precision / recall | 0.3801 / 0.6520 |
| Table MRR5 | 0.6134 |
| Answer / execution | 0.1561 / 0.1561 |

The deterministic candidate pool for the next zero-shot reranking comparison is
built and verified. Its three arms share one 71,975-pair scoring file, so sparse,
same-triple dense, and traceable note-link additions can be compared fairly.

## Setup

Keep the organizer data at `data/raw/vifinqa/` (Git ignores it):

```
code_stock.csv
questions/questions.jsonl
financial_statements/<TICKER>/<YEAR>/<REPORT_ID>/<REPORT_ID>_extracted.txt
```

Install the package — `scripts/` and `tests/` import `vifinqa`, so cloning is
not enough:

```bash
python3 -m pip install -e .
python3 -m pytest tests -q
```

## Colab notebooks

The two runnable notebooks are:

* `notebooks/03_rerank_cascade.ipynb` — BGE candidate scoring followed by the
  Qwen cascade and `run.py` packaging.
* `notebooks/05_answer_adaptive_qwen35.ipynb` — grounded Qwen3.5-9B answer
  generation from the packaged evidence, with checkpointing and validation.

The rerank notebook expects these uploaded inputs:

```
/content/vifinqa_retrieval_local/candidates.jsonl
/content/vifinqa_retrieval_local/retrieval_manifest.json
/content/vifinqa_data/
```

The answer notebook expects the `answer_inputs/` directory produced by
`scripts/build_answer_inputs.py`. Both notebooks write checkpoints and final
artifacts under `/content/output/`; mount Google Drive and change that output
directory in the first configuration cell when a persistent run is needed.

To clone this repository in Colab after publishing it:

```bash
!git clone --depth 1 https://github.com/lducc/vifinqa-final-submission.git /content/vifinqa-final-submission
```

Then open the notebooks from `/content/vifinqa-final-submission/notebooks/`.
Raw organizer data and model weights are intentionally external inputs; they
are not committed to this repository.

## Building a submission

```bash
python3 run.py --output-dir output/run
```

Writes `output/run/submission.zip`, containing `submission.json` and one CSV per
retrieved table. A strict validator runs first and rejects bad IDs, tables whose
report is not declared, duplicates, missing CSVs, and schema errors.

With a reranker, pass its ranking:

```bash
python3 run.py --ranking output/rerank/ranking.json \
    --extra-tables output/rerank/pairs.dense_tables.json \
    --output-dir output/run
```

`--extra-tables` is only needed when the ranking contains dense candidates: it
tells `run.py` which report to find a table in when its own gate did not select
that report.

## Documentation

| file | what it is |
|---|---|
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | historical retrieval experiment notes |
| [`docs/final-stretch-plan-20260825.md`](docs/final-stretch-plan-20260825.md) | preregistered scoring and submission protocol |
| [`docs/FINAL_ARCHITECTURE.md`](docs/FINAL_ARCHITECTURE.md) | final end-to-end architecture and query decomposition |
| [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) | retrieval, binding, execution errors, and evaluation limits |
| [`docs/RELATED_REPOS.md`](docs/RELATED_REPOS.md) | audited public repositories and reusable lessons |
| [`docs/PUBLISH_HANDOFF_20260826.md`](docs/PUBLISH_HANDOFF_20260826.md) | exact upload artifact and reproducibility checks |
| [`docs/GOLD_SET.md`](docs/GOLD_SET.md) | gold-set publication policy and local hashes |

## Layout

```
run.py                          build a submission
src/docs.py                     the document gate, and packaging
src/vifinqa/retrieval.py        BM25 table ranking inside the gated filings
src/vifinqa/rerank.py           how a table is described to the reranker
src/vifinqa/fusion.py           turn reranker scores into an ordering
src/vifinqa/answers.py          bind a cell and emit the pandas query
src/vifinqa/tables.py           OCR table parsing, CSV evidence

scripts/measure_questions.py    print the constants the sampler uses
scripts/sample_groups.py        draw table groups to write questions about
scripts/assemble_training.py    join questions to tables, filter, add negatives
scripts/export_table_texts.py   one passage per table, for the dense index
scripts/export_rerank_pairs.py  the candidate pool to be scored
scripts/apply_rerank_scores.py  scores → ranking, per arm
scripts/audit_gate.py           measure the gate against the corpus
scripts/audit_dense_overlap.py  where dense candidates sit next to the gate
scripts/audit_questions.py      compare generated questions to released ones
scripts/evaluate_holdout.py     score retrieval on held-out reports
scripts/validate_submission.py  strict package validator

kaggle/embed_tables.py          embed the corpus and the questions
kaggle/rerank_qwen_8b.py        score the candidate pool
```
