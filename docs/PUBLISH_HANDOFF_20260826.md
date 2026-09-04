# ViFinQA publish handoff — 2026-08-26

## Upload artifact

Upload this file to the organizer:

`output/final_slots_fast/adaptive_gap3_final_current/submission.zip`

SHA-256:

`97baa1201450801e50480ce68db055d7bdbc99029cac93c21654cdf471789067`

A convenience copy is versioned at
`artifacts/submission_best_20260826.zip` with the same SHA-256.

The package contains `submission.json` and the source-derived evidence CSVs
for all 1,012 questions. Retrieval is frozen; answer generation does not alter
`relevant_docs` or `relevant_tables`.

Reported public result for this package:

| metric | value |
|---|---:|
| TABLES_F2MACRO | 0.5495 |
| TABLES_PRECISION | 0.3801 |
| TABLES_RECALL | 0.6520 |
| TABLES_MRR5 | 0.6134 |
| DOCS_F2MACRO | 0.9711 |
| ANSWER_ACCURACY | 0.1561 |
| EXECUTION_ACCURACY | 0.1561 |

These are the supplied public metrics, not a local re-score.

## Reproduction check

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/validate_submission.py \
  output/final_slots_fast/adaptive_gap3_final_current/package
PYTHONPATH=src .venv/bin/python scripts/verify_execution.py \
  --package output/final_slots_fast/adaptive_gap3_final_current/package
```

Current checks:

- `164 passed`;
- package validator: `VALID`;
- execution replay: 989 agreed, 7 pre-existing answer/query discrepancies,
  0 raised, 16 rows without a query.

The seven discrepancies are in the inherited answer stage (IDs 424, 485, 486,
546, 561, 562, 564); they are not package-validation failures.

## Method summary

```text
question
  -> company/year/scope document gate
  -> sparse + same-triple dense table candidates
  -> slot/line-item coverage expansion
  -> Qwen reranker ordering
  -> deterministic table budget and deduplication
  -> source-derived CSV evidence
  -> deterministic cell/program answer binding
  -> validated submission.zip
```

Gemini artifacts under `output/gemini_answering/` were development experiments
only. They are not part of the publish artifact and must not be used for the
organizer submission. The final competition pipeline must use the local Qwen
implementation and the checked-in code/configuration.

## Handoff rules

- Do not include `.env`, API keys, raw organizer data, or temporary files.
- Do not rebuild retrieval before publishing this artifact.
- Preserve the ZIP and SHA-256 above as the submitted version.
- If a new reranker score is tested, create a new output directory and validate
  it independently; never overwrite this package.
