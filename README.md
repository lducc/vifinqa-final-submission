# ViFinQA end-to-end submission pipeline

This repository contains the original retrieval, reranking, packaging, and
answer stages, plus the validated public-best retrieval artifact.

```text
questions + financial statements
  -> document gate
  -> BM25 + dense candidate union
  -> Qwen reranking and adaptive table selection
  -> evidence CSV packaging
  -> grounded Qwen3.5-9B answering
  -> sandbox replay and validation
  -> submission.zip
```

The frozen public-best package is also available as
`artifacts/submission_best_20260826.zip`. Its recorded public metrics are
TABLES F2 0.5495 and ANSWER/EXECUTION 0.1561.

## Colab

Clone the repository and upload `submission_best_20260826.zip` to `/content/`:

```python
!git clone --depth 1 https://github.com/lducc/vifinqa-final-submission.git \
    /content/vifinqa-final-submission
```

For a fast answer-stage run from the frozen package, open
[`notebooks/08_answer_from_best_zip.ipynb`](notebooks/08_answer_from_best_zip.ipynb)
and run all cells on an A100 runtime. For a fresh retrieval run, use
`scripts/build_final_candidates.sh`, score the generated pair manifest, apply
the ranking, and invoke `run.py`; the exact sequence is documented in
[`docs/BEST_ZIP_PIPELINE.md`](docs/BEST_ZIP_PIPELINE.md).

## Files

* `run.py` — end-to-end retrieval/package entrypoint;
* `scripts/` and `src/vifinqa/` — retrieval, reranking, selection, and validation;
* `notebooks/08_answer_from_best_zip.ipynb` — answer run from frozen evidence;
* `scripts/build_answer_inputs.py` — deterministic ZIP-to-evidence extraction;
* `artifacts/submission_best_20260826.zip` — frozen 1,012-question control;
* `docs/ANSWER_RUN.md` and `docs/BEST_ZIP_PIPELINE.md` — run contracts.

Raw organizer data, model weights, credentials, and temporary experiment
artifacts are not included.
