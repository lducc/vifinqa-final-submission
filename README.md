# ViFinQA final answer run

This repository contains only the final answer-stage runner and the validated
retrieval baseline used as its input.

```text
submission_best_20260826.zip
  -> validate and extract evidence
  -> Qwen3.5-9B grounded program generation
  -> Pydantic validation
  -> sandbox replay with bounded repair
  -> submission_qwen35_grounded.zip
```

The retrieval ZIP is frozen and is not regenerated here. Its recorded public
metrics are TABLES F2 0.5495 and ANSWER/EXECUTION 0.1561.

## Colab

Clone the repository and upload `submission_best_20260826.zip` to `/content/`:

```python
!git clone --depth 1 https://github.com/lducc/vifinqa-final-submission.git \
    /content/vifinqa-final-submission
```

Open [`notebooks/08_answer_from_best_zip.ipynb`](notebooks/08_answer_from_best_zip.ipynb)
and run all cells on an A100 runtime. The notebook clones the repository when
needed, extracts answer inputs, checkpoints every question, validates replay,
and writes `/content/submission_qwen35_grounded.zip`.

## Files

* `notebooks/08_answer_from_best_zip.ipynb` — complete answer run;
* `scripts/build_answer_inputs.py` — deterministic ZIP-to-evidence extraction;
* `artifacts/submission_best_20260826.zip` — frozen 1,012-question input;
* `docs/ANSWER_RUN.md` — exact inputs, outputs, and validation contract.

No planner, retrieval index, reranker, raw organizer data, credentials, or
temporary experiment artifacts are included.
