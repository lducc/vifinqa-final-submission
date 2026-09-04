# ViFinQA final pipeline

One reproducible path from organizer data to a validated submission:

```text
questions + financial statements
  → document gate (company/year/scope)
  → BM25 + same-triple dense candidates
  → Qwen3 reranker over candidate pairs
  → RRF fusion
  → adaptive table selection (logit gap 3)
  → source-derived evidence CSVs
  → grounded Qwen answer program
  → sandbox replay and validation
  → submission.zip
```

The frozen public-best control is
`artifacts/submission_best_20260826.zip` (1,012 questions), SHA-256
`97baa1201450801e50480ce68db055d7bdbc99029cac93c21654cdf471789067`.
Recorded public metrics: tables F2 `0.5495`, documents F2 `0.9711`, and answer
execution `0.1561`.

## Inputs

Keep the organizer corpus outside Git at `data/raw/vifinqa/`:

```text
data/raw/vifinqa/code_stock.csv
data/raw/vifinqa/questions/questions.jsonl
data/raw/vifinqa/financial_statements/<ticker>/<year>/...
```

The dense index is also external at `output/dense/` and must contain the files
written by `kaggle/embed_tables.py`. Model weights are downloaded by the runtime.

## Fresh retrieval and ZIP

Install the package and run the deterministic candidate build in a new output
directory:

```bash
python -m pip install -e .
RUN=output/final_run
scripts/build_final_candidates.sh "$RUN"
```

Upload `$RUN/pairs_hybrid_graph.jsonl` to a GPU runtime and run
`kaggle/rerank_qwen_8b.py`. Download a complete `scores.jsonl` and its manifest,
then validate and derive the ranking:

```bash
python scripts/validate_rerank_scores.py \
  --pairs "$RUN/pairs_hybrid_graph.jsonl" \
  --scores "$RUN/scores.jsonl" \
  --manifest "$RUN/scores.jsonl.manifest.json"

python scripts/apply_rerank_scores.py \
  --pairs "$RUN/pairs_hybrid_graph.jsonl" \
  --scores "$RUN/scores.jsonl" \
  --arm hybrid --weight 0.5 \
  --output "$RUN/ranking_hybrid_qwen.json"

python scripts/select_slot_tables.py \
  --pairs "$RUN/pairs_hybrid_graph.jsonl" \
  --scores "$RUN/scores.jsonl" \
  --base-ranking "$RUN/ranking_hybrid_qwen.json" \
  --starting-ranking "$RUN/ranking_hybrid_qwen.json" \
  --adaptive-append --logit-gap 3 \
  --output "$RUN/ranking_adaptive_gap3.json" \
  --trace "$RUN/ranking_adaptive_gap3.trace.jsonl"
```

Merge the generated sidecars, package the evidence, and validate:

```bash
python scripts/merge_table_sidecars.py \
  --input "$RUN/dense_tables.json" \
  --input "$RUN/ranking_graph_candidates.graph_tables.json" \
  --output "$RUN/extra_tables.json"

python run.py \
  --data-root data/raw/vifinqa \
  --ranking "$RUN/ranking_adaptive_gap3.json" \
  --extra-tables "$RUN/extra_tables.json" \
  --table-top-k ranking --rerank-depth 130 \
  --output-dir "$RUN/package_run"

python scripts/validate_submission.py "$RUN/package_run/package"
python scripts/verify_execution.py --package "$RUN/package_run/package"
```

The retrieval ZIP is `$RUN/package_run/submission.zip`.

## Final answering notebook

For the frozen public-best ZIP, upload it to
`/content/submission_best_20260826.zip`, clone this repository, and open
`notebooks/08_answer_from_best_zip.ipynb` on an A100 runtime:

```python
!git clone --depth 1 https://github.com/lducc/vifinqa-final-submission.git \
    /content/vifinqa-final-submission
```

The notebook extracts evidence with `scripts/build_answer_inputs.py`, starts
Qwen3.5-9B through vLLM, validates grounded programs with Pydantic, retries at
most twice, replays every answer, and writes:

```text
/content/submission_qwen35_grounded.zip
```

For a newly generated retrieval ZIP, run the same builder manually with that
ZIP and use `notebooks/05_answer_adaptive_qwen35.ipynb`.

## Code map

* `run.py` — retrieval, evidence materialization, and package creation;
* `src/vifinqa/` — table parsing, retrieval, fusion, reranking, binding;
* `scripts/` — candidate export, score validation, adaptive selection, checks;
* `kaggle/embed_tables.py` — dense-index generation;
* `kaggle/rerank_qwen_8b.py` — GPU pair scoring;
* `notebooks/03_rerank_cascade.ipynb` — BGE/Qwen cascade experiment;
* `notebooks/05_answer_adaptive_qwen35.ipynb` — answer stage from extracted evidence;
* `notebooks/08_answer_from_best_zip.ipynb` — answer stage from the frozen control.

No raw organizer data, credentials, model weights, or temporary outputs belong
in Git. Every run must use a new output directory and preserve its hashes.
