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

The component boundaries and invariants are described in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Inputs

Keep the organizer corpus outside Git at `data/raw/vifinqa/`:

```text
data/raw/vifinqa/code_stock.csv
data/raw/vifinqa/questions/questions.jsonl
data/raw/vifinqa/financial_statements/<ticker>/<year>/...
```

The dense index is also external at `output/dense/` and must contain the files
written by `kaggle/embed_tables.py`. Model weights are downloaded by the runtime.

## Fresh run, step by step

### 1. Check out the code and install it

```bash
git clone https://github.com/lducc/vifinqa-final-submission.git
cd vifinqa-final-submission
python -m pip install -e .
```

Copy the organizer files into the `data/raw/vifinqa/` layout above and place the
completed dense index in `output/dense/`. Build a new run directory:

```bash
RUN=output/final_run
PYTHON=python DENSE_DIR=output/dense scripts/build_final_candidates.sh "$RUN"
```

This checks the code and dense index, then writes the complete candidate union
and sidecars. It does not score with a model.

### 2. Score the exported pairs on a GPU

Upload `$RUN/pairs_hybrid_graph.jsonl` to a GPU runtime as
`/kaggle/input/vifinqa-rerank-pairs/pairs_hybrid_graph.jsonl`, then run the
reranker with positional paths:

```bash
python kaggle/rerank_qwen_8b.py \
  /kaggle/input/vifinqa-rerank-pairs/pairs_hybrid_graph.jsonl \
  /kaggle/working/scores.jsonl
```

Download the complete `scores.jsonl` and its generated
`scores.jsonl.manifest.json` to `$RUN`, then validate and derive the ranking:

### 3. Validate scores and select tables

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

### 4. Package the fresh retrieval result

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

### 5. Generate answers

For a fresh retrieval ZIP, upload it to the answer runtime as
`/content/retrieval_fresh.zip`, clone the repository there, and create the input
directory:

```bash
git clone --depth 1 https://github.com/lducc/vifinqa-final-submission.git \
  /content/vifinqa-final-submission
python /content/vifinqa-final-submission/scripts/build_answer_inputs.py \
  /content/retrieval_fresh.zip \
  --output-dir /content/answer_inputs
```

The final notebook expects this exact layout. The manifest must contain all
1,012 retrieval rows and reference every evidence CSV:

```text
/content/answer_inputs/
├── retrieval_manifest.jsonl
└── data/
    └── tables/
        ├── table_*.csv
        └── ...
```

Clone the repository in that runtime and run all cells in
`notebooks/final_answer.ipynb` on an A100. It writes the final
answer ZIP and checkpoint files under `/content/`.

## Frozen-control answering notebook

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

The frozen-control notebook performs the same answer/validation stage, but its
input is the committed public-best ZIP rather than a fresh retrieval output.

## Code map

* `run.py` — retrieval, evidence materialization, and package creation;
* `src/vifinqa/` — table parsing, retrieval, fusion, reranking, binding;
* `scripts/` — candidate export, score validation, adaptive selection, checks;
* `kaggle/embed_tables.py` — dense-index generation;
* `kaggle/rerank_qwen_8b.py` — GPU pair scoring;
* `notebooks/03_rerank_cascade.ipynb` — BGE/Qwen cascade experiment;
* `notebooks/final_answer.ipynb` — answer stage from extracted evidence;
* `notebooks/08_answer_from_best_zip.ipynb` — answer stage from the frozen control.

No raw organizer data, credentials, model weights, or temporary outputs belong
in Git. Every run must use a new output directory and preserve its hashes.
