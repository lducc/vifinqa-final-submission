# Best ZIP generator map

`main` contains the lean end-to-end generator. Unrelated training, benchmark,
and infrastructure experiments are omitted.

The generator is composed of:

```text
run.py
scripts/build_final_candidates.sh
scripts/export_rerank_pairs.py
scripts/apply_rerank_scores.py
scripts/select_slot_tables.py
scripts/validate_submission.py
src/vifinqa/{retrieval,fusion,rerank,slots,answers,tables}.py
```

The public-best package used this flow:

```text
document gate
→ sparse BM25 + same-triple dense candidates
→ Qwen reranker score fusion
→ select_slot_tables.py adaptive append, logit gap 3
→ run.py package evidence CSVs
→ validate_submission.py
```

The frozen output is `artifacts/submission_best_20260826.zip`. Its SHA-256 is
`97baa1201450801e50480ce68db055d7bdbc99029cac93c21654cdf471789067`.

## Fresh reconstruction

The organizer corpus is external and must be available at
`data/raw/vifinqa/`. A compatible dense index must be at `output/dense/`.
Build a new directory; never overwrite an earlier run:

```bash
RUN=output/final_repro
scripts/build_final_candidates.sh "$RUN"
```

Upload `$RUN/pairs_hybrid_graph.jsonl` to the reranker runtime and download a
complete score file plus its manifest. Validate it before deriving rankings:

```bash
python scripts/validate_rerank_scores.py \
  --pairs "$RUN/pairs_hybrid_graph.jsonl" \
  --scores "$RUN/scores_qwen8b.jsonl" \
  --manifest "$RUN/scores_qwen8b.jsonl.manifest.json"
```

Derive the fused ranking, then apply the public-best adaptive-gap selection:

```bash
python scripts/apply_rerank_scores.py \
  --pairs "$RUN/pairs_hybrid_graph.jsonl" \
  --scores "$RUN/scores_qwen8b.jsonl" \
  --arm hybrid --weight 0.5 \
  --output "$RUN/ranking_hybrid_qwen.json"

python scripts/select_slot_tables.py \
  --pairs "$RUN/pairs_hybrid_graph.jsonl" \
  --scores "$RUN/scores_qwen8b.jsonl" \
  --base-ranking "$RUN/ranking_hybrid_qwen.json" \
  --starting-ranking "$RUN/ranking_hybrid_qwen.json" \
  --adaptive-append --logit-gap 3 \
  --output "$RUN/ranking_adaptive_gap3.json" \
  --trace "$RUN/ranking_adaptive_gap3.trace.jsonl"
```

Merge dense/graph sidecars and package all 1,012 questions:

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

The resulting retrieval ZIP is `$RUN/package_run/submission.zip`. To run the
answer stage on that fresh package, use `scripts/build_answer_inputs.py` and
then `notebooks/05_answer_adaptive_qwen35.ipynb`. The notebook preserves the
retrieval fields, checkpoints every question, and writes the final answer ZIP.

If the large pair/score artifacts are unavailable, use the preserved
`artifacts/submission_best_20260826.zip` as the validated control with
`notebooks/08_answer_from_best_zip.ipynb`; that is a frozen-baseline answer run,
not a fresh retrieval reconstruction.

The historical full source is also available at commit `5c18ac3`:

```bash
git clone https://github.com/lducc/vifinqa-final-submission.git
cd vifinqa-final-submission
git checkout 5c18ac3
```

An exact fresh rerun additionally needs the original large candidate-pair,
reranker-score, ranking, and sidecar files. Those are experiment artifacts,
not part of the minimal answer repository. The frozen ZIP is the reproducible
control when those inputs are unavailable.
