# Best ZIP generator map

`main` is intentionally answer-only. The full generator is preserved in the
previous repository commit `5c18ac3` and in the original `publish/rebuild-clean`
source tree.

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

To inspect the generator without bloating `main`:

```bash
git clone https://github.com/lducc/vifinqa-final-submission.git
cd vifinqa-final-submission
git checkout 5c18ac3
```

An exact fresh rerun additionally needs the original large candidate-pair,
reranker-score, ranking, and sidecar files. Those are experiment artifacts,
not part of the minimal answer repository. The frozen ZIP is the reproducible
control when those inputs are unavailable.
