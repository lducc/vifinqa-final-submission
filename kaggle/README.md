# Cross-encoder reranking on Kaggle

Every team ahead of us on table retrieval runs a reranker; we do not. Our MRR@5 is
0.482 against 0.51–0.63 for the systems above us, and reranking is the stage that
moves exactly that number. A local attempt with an English web-passage
cross-encoder (`mmarco-mMiniLMv2`) lost 0.15 F2 — the model was wrong for
Vietnamese financial tables, not the idea. This runs the multilingual model the
leaders use, on a GPU we do not have locally.

Only the ordering of already-retrieved candidates changes. The document gate, the
candidate set, and the table IDs are produced locally and untouched here, so
discarding the result means deleting one file.

## Steps

**1. Export the pairs (local, ~35 min)**

```
python3 scripts/export_rerank_pairs.py --output output/rerank/pairs_v4.jsonl
```

Writes one record per question with its top-50 candidates and the table
representation the model scores — 50,335 pairs, 52 MB.

The representation changed after the first run. Qwen-4B moved live F2 by only
+0.023, and the likely reason was that each candidate was described by the single
row BM25 matched: the model was judging tables through the sparse ranker's choice
and could not recover when that choice was wrong. Candidates now also carry the
table's line-item inventory (`Các chỉ tiêu: …`), which says what the table
actually contains. Pass `--inventory 0` to reproduce the old representation.

**2. Upload to Kaggle**

Create a Dataset named `vifinqa-rerank-pairs` containing `pairs_v4.jsonl` and
`pairs_bench_v4.jsonl`, then point `PAIRS_PATH` at the one you want. Qwen scored the first export
at F2 +0.0355 on the benchmark and +0.023 live, so it is the model to use;
`rerank_bge.py` is kept only to document that a general web-passage reranker
loses 0.15 F2 here.

**3. Run one of the notebooks (Kaggle)**

New notebook → Accelerator: **GPU T4** (or **T4 x2** for fp16) → attach the
dataset → paste the script into a cell → edit the settings block at the top →
Run. It prints throughput and an ETA and writes `SCORES_PATH` (~1 MB).

| Script | Model | Memory | 50,335 pairs at 1024 |
|---|---|---|---|
| `rerank_qwen_8b.py` | pinned Qwen3 8B or 4B reranker | ~8.2 GB at 8-bit | resumable on one T4 |
| `rerank_bge.py` | bge-reranker-v2-m3 | ~1 GB | kept only as the record of a failure |

One script covers both sizes: `MODEL_NAME=Qwen/Qwen3-Reranker-4B` selects its
own pinned repository revision and runs 4B in
6–8 h. The separate `rerank_qwen.py` and `rerank_qwen_4b.py` were the same
loop with the settings hard-coded and are gone.

The settings block at the top of the script:

| setting | default | what it does |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3-Reranker-8B` | selects the separately pinned 8B or 4B revision |
| `PAIRS_PATH` | `pairs_bench_v4.jsonl` | the export to score |
| `SCORES_PATH` | `/kaggle/working/scores.jsonl` | appended per question |
| `QUANTIZATION` | `fp16` | use a 24 GB card or T4 x2; one T4 needs `int8` or `nf4` |
| `RESUME_PATHS` | `[]` | prior score shards: skips whole questions; keep shards separate |
| `SKIP_PATH` | `""` | a finished scores.jsonl: skips individual candidates |
| `ADAPTER_PATH` | `""` | a LoRA adapter from `train_reranker.py` |

Below the block, `MAX_LENGTH = 1024` and `RERANK_DEPTH = 130` are settled: 512
lost the line-item inventory and cost 0.023 live, and 130 sits above the deepest
augmented export so nothing is cut by accident.

**Run 8B.** Both were scored at max_length 1024 on the `pairs_v3` export, measured
on the 233-record benchmark against the sparse ranking:

| ranking | MRR@5 | recall@5 | easy | medium | intermediate | hard |
|---|---:|---:|---:|---:|---:|---:|
| sparse | 0.7466 | 0.7234 | 0.890 | 0.765 | 0.673 | 0.614 |
| 4B fuse | 0.7937 | 0.7670 | 0.914 | 0.818 | 0.736 | 0.672 |
| 8B fuse | 0.8190 | 0.7932 | 0.937 | 0.827 | 0.779 | 0.694 |

Replace loses at both sizes, as it has every time on this task. Fusing 4B and 8B
together raises MRR to 0.8250 but drops recall to 0.7855 and regresses hard from
0.694 to 0.669, so 8B alone is the ranking to use — F2 pays for recall.

The window is what unlocked this. At 512 the same family moved live F2 by only
+0.023; the 16% of candidates that overflowed were losing the line-item inventory
that says what a table holds.

**8B loads in 8-bit** because 8.19B parameters is 16.4 GB in fp16 and a T4 has
16 GB. At int8 it is about 8.2 GB, fits one card, and stays near-lossless. It is
the slower of the two models by more than the parameter count suggests:
bitsandbytes computes outlier features in higher precision, roughly 1.5–2× fp16
on Turing.

Candidates are sorted by length before batching, since a batch costs its longest
member — worth 25–40% of the runtime — and scores append per question, so a
session that hits the 12-hour limit is rerun rather than restarted.

All models are open, inside the 14B limit, and released before the 2026-06-01
cutoff. Kaggle needs internet enabled to download them.

The two scripts score differently under the hood. BGE is a cross-encoder with a
classification head, so its logit is the score directly. Qwen3-Reranker is a
causal model asked whether the document answers the query, and the score is the
probability it answers "yes" — which needs a specific prompt format and left
padding so the final position is the real one in every row of a batch.

**4. Download and fuse (local, seconds)**

Put `scores.jsonl` in `output/rerank/`, then:

```
python3 scripts/apply_rerank_scores.py --pairs output/rerank/pairs_v4.jsonl \
    --scores output/rerank/scores.jsonl
python3 scripts/apply_rerank_scores.py --pairs output/rerank/pairs_v4.jsonl \
    --scores output/rerank/scores.jsonl --mode replace   # trust the model outright
```

Writes `output/rerank/ranking.json`. Fusion is the default because on this task
every replacement has lost and every fusion that earned its place has won;
`--mode replace` exists so the difference can be measured rather than assumed.

## Score both cells of an A/B in one session

Scores are not reproducible across sessions: the same model, prompt and candidate
text scored twice agree on 138 of 11,592 pairs, and rebuilding the ranking from
the other run's scores moves benchmark F2 by +0.0076, CI [-0.0063, +0.0235]. int8
is not batch-invariant — bitsandbytes decomposes outlier features per batch, and
batches are packed from whatever candidate set the run holds. So a control from
one session and a treatment from another differ by more than the treatment.

**The fix that always works is running both cells in one session**, changing only
the factor under test. Do that first; it costs nothing and needs no new code
path.

For the promoted hierarchy experiment, the paired entrypoint is already
fail-closed:

```bash
bash kaggle/score_hierarchy_ab.sh
```

It scores `pairs_flat_233.jsonl` followed by `pairs_hierarchy_233.jsonl`, rejects
pre-existing output files, verifies that every question and candidate was
scored, and records SHA-256 hashes in
`output/table_retrieval/gpu_ab/sha256sum.txt`. Override `FLAT_PAIRS`,
`HIERARCHY_PAIRS`, or `OUTPUT_DIR` when the attached GPU filesystem uses other
paths.

After downloading that directory, build both fused rankings:

```bash
.venv/bin/python scripts/apply_rerank_scores.py \
  --pairs output/table_retrieval/pairs_flat_233.jsonl \
  --scores output/table_retrieval/gpu_ab/scores_flat_233.jsonl \
  --output output/table_retrieval/gpu_ab/ranking_flat.json
.venv/bin/python scripts/apply_rerank_scores.py \
  --pairs output/table_retrieval/pairs_hierarchy_233.jsonl \
  --scores output/table_retrieval/gpu_ab/scores_hierarchy_233.jsonl \
  --output output/table_retrieval/gpu_ab/ranking_hierarchy.json
```

The active comparison is the canonical candidate pool described in
`docs/final-stretch-plan-20260825.md`; do not use this historical A/B example
to select the shipping arm.

`QUANTIZATION=fp16` on **GPU T4 x2** is the optional upgrade. 16.4 GB does not
fit one T4 but splits across two, and dropping the int8 outlier path should
shrink the drift — not to zero, since fp16 matmuls are not batch-invariant
either. Two caveats before spending a session on it: nothing here has ever run
it, and the 1.5-2x speed figure is a single-card int8-versus-fp16 comparison, not
a measurement of this model split across two cards, where the layers run as a
pipeline and one card idles while the other computes. If it misbehaves, fall back
to int8 and keep both cells in the one session.

Paste the script into two cells and change two lines between them:

```python
# cell 1, the control                 # cell 2, the treatment
SCORES_PATH = ".../scores_ctrl.jsonl"   SCORES_PATH = ".../scores_peritem.jsonl"
PER_ITEM = False                        PER_ITEM = True
```

11,592 pairs then 16,383. The model frees when the first cell returns, so the
second reloads it without running out of memory.

Then locally:

```
python3 scripts/compare_rerank_runs.py --pairs output/rerank/pairs_bench_v4.jsonl \
    --control output/rerank/scores_bench_fp16.jsonl \
    --treatment output/rerank/scores_bench_fp16_peritem.jsonl
```

It reports the raw agreement, an A/A stratum — the 150 of 233 questions whose
prompt is byte-identical under both settings, so their difference is drift — and
the 83 treated questions. The treated stratum has to clear the A/A stratum before
any of it is the method.

`SKIP_PATH` is the other half of not wasting a session: point it at a finished
score file and its (question, table) pairs are not judged again, so scoring
`pairs_v4_d100.jsonl` on top of `scores_v4.jsonl` costs 40,391 pairs rather than
90,726. `apply_rerank_scores.py --scores a --scores b` unions them back together.

## Testing a prompt without burning a full run

The instruction and the query shape matter as much as the model here. Scoring all
1,012 questions to compare two prompts is wasteful — export only the benchmark
questions instead:

```
python3 scripts/export_rerank_pairs.py \
    --questions <(python3 -c "import json;[print(json.dumps({'id':r['id'],'question':r['question']},ensure_ascii=False)) for r in map(json.loads,open('annotations/benchmark.jsonl'))]") \
    --output output/rerank/pairs_bench.jsonl
```

That is 221 questions, ~11k pairs, roughly 45 minutes at 4B instead of 3-4 hours,
and every one of them has verified labels, so the comparison is measurable
locally with no submission spent. Run the winner on the full export afterwards.

The current prompt does two things the first version did not. It tells the model
that company, period and scope are already settled — 99% of candidates sit in a
gold report, so judging those again is wasted capacity — and it names the line
item explicitly, quoted from the question with its diacritics intact:

```
<Query>: Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?
         Chỉ tiêu cần tìm: Lãi tiền gửi
```

It also warns that Vietnamese accounting labels differ by one word and mean
different figures, which is the discrimination the whole task turns on, and that
a note restating a figure still counts — restatements are gold here.

## What to check before shipping it

Measure on the benchmark first — the reranked ordering has to beat the sparse one
on the 192 labelled questions, and the tier breakdown has to not regress. If the
model helps, the submission is rebuilt with the fused ordering. If it does not,
delete `scores.jsonl` and nothing else changes.
