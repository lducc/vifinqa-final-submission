# Three notebooks, one A/B

Generated from `kaggle/rerank_qwen_8b.py` with the settings block edited; the
code is otherwise identical. Each file is one notebook. Where a file has two
parts separated by `# %% ---- next cell`, paste each part into its own cell and
run them in order — the second is a control scored in the same session, because
scores drift across sessions and a comparison has to stay inside one.

Current Kaggle launch notebooks:

- `qwen3_reranker_benchmark.ipynb`: 233-question smoke and prompt-quality gate.
- `qwen3_reranker_full.ipynb`: complete canonical-pool scoring run.

Both install `bitsandbytes>=0.46.1` and `accelerate` in their first code cell.
Their second cell embeds the complete scorer; attach only the pair file as the
Kaggle dataset named in that notebook.

Attach the `vifinqa-rerank-pairs` dataset holding `pairs_bench_v4.jsonl` and
`pairs_bench_v5.jsonl`. GPU T4. Use commit ("Save & Run All") so the session
survives a closed tab.

| notebook | cell 1 | cell 2 (control, same session) | ~hours |
|---|---|---|---:|
| `nb1_control.py` | v4, prompt v1 -> `scores_ctrl.jsonl` | — | 2.3 |
| `nb2_position.py` | v5, prompt v2 -> `scores_v5.jsonl` | v4, v1 -> `scores_ctrl_nb2.jsonl` | 4.6 |
| `nb3_peritem.py` | v5, v2, PER_ITEM -> `scores_v5_peritem.jsonl` | v5, v2 -> `scores_v5_nb3.jsonl` | 5.5 |

Download every scores file into `output/rerank/`. Then:

```
python3 scripts/compare_rerank_runs.py --pairs output/rerank/pairs_bench_v4.jsonl \
    --control output/rerank/scores_ctrl_nb2.jsonl --treatment output/rerank/scores_v5.jsonl --aa-items 99
python3 scripts/compare_rerank_runs.py --pairs output/rerank/pairs_bench_v5.jsonl \
    --control output/rerank/scores_v5_nb3.jsonl --treatment output/rerank/scores_v5_peritem.jsonl
```

The first reads position + instruction; every prompt changes, so `--aa-items 99`
reports the whole set and the drift is what nb1 against nb2's control shows.
The second reads PER_ITEM with its 150-question A/A stratum built in.
