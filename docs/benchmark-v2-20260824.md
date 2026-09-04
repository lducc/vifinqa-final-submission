# Benchmark v2: repairing the disputed table labels

Date: 2026-08-24

This records a labelling pass over the 233-question retrieval benchmark, what it
fixed, and — more importantly — what it failed to fix. The negative result at the
end is the part that should drive the next decision.

## 1. Why the old labels were suspect

`attic/annotations/benchmark.jsonl` records `seeded_share: 0.6438`. Its gold
tables were discovered by looking at what public submission 2333 retrieved and
then relocating each retained table in the raw reports. Discovery from a
retriever's own output cannot surface a table no retriever proposed.

Scoring the nine ablation packages against those labels and against the
organizer shows the consequence:

| | binding gold | organizer |
|---|---:|---:|
| gold tables per question | 3.24 | 3.27 |
| precision on the same packages | 0.4412 | 0.3516 |
| recall on the same packages | 0.7634 | 0.6148 |

The label set is the right size and the wrong contents. It agrees with our own
retriever roughly twenty per cent more than the organizer does.

One hypothesis was rejected immediately: the anti-correlation is not caused by
newer arms submitting more unlabelled tables. Across arms 02 to 08 the off-gold
fraction is flat at 48.4–48.9 per cent, correlating at `-0.06` with public F2.

The sharpest symptom is the single decision the sprint used the benchmark for.
Packages `00` and `08` differ by 222 swapped tables on these questions:

```text
local  : recall 0.7823 -> 0.7634  (-0.0189)   F2 -0.0165
public : recall 0.6129 -> 0.6148  (+0.0019)   F2 +0.0030
```

The benchmark said the change was harmful. The organizer said it helped.

## 2. What was judged

The 178 (question, table) pairs that `08` adds and the labels reject, covering
94 questions. Each was rendered against its source report — the note heading
above the table, the column headers, and every row label — and judged for
whether it carries evidence the question needs.

| verdict | count | high confidence | medium | low |
|---|---:|---:|---:|---:|
| relevant | 25 | 17 | 8 | 0 |
| uncertain | 14 | 0 | 13 | 1 |
| not relevant | 139 | 126 | 13 | 0 |

By tier:

| tier | relevant | uncertain | not relevant |
|---|---:|---:|---:|
| easy | 0 | 1 | 3 |
| medium | 4 | 3 | 14 |
| intermediate | 11 | 5 | 78 |
| hard | 10 | 5 | 44 |

### What the relevant ones look like

Almost every confirmed addition is the **note that backs a statement line**. For
question 13 the labels hold only the balance-sheet cash line; the disputed table
is note *"4. TIỀN VÀ CÁC KHOẢN TƯƠNG ĐƯƠNG TIỀN"*, whose `TỔNG CỘNG` is the
identical figure. The same shape recurs: the borrowings-terms note behind
`Vay ngắn hạn`, the intangible-asset movement note behind `Tài sản cố định vô
hình`, the selling-expense note behind `Chi phí bán hàng`. Value identity
against a bound cell is a strong signal for this class.

### What the rejected ones look like

The 139 rejections are dominated by vocabulary collisions, not near-misses of
judgement:

- **polarity** — `phải thu` (receivable) matched against `phải trả` (payable);
- **tangible against intangible** — `tài sản cố định hữu hình` against `vô hình`;
- **short against long term** — `chi phí trả trước ngắn hạn` against `dài hạn`;
- **wrong report year** — a 2023 income statement retrieved for a 2024 question;
- **boilerplate** — subsidiary lists, related-party schedules, branch addresses,
  and company information sheets, which share entity names with every question
  about that company.

## 3. The result, including the part that failed

Folding the 25 confirmed labels in moves the close-arm correlation:

| label set | gold/q | close-arm Pearson | close-arm Spearman |
|---|---:|---:|---:|
| v1 as-is | 3.24 | −0.8066 | −0.5273 |
| v2, +25 relevant | 3.35 | −0.4983 | −0.3091 |
| v2, +25 relevant +14 uncertain | 3.41 | −0.1559 | +0.0545 |

And it corrects the decision that motivated the work:

| label set | local `08` − `00` | public |
|---|---:|---:|
| v1 as-is | −0.0165 | +0.0030 |
| v2, +25 relevant | **+0.0076** | +0.0030 |

**The correlation improvement does not survive scrutiny.** The judged pool was
drawn from `08` alone, so every added label is a table `08` retrieves. Holding
that arm out:

| label set | arms 02–07 Pearson | Spearman |
|---|---:|---:|
| v1 as-is | −0.7445 | −0.2353 |
| v2, +25 relevant | −0.7042 | −0.5294 |
| v2, +25 relevant +14 uncertain | −0.8255 | −0.5294 |

No improvement. The apparent repair was pooling asymmetry, not better labels.

## 4. What this means

1. **The old labels were mostly right.** 139 of 178 disputed tables — 78 per cent
   — are genuinely irrelevant. The label set's failure is concentrated in one
   specific class, the backing note, not spread across the whole set.
2. **The line-item slot mechanism has poor precision.** It adds 222 tables across
   233 questions and roughly 14 per cent (25) are gold, 22 per cent counting the
   uncertain ones. That is why `08`'s public precision barely moved. A filter on
   that mechanism is a concrete, cheap target: rejecting the polarity,
   tangible/intangible, term, and boilerplate classes above is deterministic
   work, not modelling.
3. **Partial pooling cannot repair this benchmark.** Only a symmetric pool over
   all retrievers can, because the bias being corrected is exactly a bias toward
   one retriever's output.
4. **v2 is a harm gate, not a selector.** It may be used for the `00` versus `08`
   comparison it was built to settle, and to reject large regressions. It must
   not be used to choose among half-point variants. `scripts/evaluate_benchmark_fidelity.py`
   enforces this: pass `--min-close-spearman` and it exits non-zero.

## 5. Artifacts

```text
data/annotations/benchmark_v2/verdicts.jsonl
  sha256: 9a3f81d9549e3375a7b5273f14737d504cf3930e0d301e420267f969edbb10fb
  178 adjudications, each with verdict, confidence, and a written reason

data/annotations/benchmark_v2/benchmark_v2.jsonl
  sha256: f3a10e95d6a2fd84ea1ead2bcde1c8787a327b8b6982adabdaa533f0a76584d5
  233 questions; adds gold_tables_v2, added_in_v2, deferred_uncertain

data/annotations/benchmark_v2/manifest.json
  counts, input hashes, scope, and known limitations
```

`attic/annotations/benchmark.jsonl` is unchanged, hash
`c6ecedeca223bad686f1a47254ca1d7f3d819496a8ff37d7ca84f615c0d811ec`.

## 6. Reproducing

```bash
.venv/bin/python scripts/build_review_queue.py \
  --disputed <disputed-pairs.jsonl> --output <review-queue.jsonl>

.venv/bin/python scripts/apply_review_verdicts.py \
  --verdicts data/annotations/benchmark_v2/verdicts.jsonl \
  --output data/annotations/benchmark_v2/benchmark_v2.jsonl \
  --manifest data/annotations/benchmark_v2/manifest.json

.venv/bin/python scripts/evaluate_benchmark_fidelity.py \
  --benchmark data/annotations/benchmark_v2/benchmark_v2.jsonl
.venv/bin/python scripts/evaluate_benchmark_fidelity.py \
  --benchmark data/annotations/benchmark_v2/benchmark_v2.jsonl --hold-out 08
```

Add `--include-uncertain` to `apply_review_verdicts.py` for the sensitivity run.

## 7. Provenance and disclosure

The 178 judgements were made by an LLM assistant reading the rendered source
tables, under the trust boundary that applies to every internal label in this
repository: they are self-reviewed internal annotation, not organizer ground
truth and not independent double annotation. They are development-time
evaluation data and form no part of the submitted system, which remains
deterministic. Each verdict carries a written reason so any of them can be
contested individually. The 14 uncertain verdicts are held out of the default
gold set precisely because they need a second reviewer.
