# Validated retrieval results

This document records the two strongest retrieval results used during the
project. Both are valid results under the evaluation protocol that produced
them:

* `.7364` is the strongest local result on the reviewed 233-question benchmark.
* `.5647` is the strongest reported public result from the organizer runs.

They must not be compared as if they were the same evaluation set. The
versioned public package is recorded separately in
[`PUBLISH_HANDOFF_20260826.md`](PUBLISH_HANDOFF_20260826.md).

## Shared retrieval idea

The system retrieves evidence before answering:

```text
question
  -> company / year / scope gate
  -> parent-question candidates
  -> decomposed fact candidates
  -> candidate union
  -> table ranking
  -> conservative table selection
  -> source-derived evidence CSVs
  -> deterministic row/cell binding and answer execution
```

The document gate is deliberately deterministic. This prevents a table for a
different company, reporting year, or scope from entering the answer merely
because its wording is similar.

## Strongest local result: `.7364`

The `.7364` result is the safe-hard ranking control measured on the reviewed
233-question benchmark.

| metric | value |
|---|---:|
| F2 | .7364 |
| precision | .5908 |
| recall | .8224 |
| MRR5 | .8039 |

### How it was created

```text
deterministic company/year/scope gate
  -> parent BM25 candidates
  -> validated decomposed fact candidates
  -> parent + fact candidate fusion
  -> safe-hard ordering and bounded selection
  -> benchmark scoring
```

The selector preserves a parent candidate as an anchor, covers missing fact
slots, removes only objectively redundant entries, and keeps the original
ordering when a safety check fails. It does not allow a small semantic model to
delete evidence merely because a table looks similar.

### Why it helped

At candidate depth 130, validated decomposition increased reviewed candidate
recall from `.9480` to `.9906`. The gain came from exposing tables that the
whole-question query ranked too low, especially on multi-year and multi-table
questions.

Examples:

* **Q98 — inventory:** the parent ranking contained the correct closing
  inventory table below a tempting inventory-movement table. Keeping the parent
  lane prevented the same-label movement table from replacing the balance
  table.
* **Q175 — provisions:** the plan needs short-term and long-term provision
  balances. A movement or reconciliation note may contain the same words, but
  it is not automatically the requested closing balance.
* **Q955 — four banks:** the question needs short-term loans and total loans
  for every bank. A fixed two-table budget can starve this high-arity case, so
  coverage is considered before pruning.

## Strongest reported public result: `.5647`

The `.5647` result is the strongest reported public table-retrieval run:

| metric | value |
|---|---:|
| F2 | .5647 |
| precision | .4024 |
| recall | .6644 |
| MRR5 | .6159 |
| answer / execution | .1502 |

### How it was created

```text
deterministic document gate
  -> BM25 and same-triple dense candidate union
  -> Qwen3-Reranker-8B question/table scoring
  -> conservative table budget and deduplication
  -> fresh source-derived evidence CSVs
  -> validated submission package
```

The reranker changed table ordering; it did not invent table IDs or values. The
final package was rebuilt from the selected table IDs so that evidence files
and `relevant_tables` stayed aligned.

## Typical failure cases

These failures explain why both precision and recall matter:

1. **Total versus component.** A question asking for total assets can retrieve
   current-assets and non-current-assets schedules. Those are useful clues but
   are not automatically the requested total.
2. **Balance versus movement.** A provision or inventory movement note can share
   the exact line-item words with the closing balance table. Lexical retrieval
   alone cannot reliably distinguish their roles.
3. **Wrong period.** Comparative columns and later filings can contain an older
   year. Report eligibility and period-aware binding must be checked separately.
4. **High-arity budget starvation.** Questions such as Q363 and Q955 require
   many year/company operands. A global top-five table cap raises precision on
   simple questions while silently losing recall on these questions.
5. **Answer-stage binding.** Even when the right table is selected, the answer
   can fail by reading a label column, applying the wrong unit scale, or binding
   a movement value instead of a closing value. These are row/cell errors, not
   candidate-retrieval errors.

## Interpretation

The local `.7364` result validates the conservative selection methodology on a
reviewed diagnostic set. The public `.5647` result validates the larger
reranker-based retrieval approach on organizer scoring. The later `.5495`
package is the fully versioned public fallback, while these two higher results
remain important validated research results and controls.

Further planner and reranker experiments must use new artifacts and new hashes;
they must not overwrite these results or silently mix their evaluation sets.
