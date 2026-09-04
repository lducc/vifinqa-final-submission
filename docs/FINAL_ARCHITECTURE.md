# Final architecture

The system is a provenance-preserving Vietnamese financial QA pipeline. It
retrieves source tables first, then emits an executable answer from only the
submitted CSV evidence.

```text
HTML reports
  -> span-aware table parser + metadata
  -> company/year/scope document gate
  -> sparse BM25 + same-triple dense candidates
  -> slot expansion (report, year, line item, operand, table family)
  -> Qwen3-Reranker-8B ordering
  -> report-aware budget, deduplication, evidence trace
  -> minimal source-derived CSVs
  -> direct cell lookup OR safe Pandas program
  -> AST validation + execution replay
  -> submission.json / submission.zip
```

## Query decomposition

Decomposition is structured rather than free-form LLM chain-of-thought. The
question is normalized into slots containing:

```text
report/company, year(s), line item(s), unit, table family, operation, role
```

Examples:

```text
"Doanh thu tăng bao nhiêu phần trăm từ 2022 đến 2023?"
  -> revenue / 2022 / old_value
  -> revenue / 2023 / new_value
  -> operation=growth

"Công ty nào có doanh thu cao nhất và lợi nhuận của công ty đó?"
  -> all companies / revenue / selector
  -> winning company / profit / target
```

One-period, one-cell questions use deterministic lookup. Arithmetic,
comparison, aggregation, year selection, and multi-hop questions use a
validated one-assignment Pandas program. Units, OCR number formatting, and
rounding are handled by code rather than guessed by the model.

## Retrieval invariants

- candidate tables remain traceable to an organizer report and table ID;
- dense and graph additions cannot silently change the document gate;
- final answer generation cannot change `relevant_docs` or `relevant_tables`;
- every query reads at least one submitted evidence frame;
- unsafe syntax, imports, file access, and non-scalar results are rejected.

The final package and its exact checks are documented in
[`PUBLISH_HANDOFF_20260826.md`](PUBLISH_HANDOFF_20260826.md).
