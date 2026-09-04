# Error analysis

## Metric diagnosis

The supplied public result for the current package is TABLES F2 `0.5495`,
precision `0.3801`, recall `0.6520`, MRR5 `0.6134`, and DOCS F2 `0.9711`.
Documents are therefore near-solved. Table precision is the largest retrieval
gap; recall is already close to the strongest public systems.

The package replay covers 1,012 questions: 989 agreed executions, 7 inherited
answer/query discrepancies, 0 raised exceptions, and 16 rows without a query.

## Retrieval failures

The reviewed candidate errors fall into five recurring classes:

1. polarity collisions: `phải thu` versus `phải trả`;
2. qualifier collisions: tangible versus intangible, short-term versus
   long-term, consolidated versus separate;
3. wrong period: a correct statement family from the wrong year;
4. boilerplate: subsidiary lists, addresses, and related-party schedules that
   share company vocabulary;
5. backing-note ambiguity: a note table can duplicate the statement value and
   is useful evidence, but lexical matching alone cannot distinguish it from a
   distractor.

The slot mechanism improves operand coverage but can add noisy tables. The
safe design is multi-label table metadata plus reranking, with deterministic
qualifier/year filters and no unconditional dense replacement.

## Answer/execution failures

The remaining answer errors are concentrated in:

- selecting a label/code column instead of the numeric period column;
- OCR thousands/decimal separators and unit scaling;
- multi-year arithmetic using the wrong year column;
- multi-hop selection leaking the selector metric into the target metric;
- unsupported or missing query expressions.

The current compiler addresses these with structured unit context, year-aware
cell binding, safe AST validation, and execution replay. The seven replay
discrepancies remain documented rather than silently patched.

## Evaluation limits

The internal reviewed table labels are development diagnostics, not organizer
ground truth. They were partly sampled from retriever outputs and can miss
unproposed gold tables. Public leaderboard values are the authoritative
competition result; local benchmarks are used for regression checks and error
classification.

Details and adjudicated examples are in
[`benchmark-v2-20260824.md`](benchmark-v2-20260824.md).
