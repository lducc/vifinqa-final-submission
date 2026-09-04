# Related public repositories

This is a comparative audit, not an endorsement or a claim that any repository
is a hidden winning submission. No audited public repository was verified to be
the closed-source top leaderboard system.

| Repository | Reusable lesson | Limitation |
|---|---|---|
| [Noone9725/R2AI-Stage-2](https://github.com/Noone9725/R2AI-Stage-2) | Complete hybrid retrieval, adaptive evidence selection, Pandas generation, repair | No independently verified top score; no license found |
| [kimmttrung/vifinqa-system](https://github.com/kimmttrung/vifinqa-system) | Fact lookup bypass, adaptive read/submit budgets, reproducible packaging | Public result below the current system |
| [TanKai-247/financial-text-to-pandas](https://github.com/TanKai-247/financial-text-to-pandas) | Explicit evidence slots, cell linker, typed plan, constrained compiler | Public score around 0.253 |
| [Le-Ngoc-Tu/finwhale](https://github.com/Le-Ngoc-Tu/finwhale) | Numeric masking, auditor, source-traceable final package | Dense reranking was not consistently beneficial |
| [CryAndRRich/r2ai-stage2](https://github.com/CryAndRRich/r2ai-stage2) | Safe execution, self-contained queries, packaging checks | Reranking reduced its measured table score |
| [Dle28/nlp-finance-query-](https://github.com/Dle28/nlp-finance-query-) | Exact-cell binding, typed plans, claim/evidence graphs, certification | Research branch intentionally abstains on most questions; not a deadline submission |

The cross-repository consensus is to keep lexical/structural retrieval as the
backbone, add dense candidates conservatively, bind exact cells, and execute
only traceable programs. The full branch audit is preserved in the project
history and summarized in [`HANDOFF.md`](HANDOFF.md).
