# Outputs

Generated candidate pools, reranker scores, evaluation reports, evidence CSVs,
and submission archives are intentionally not versioned. They can be large,
contain source-derived competition data, and are reproducible from the checked-in
code plus the organizer corpus.

To build a fresh package:

```bash
python3 run.py --output-dir output/run
PYTHONPATH=src .venv/bin/python scripts/validate_submission.py output/run/package
```

The publish-ready local artifact used for the current assessment is:

```text
output/final_slots_fast/adaptive_gap3_final_current/submission.zip
```

Its SHA-256 and public metrics are recorded in
`docs/PUBLISH_HANDOFF_20260826.md`. Upload that archive to the organizer or
attach it to a private GitHub release; do not commit the generated CSV tree.
