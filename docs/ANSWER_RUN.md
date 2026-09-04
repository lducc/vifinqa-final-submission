# Answer run

## Input

Upload the frozen retrieval package as:

```text
/content/submission_best_20260826.zip
```

The package must contain 1,012 unique rows and all referenced evidence CSVs.
The notebook verifies ZIP integrity before extracting anything.

## Run

Open `notebooks/08_answer_from_best_zip.ipynb` on a GPU runtime and run all
cells. It uses Qwen3.5-9B through vLLM. Numbers are masked in the model context;
the model returns a bounded cell program, not an answer detached from evidence.

Each result is validated with Pydantic, replayed in the sandbox, and retried at
most twice. Results are appended to a checkpoint so an interrupted run can be
resumed.

## Outputs

```text
/content/answer_inputs_best/
/content/qwen35_answer/answer_programs_qwen35.jsonl
/content/qwen35_answer/answer_audit_qwen35.jsonl
/content/qwen35_answer/input_provenance.json
/content/qwen35_answer/answer_manifest_qwen35.json
/content/submission_qwen35_grounded.zip
```

The final ZIP is written only after all 1,012 rows have valid answers,
replayable programs, and packaged evidence. Retrieval fields are checked to be
unchanged during answering.
