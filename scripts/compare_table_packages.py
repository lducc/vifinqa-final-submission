#!/usr/bin/env python3
"""Compare table selections while proving answer fields stayed untouched."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile


def load_submission(path: Path) -> dict[int, dict]:
    if path.is_dir():
        rows = json.loads((path / "submission.json").read_text(encoding="utf-8"))
    else:
        with ZipFile(path) as archive:
            rows = json.loads(archive.read("submission.json"))
    result = {int(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate question IDs in {path}")
    return result


def compare(left: dict[int, dict], right: dict[int, dict]) -> dict:
    if left.keys() != right.keys():
        raise ValueError("packages contain different question IDs")
    same_docs = same_tables = same_order = same_answers = same_queries = 0
    added = removed = 0
    for identifier in left:
        before, after = left[identifier], right[identifier]
        old_tables = before.get("relevant_tables", [])
        new_tables = after.get("relevant_tables", [])
        same_docs += before.get("relevant_docs") == after.get("relevant_docs")
        same_tables += set(old_tables) == set(new_tables)
        same_order += old_tables == new_tables
        same_answers += before.get("answer") == after.get("answer")
        same_queries += before.get("pandas_query") == after.get("pandas_query")
        added += len(set(new_tables) - set(old_tables))
        removed += len(set(old_tables) - set(new_tables))
    total = len(left)
    return {
        "questions": total,
        "same_docs": same_docs,
        "same_table_set": same_tables,
        "same_table_order": same_order,
        "same_answers": same_answers,
        "same_pandas_queries": same_queries,
        "tables_added": added,
        "tables_removed": removed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    print(json.dumps(
        compare(load_submission(args.left), load_submission(args.right)), indent=2,
    ))


if __name__ == "__main__":
    main()
