#!/usr/bin/env python3
"""Build a table-relevance review queue from disputed (question, table) pairs.

The 233-question benchmark labels were discovered from one public submission's
predictions, so they agree with our own retriever more than the organizer does:
on the same packages, local binding precision is 0.4412 against a public 0.3516,
and local recall 0.7634 against a public 0.6148. The gold set is the right size
and the wrong contents.

A pair is disputed when a treatment retrieves a table the labels reject while
the organizer's score says the treatment gained recall. Those pairs carry more
information per judgement than any other part of the set, so they are reviewed
first. This script renders each one against its source so a reviewer can decide
whether the table genuinely carries evidence the question needs.

Nothing here writes a label. It only assembles what a judgement requires: the
question, the tables already believed to be gold and the rows they were bound
to, and the disputed table as it appears in the report.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from vifinqa.tables import (
    cached_report_text,
    extract_rows_at_line,
    iter_report_paths,
    parse_report_identity,
)


def report_index(dataset_root: Path) -> dict[str, Path]:
    return {
        parse_report_identity(path, dataset_root).report_id: path
        for path in iter_report_paths(dataset_root)
    }


def split_table_id(table_id: str) -> tuple[str, int]:
    report_id, _, line = table_id.rpartition("|")
    if not report_id or not line.isdigit():
        raise ValueError(f"Malformed table id: {table_id}")
    return report_id, int(line)


def render(rows: list[list[str]], max_rows: int, max_columns: int) -> dict:
    """Compact a table for review without hiding what makes it relevant.

    Row labels decide relevance, so column 0 is never truncated by width; the
    numeric columns beside it only need to be present enough to show periods.
    """
    kept = rows[:max_rows]
    body = []
    for row in kept:
        label = (row[0] if row else "").strip()
        rest = [cell.strip() for cell in row[1:max_columns]]
        body.append([label, *rest])
    return {
        "rows_shown": len(kept),
        "rows_total": len(rows),
        "columns_total": max((len(row) for row in rows), default=0),
        "grid": body,
    }


def context_lines(report_text: str, start_line: int, before: int) -> list[str]:
    """The prose immediately above a table, which usually names it."""
    lines = report_text.splitlines()
    start = max(0, start_line - 1 - before)
    return [line.strip() for line in lines[start:start_line - 1] if line.strip()][-before:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disputed", type=Path, required=True,
                        help="JSONL of {question_id, table} pairs to review")
    parser.add_argument("--benchmark", type=Path,
                        default=Path("attic/annotations/benchmark.jsonl"))
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("data/raw/vifinqa"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=45)
    parser.add_argument("--max-columns", type=int, default=7)
    parser.add_argument("--context-lines", type=int, default=4)
    args = parser.parse_args()

    benchmark = {}
    with args.benchmark.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            benchmark[record["id"]] = record

    disputed: dict[int, list[str]] = defaultdict(list)
    with args.disputed.open(encoding="utf-8") as handle:
        for line in handle:
            pair = json.loads(line)
            disputed[pair["question_id"]].append(pair["table"])

    reports = report_index(args.dataset_root)
    missing: list[str] = []
    written = 0

    with args.output.open("w", encoding="utf-8") as out:
        for question_id in sorted(disputed):
            record = benchmark[question_id]
            annotation = record["annotation"]
            bindings = defaultdict(list)
            for binding in annotation.get("row_column_bindings", []):
                if binding.get("table"):
                    bindings[binding["table"]].append({
                        "row": binding.get("row"),
                        "column": binding.get("column"),
                        "row_label": binding.get("row_label"),
                        "raw": binding.get("raw"),
                    })

            entries = []
            for table_id in sorted(set(disputed[question_id])):
                report_id, start_line = split_table_id(table_id)
                path = reports.get(report_id)
                if path is None:
                    missing.append(table_id)
                    continue
                text = cached_report_text(str(path))
                try:
                    rows = extract_rows_at_line(text, start_line)
                except KeyError:
                    missing.append(table_id)
                    continue
                entries.append({
                    "table": table_id,
                    "report_id": report_id,
                    "start_line": start_line,
                    "context": context_lines(text, start_line, args.context_lines),
                    **render(rows, args.max_rows, args.max_columns),
                })

            if not entries:
                continue
            out.write(json.dumps({
                "question_id": question_id,
                "question": record["question"],
                "tier": record["tier"],
                "label_source": record["source"],
                "current_gold": [
                    {"table": table, "bound_rows": bindings.get(table, [])}
                    for table in sorted(set(
                        binding["table"]
                        for binding in annotation.get("row_column_bindings", [])
                        if binding.get("table")
                    )) or sorted(set(annotation["gold_tables"]))
                ],
                "disputed": entries,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"questions written: {written}", file=sys.stderr)
    print(f"disputed pairs rendered: "
          f"{sum(len(v) for v in disputed.values()) - len(missing)}", file=sys.stderr)
    if missing:
        print(f"unresolved tables: {len(missing)}", file=sys.stderr)
        for table_id in missing[:10]:
            print(f"  {table_id}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
