#!/usr/bin/env python3
"""Score value-column choice on gold-150, holding the row binding at oracle.

Execution accuracy dies on reading the wrong cell far more often than on
arithmetic — the organizers' own error analysis puts numerical extraction at
54.7% of end-to-end failures and calculation at 0.9%. The gold-150 evidence file
pins every operand down to (table, row, column), so feeding the selector the
gold row isolates the one decision this script measures: which period column a
row's figure is read from.

    python scripts/evaluate_extraction.py
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.answers import first_numeric_cell
from vifinqa.jsonl import load_jsonl
from vifinqa.retrieval import header_cells, load_reports, report_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--evidence", type=Path,
                        default=ROOT / "data" / "results" / "e027_gold_150" / "evidence.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "cells" / "extraction_report.json")
    args = parser.parse_args()

    reports_by_id = {report.identity.report_id: report for report in load_reports(args.dataset_root)}
    questions = {int(record["id"]): record["question"] for record in load_jsonl(
        args.dataset_root / "questions" / "questions.jsonl")}
    operands = list(csv.DictReader(args.evidence.open(encoding="utf-8")))

    grids: dict[str, object] = {}
    outcomes: Counter = Counter()
    misses = []
    for operand in operands:
        table_id, report_id = operand["table"], operand["report_id"]
        if table_id not in grids:
            report = reports_by_id.get(report_id)
            grids[table_id] = next(
                (item for item in report_tables(str(report.path), report.identity)
                 if item.table_id == table_id), None) if report else None
        table = grids[table_id]
        if table is None:
            outcomes["table_unreadable"] += 1
            continue
        row = int(operand["row"])
        if row >= len(table.rows):
            outcomes["row_out_of_range"] += 1
            continue
        chosen = first_numeric_cell(list(table.rows[row]), header_cells(table))
        column = int(operand["column"])
        if chosen is None:
            outcome = "no_numeric"
        elif chosen[0] != column:
            outcome = "column_missed"
        else:
            outcome = "hit"
        outcomes[outcome] += 1
        if outcome == "column_missed":
            misses.append({
                "id": operand["question_id"],
                "table": table_id,
                "gold_column": column,
                "chosen_column": chosen[0],
                "gold_header": operand["column_header"][:60],
                "gold_raw": operand["raw"][:40],
                "chosen_raw": str(table.rows[row][chosen[0]])[:40],
                "headers": header_cells(table),
                "question": questions.get(int(operand["question_id"]), "")[:120],
            })

    total = sum(outcomes.values())
    report = {
        "bindings": total,
        "outcomes": dict(outcomes),
        "column_hit_rate": round(outcomes["hit"] / total, 4),
        "example_misses": misses[:25],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("bindings", "outcomes", "column_hit_rate")}, indent=2))


if __name__ == "__main__":
    main()
