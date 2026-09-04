#!/usr/bin/env python3
"""Measure how many gold-150 operands the pipeline binds, end to end.

Execution accuracy needs every operand of a question's formula to sit inside
the bound cells; one missing operand makes the whole plan fail. This runs the
submission path — gate, retrieval, row binding, year matching — over the 150
questions with reviewed gold evidence and reports, per question, whether every
gold (table, row, column) is covered and whether the planner emits anything.

    python scripts/evaluate_operand_coverage.py [--no-years]
"""

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from docs import (
    load_companies, load_reports as load_doc_reports, make_row, parse_question,
    required_report_years, retrieve_docs,
)
from vifinqa.answers import bound_cells, first_numeric_cell
from vifinqa.jsonl import load_jsonl
from vifinqa.retrieval import (
    load_reports as load_table_reports, metric_query_tokens, retrieve_rows, table_budget,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--evidence", type=Path,
                        default=ROOT / "data" / "results" / "e027_gold_150" / "evidence.csv")
    parser.add_argument("--no-years", action="store_true",
                        help="skip the year-matched cells, reproducing the pre-change binding")
    parser.add_argument("--start", type=int, default=0,
                        help="zero-based offset into reviewed questions; useful for resumable batches")
    parser.add_argument("--limit", type=int,
                        help="maximum reviewed questions to measure after --start")
    args = parser.parse_args()

    questions = {int(row["id"]): row for row in load_jsonl(
        args.dataset_root / "questions" / "questions.jsonl")}
    gold: dict[int, set[tuple[str, int, int]]] = {}
    for operand in csv.DictReader(args.evidence.open(encoding="utf-8")):
        gold.setdefault(int(operand["question_id"]), set()).add(
            (operand["table"], int(operand["row"]), int(operand["column"])))

    companies = load_companies(args.dataset_root / "code_stock.csv")
    doc_reports = load_doc_reports(args.dataset_root / "financial_statements")
    table_reports = load_table_reports(args.dataset_root)

    identifiers = sorted(gold)[args.start:]
    if args.limit is not None:
        identifiers = identifiers[:args.limit]

    outcomes = Counter()
    for identifier in identifiers:
        source = questions[identifier]
        parsed = parse_question(source["question"], companies)
        if not parsed.tickers and parsed.candidate_tickers:
            parsed.tickers = parsed.candidate_tickers[:1]
        docs, _ = retrieve_docs(parsed, doc_reports)
        metadata = {"tickers": parsed.tickers, "years": parsed.years,
                    "slot_years": required_report_years(parsed), "scope": parsed.scope}
        result = retrieve_rows(source["question"], metadata, table_reports,
                               top_k=table_budget(len(docs), "auto"),
                               report_ids=docs, mode="report-coverage")
        years = [] if args.no_years else required_report_years(parsed)
        tokens = {token for token in metric_query_tokens(source["question"], {"tickers": [], "years": []}) if len(token) > 2}
        bound: set[tuple[str, int, int]] = set()
        for rank, table in enumerate(result["tables"]):
            for index, column in bound_cells(table["rows"], table.get("header_cells"), tokens, years):
                bound.add((table["table_id"], index, column))
        needed = gold[identifier]
        covered = needed & bound
        if covered == needed:
            outcomes["all_operands"] += 1
        elif covered:
            outcomes["partial"] += 1
        else:
            outcomes["none"] += 1

    total = sum(outcomes.values())
    print(f"questions={total} all_operands={outcomes['all_operands']}"
          f" partial={outcomes['partial']} none={outcomes['none']}"
          f" full_rate={outcomes['all_operands'] / total:.4f}")


if __name__ == "__main__":
    main()
