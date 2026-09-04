#!/usr/bin/env python3
"""Measure the document gate, which nothing downstream can compensate for.

The gate reads a ticker, a year and a scope out of the question and selects the
filings that may hold the answer. Everything after it only reorders inside that
selection, so a report the gate leaves out is a gold table no reranker can
recover and no embedding index can rank. It is the one stage whose errors are
final.

It has never been measured here, and the dense-overlap audit cannot do it: that
compares the index against `parse_question`, which is the gate's own parser, so
the two are wrong together whenever the parser is. This measures the gate
against the corpus instead, which needs no labels at all.

Four failures, all of them things the gate already knows and does not report:

  No ticker. The question names a company the parser could not resolve to one of
  the 98 codes, so `run.py` guesses the first candidate. Every question naming a
  bank by its full Vietnamese name and no code is a chance for this.

  No year. `required_report_years` comes back empty and the selection widens to
  every year in the corpus, which is not a gate at all.

  Nothing selected. A (ticker, year) pair the parser believed in has no filing
  in the corpus. Either the parser read it wrong or the corpus does not have it,
  and the two are worth telling apart.

  Wrong scope served. The question asked for one scope and `choose_report`
  returned another. This only counts when a differently-labelled filing was
  actually available: 55 of the corpus's reports carry no scope in their name at
  all, and for 38 issuer-years that unnamed filing is the only one there is, so
  serving it is the right answer rather than a miss.

    python scripts/audit_gate.py
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from docs import (
    load_companies, load_reports, parse_question, required_report_years, retrieve_docs,
)
from vifinqa.jsonl import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/vifinqa")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--show", type=int, default=15, help="examples to print per failure")
    args = parser.parse_args()

    questions = load_jsonl(args.data_root / "questions" / "questions.jsonl")
    if args.limit:
        questions = questions[: args.limit]
    companies = load_companies(args.data_root / "code_stock.csv")
    reports = load_reports(args.data_root / "financial_statements")
    have = {(report.ticker, report.year) for report in reports.values()}
    offered: dict[tuple[str, int], set] = {}
    for report in reports.values():
        offered.setdefault((report.ticker, report.year), set()).add(report.scope)

    counts, sizes = Counter(), Counter()
    examples: dict[str, list] = {}

    def note(kind, question, detail=""):
        counts[kind] += 1
        examples.setdefault(kind, []).append((question["id"], question["question"][:110], detail))

    for question in questions:
        parsed = parse_question(question["question"], companies)
        guessed = False
        if not parsed.tickers:
            if parsed.candidate_tickers:
                note("no ticker parsed, first candidate guessed", question,
                     f"guessed {parsed.candidate_tickers[0]}")
                parsed.tickers = parsed.candidate_tickers[:1]
                guessed = True
            else:
                note("no ticker at all", question)
        years = required_report_years(parsed)
        if not years:
            note("no year parsed, every year eligible", question)

        docs, decisions = retrieve_docs(parsed, reports)
        sizes[min(len(docs), 12)] += 1
        if not docs:
            note("no report selected", question, f"tickers={parsed.tickers} years={years}")
        for decision in decisions:
            if decision["selected_doc"] is None:
                pair = (decision["ticker"], decision["requested_year"])
                note("corpus has no filing for this issuer and year" if pair not in have
                     else "filing exists but no scope matched", question, f"{pair}")
            elif decision["scope"] and decision["selected_scope"] != decision["scope"]:
                choices = offered.get((decision["ticker"], decision["requested_year"]), set())
                note("served another scope while the asked one was on the shelf"
                     if decision["scope"] in choices else
                     "asked scope not in the corpus, served what there was", question,
                     f"asked {decision['scope']}, served {decision['selected_scope']}")
        if guessed:
            pass

    total = len(questions)
    print(json.dumps({
        "questions": total,
        "failures, as a share of questions":
            {kind: round(count / total, 4) for kind, count in counts.most_common()},
        "reports selected per question":
            {str(size): round(count / total, 3) for size, count in sorted(sizes.items())},
    }, ensure_ascii=False, indent=2))

    for kind, rows in examples.items():
        print(f"\n--- {kind} ({len(rows)}) ---")
        for identifier, text, detail in rows[: args.show]:
            print(f"  {identifier:>5}  {detail:<34}  {text}")


if __name__ == "__main__":
    main()
