#!/usr/bin/env python3
"""Ask what the dense index returns that the report gate did not, before paying to score it.

There is no offline scorer on this task — the released questions carry no gold
tables — so the usual question, "does the dense stage help", cannot be answered
without a submission. This answers a smaller one that costs nothing and still
decides whether the scoring run is worth starting: *where* do the dense
candidates come from relative to the gate.

Three outcomes, and they call for three different things.

  Mostly inside the gated reports. The index works and agrees with the gate, so
  it can only reorder within a set the gate already chose. Little to gain, and
  the hybrid arm should be expected to land on top of the sparse arm.

  Mostly outside, and concentrated on the right issuer or the right year. It is
  finding filings the gate ruled out — which is the entire thesis, because a
  gate miss is the one failure no reranker downstream can repair.

  Mostly outside, and scattered across unrelated issuers and years. The index is
  noise. Ship the sparse arm and skip the scoring run.

The question's ticker and year are taken from the same parser the gate uses, so
this compares the dense index against the system as it actually runs rather than
against an idealised one.

    python scripts/audit_dense_overlap.py --dense output/dense
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from docs import load_companies, load_reports as load_doc_reports, parse_question, retrieve_docs
from vifinqa.jsonl import load_jsonl

sys.path.insert(0, str(ROOT / "scripts"))
from export_rerank_pairs import load_dense


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/vifinqa")
    parser.add_argument("--dense", type=Path, default=ROOT / "output/dense")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import numpy as np

    dense = load_dense(args.dense)
    questions = load_jsonl(args.dataset_root / "questions" / "questions.jsonl")
    if args.limit:
        questions = questions[: args.limit]
    companies = load_companies(args.dataset_root / "code_stock.csv")
    doc_reports = load_doc_reports(args.dataset_root / "financial_statements")

    tables = dense["tables"].astype(np.float32)
    placement, top_one, covered = Counter(), Counter(), []
    for number, question in enumerate(questions, 1):
        row = dense["order"].get(question["id"])
        if row is None:
            continue
        parsed = parse_question(question["question"], companies)
        if not parsed.tickers and parsed.candidate_tickers:
            parsed.tickers = parsed.candidate_tickers[:1]
        gated, _ = retrieve_docs(parsed, doc_reports)
        gated = set(gated)
        wanted_tickers = set(parsed.tickers)
        wanted_years = {str(year) for year in parsed.years}

        scores = tables @ dense["questions"][row].astype(np.float32)
        inside = 0
        for rank, index in enumerate(np.argsort(-scores)[: args.depth], 1):
            report_id, _ = dense["ids"][index]
            if report_id in gated:
                where = "inside the gate"
                inside += 1
            else:
                ticker = report_id.split("_")[0]
                year = next((part for part in report_id.split("_") if part.isdigit()), "")
                scope = ("separate" if "separate" in report_id
                         else "consolidated" if "consolidated" in report_id else "")
                if ticker in wanted_tickers and year in wanted_years:
                    # The gate filters on scope as well, so a report with the right
                    # ticker and year that it still rejected is usually the other
                    # scope — which it was right to reject if the question named
                    # one. Only a rejection at the *same* scope is a gate miss,
                    # and those are the corpus's multi-file filings.
                    where = ("right issuer and year, other scope"
                             if parsed.scope and scope and scope != parsed.scope
                             else "right issuer, year and scope, gate said no")
                elif ticker in wanted_tickers:
                    where = "right issuer, other year"
                elif year in wanted_years:
                    where = "other issuer, right year"
                else:
                    where = "unrelated issuer and year"
            placement[where] += 1
            if rank == 1:
                top_one[where] += 1
        covered.append(inside / max(1, args.depth))
        if number % 100 == 0:
            print(f"  {number}/{len(questions)} questions", file=sys.stderr, flush=True)

    total = sum(placement.values())
    print(json.dumps({
        "questions": len(covered),
        "depth": args.depth,
        "mean share of the dense pool inside the gated reports":
            round(sum(covered) / max(1, len(covered)), 3),
        "where every dense candidate sits":
            {key: round(count / max(1, total), 3) for key, count in placement.most_common()},
        "where the dense top-1 sits":
            {key: round(count / max(1, len(covered)), 3) for key, count in top_one.most_common()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
