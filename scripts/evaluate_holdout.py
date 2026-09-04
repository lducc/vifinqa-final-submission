#!/usr/bin/env python3
"""Score retrieval on questions written from tables we know, on reports never trained on.

This project has been running without an instrument. The released questions
carry no gold tables and our own labels are quarantined, so every decision has
been argued rather than measured. But the generator produces labelled data by
construction: a question written from three tables has those three tables as its
gold, and nothing about that label is a judgement of ours.

`sample_groups.py --side holdout` draws groups from the fifth of the corpus the
sampler withholds, so questions written from them test reports the reranker
never saw. That makes an evaluation set, and it is the only one available here.

Two things it measures, and they cost very differently.

  **First stage, free.** Recall at depth for the gated BM25 pool, the dense
  pool, and their union. No GPU at all, and it answers the question the dense
  index was built to answer — does it find gold the gate misses — with a number
  rather than an argument.

  **Whole pipeline, cheap.** Given a --ranking built from scored pairs, the F2
  at the submitted budget, per arm. That needs the holdout questions to have
  been scored, which is why a slice of them belongs in the same scoring run as
  the test questions rather than in a run of their own.

What it is not: ground truth. The questions are a 14B model's, the round-trip
filter is not applied here on purpose, and some gold is unreachable by any
retriever. That last point is the reason it still works — an unreachable gold
table penalises every arm equally, so differences between arms remain readable
even where the absolute number is depressed. Read it for comparisons, never as
a score.

    python scripts/evaluate_holdout.py --dense output/dense
    python scripts/evaluate_holdout.py --dense output/dense --ranking output/rerank/ranking_hybrid.json
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from docs import load_companies, load_reports as load_doc_reports, parse_question, required_report_years, retrieve_docs
from audit_questions import tier_of
from export_rerank_pairs import load_dense
from vifinqa.jsonl import load_jsonl
from vifinqa.retrieval import load_reports, retrieve_rows, table_budget
from vifinqa.scoring import mean, prefix_score
from measure_questions import tickers_of


def load_dense_questions(directory: Path) -> dict:
    """Question vectors alone, from a run that embedded no tables."""
    import numpy as np

    identifiers = json.loads((directory / "question_ids.json").read_text("utf-8"))
    return {"questions": np.load(directory / "question_vectors.npy"),
            "order": {identity: row for row, identity in enumerate(identifiers)}}


def recall(gold: set, ranked: list, depth: int) -> float:
    return len(gold & set(ranked[:depth])) / max(1, len(gold))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/vifinqa")
    parser.add_argument("--tasks", type=Path, default=ROOT / "output/rerank/tasks_holdout.jsonl")
    parser.add_argument("--questions", type=Path,
                        default=ROOT / "output/rerank/questions_holdout.jsonl")
    parser.add_argument("--dense", type=Path, help="index directory; omitted, only the sparse arm is measured")
    parser.add_argument("--dense-questions", type=Path,
                        help="where the held-out questions' vectors live, if not beside "
                             "the tables; the corpus index embeds the released questions "
                             "and these are written later by a second short run")
    parser.add_argument("--depth", type=int, default=50, help="sparse candidates per question")
    parser.add_argument("--dense-depth", type=int, default=100)
    parser.add_argument("--ranking", type=Path, help="ranking.json to score at the submitted budget")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import numpy as np

    tasks = {task["task_id"]: task for task in load_jsonl(args.tasks)}
    gold_of = {identifier: {table["table_id"] for table in task["tables"]}
               for identifier, task in tasks.items()}
    asked = [record for record in load_jsonl(args.questions)
             if record["task_id"] in gold_of]
    if args.limit:
        asked = asked[: args.limit]
    if not asked:
        raise SystemExit(f"no questions in {args.questions}")

    companies = load_companies(args.dataset_root / "code_stock.csv")
    tickers = tickers_of(args.dataset_root)
    doc_reports = load_doc_reports(args.dataset_root / "financial_statements")
    table_reports = load_reports(args.dataset_root)
    dense = load_dense(args.dense) if args.dense else None
    if dense is not None and args.dense_questions:
        other = load_dense_questions(args.dense_questions)
        dense["questions"], dense["order"] = other["questions"], other["order"]
    ranking = json.loads(args.ranking.read_text("utf-8")) if args.ranking else None

    scores = {arm: [] for arm in ("sparse", "dense", "hybrid")}
    by_stratum: dict[str, dict[str, list[float]]] = {}
    budgeted, reachable = [], Counter()
    for number, record in enumerate(asked, 1):
        gold = gold_of[record["task_id"]]
        parsed = parse_question(record["question"], companies)
        if not parsed.tickers and parsed.candidate_tickers:
            parsed.tickers = parsed.candidate_tickers[:1]
        docs, _ = retrieve_docs(parsed, doc_reports)
        result = retrieve_rows(
            record["question"],
            {"tickers": parsed.tickers, "years": parsed.years,
             "slot_years": required_report_years(parsed), "scope": parsed.scope},
            table_reports, top_k=args.depth, report_ids=docs, mode="report-coverage",
        )
        sparse = [table["table_id"] for table in result["tables"]]

        ranked_dense = []
        if dense is not None:
            row = dense["order"].get(record["task_id"])
            if row is not None:
                similarity = (dense["tables"].astype(np.float32)
                              @ dense["questions"][row].astype(np.float32))
                ranked_dense = [dense["ids"][index][1]
                                for index in np.argsort(-similarity)[: args.dense_depth]]

        scores["sparse"].append(recall(gold, sparse, args.depth))
        scores["dense"].append(recall(gold, ranked_dense, args.dense_depth))
        union = sparse + [t for t in ranked_dense if t not in set(sparse)]
        scores["hybrid"].append(recall(gold, union, len(union)))
        task = tasks[record["task_id"]]
        stratum = f"{task.get('kind', 'single')} / {tier_of(record['question'], tickers)}"
        bucket = by_stratum.setdefault(stratum, {arm: [] for arm in scores} | {"f2": []})
        for arm in scores:
            bucket[arm].append(scores[arm][-1])
        reachable["sparse only" if scores["sparse"][-1] and not scores["dense"][-1] else
                  "dense only" if scores["dense"][-1] and not scores["sparse"][-1] else
                  "both" if scores["sparse"][-1] else "neither"] += 1

        if ranking is not None:
            ordered = ranking.get(str(record["task_id"]), [])
            f2 = prefix_score(sorted(gold), ordered, table_budget(len(docs), "auto"))["f2"]
            budgeted.append(f2)
            bucket["f2"].append(f2)
        if number % 100 == 0:
            print(f"  {number}/{len(asked)} questions", file=sys.stderr, flush=True)

    report = {
        "questions": len(asked),
        "gold tables per question": round(mean([len(gold_of[r["task_id"]]) for r in asked]), 2),
        f"recall of the gated BM25 pool at {args.depth}": round(mean(scores["sparse"]), 4),
    }
    if dense is not None:
        report[f"recall of the dense pool at {args.dense_depth}"] = round(mean(scores["dense"]), 4)
        report["recall of the union"] = round(mean(scores["hybrid"]), 4)
        report["which pool found gold"] = dict(reachable.most_common())
    if ranking is not None:
        report["F2 of the supplied ranking at the submitted budget"] = round(mean(budgeted), 4)
    report["by question stratum"] = {
        name: {
            "questions": len(bucket["sparse"]),
            "sparse recall": round(mean(bucket["sparse"]), 4),
            **({"dense recall": round(mean(bucket["dense"]), 4),
                "hybrid recall": round(mean(bucket["hybrid"]), 4)} if dense is not None else {}),
            **({"ranking F2": round(mean(bucket["f2"]), 4)} if ranking is not None else {}),
        }
        for name, bucket in sorted(by_stratum.items())
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
