#!/usr/bin/env python3
"""Measure sparse/dense candidate coverage on reviewed table evidence.

This is a diagnostic over our reviewed 150-question evidence, not organizer
gold. It is useful for finding a broken candidate stage; it is not a leaderboard
estimate and must not be used for a hyperparameter sweep.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from vifinqa.fusion import first_stage
from vifinqa.jsonl import load_jsonl


def gold_tables(path: Path) -> dict[int, set[str]]:
    grouped: dict[int, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[int(row["question_id"])].add(row["table"])
    return dict(grouped)


def arm_tables(record: dict, arm: str) -> list[str]:
    candidates = [
        candidate for candidate in record["candidates"]
        if first_stage(candidate, arm) > 0
    ]
    return [
        candidate["table_id"]
        for candidate in sorted(candidates, key=lambda item: -first_stage(item, arm))
    ]


def load_pairs(path: Path, expected_ids: set[int]) -> dict[int, dict]:
    """Load a complete candidate export without silently overwriting records."""
    pairs = {}
    for record in load_jsonl(path):
        identifier = int(record["id"])
        if identifier in pairs:
            raise ValueError(f"duplicate question id {identifier} in {path}")
        pairs[identifier] = record
    missing = expected_ids - pairs.keys()
    if missing:
        sample = ", ".join(map(str, sorted(missing)[:10]))
        raise ValueError(
            f"{path} is missing {len(missing)} expected questions: {sample}"
        )
    return pairs


def coverage(
    gold: dict[int, set[str]], pairs: dict[int, dict], arm: str,
    budget_multiplier: int = 2,
) -> dict:
    recalls, recall_at_5, reciprocal_ranks, budgeted_f2, counts = [], [], [], [], []
    any_gold = all_gold = 0
    for identifier, expected in gold.items():
        record = pairs[identifier]
        returned = arm_tables(record, arm)
        found = expected & set(returned)
        recalls.append(len(found) / len(expected))
        head = returned[:5]
        recall_at_5.append(len(expected & set(head)) / len(expected))
        reciprocal_ranks.append(next(
            (1 / rank for rank, table_id in enumerate(head, 1) if table_id in expected), 0,
        ))
        budget = min(30, max(1, budget_multiplier * len(record.get("selected_docs", []))))
        submitted = returned[:budget]
        hits = len(expected & set(submitted))
        budgeted_f2.append(5 * hits / (4 * len(expected) + len(submitted)))
        counts.append(len(returned))
        any_gold += bool(found)
        all_gold += found == expected
    total = len(gold)
    return {
        "questions": total,
        "mean_candidates": round(sum(counts) / total, 2),
        "mean_gold_recall": round(sum(recalls) / total, 4),
        "mean_gold_recall_at_5": round(sum(recall_at_5) / total, 4),
        "mrr5": round(sum(reciprocal_ranks) / total, 4),
        "budget_multiplier": budget_multiplier,
        "f2_at_report_budget": round(sum(budgeted_f2) / total, 4),
        "questions_with_any_gold": round(any_gold / total, 4),
        "questions_with_all_gold": round(all_gold / total, 4),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--budget-multiplier", type=int, default=2)
    parser.add_argument(
        "--evidence", type=Path,
        default=root / "data" / "results" / "e027_gold_150" / "evidence.csv",
    )
    args = parser.parse_args()

    gold = gold_tables(args.evidence)
    pairs = load_pairs(args.pairs, set(gold))
    print(json.dumps(
        {
            arm: coverage(gold, pairs, arm, args.budget_multiplier)
            for arm in ("sparse", "dense", "hybrid")
        },
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
