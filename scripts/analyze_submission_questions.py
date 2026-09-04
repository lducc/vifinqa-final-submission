#!/usr/bin/env python3
"""Write a question-level retrieval and execution audit for a submission.

Gold-backed error labels are emitted only for questions in benchmark_v2. The
remaining questions receive observable diagnostics, never invented correctness
labels. Outputs are CSV for sorting, JSONL for inspection, and a Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from audit_questions import tier_of
from measure_questions import kind_of, tickers_of
from vifinqa.answers import fold
from vifinqa.slots import load_line_items, named_line_items, normalize


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def keyed_json(path: Path) -> dict[int, object]:
    return {int(key): value for key, value in json.loads(path.read_text("utf-8")).items()}


def package_rows(path: Path) -> dict[int, dict]:
    return {int(row["id"]): row for row in json.loads((path / "submission.json").read_text("utf-8"))}


def zip_rows(path: Path) -> dict[int, dict]:
    with zipfile.ZipFile(path) as archive:
        name = next(item for item in archive.namelist() if item.endswith("submission.json"))
        return {int(row["id"]): row for row in json.loads(archive.read(name))}


def f2(gold: set[str], predicted: list[str]) -> tuple[float, float, float, float, int | None]:
    unique = list(dict.fromkeys(predicted))
    hits = len(gold & set(unique))
    precision = hits / len(unique) if unique else 0.0
    recall = hits / len(gold) if gold else 0.0
    score = 5 * hits / (4 * len(gold) + len(unique)) if gold or unique else 0.0
    first = next((rank for rank, table in enumerate(unique, 1) if table in gold), None)
    return score, precision, recall, (1 / first if first and first <= 5 else 0.0), first


def logit(probability: float) -> float:
    probability = min(1 - 1e-9, max(1e-9, probability))
    return math.log(probability / (1 - probability))


def replay(row: dict, package: Path, tolerance: float) -> dict:
    query = row.get("pandas_query", "")
    if not query or query.startswith("result = 0 * df"):
        return {"status": "no_query", "actual": None, "relative_error": None, "detail": ""}
    try:
        import pandas as pd

        frames = {
            item["variable"]: pd.read_csv(package / item["csv_path"], dtype=str)
            for item in row.get("evidence", [])
        }
        local: dict = {}
        exec(f"import pandas as pd\n{query}", {"pd": pd, "dfs": frames, **frames}, local)
        expected, actual = float(row["answer"]), float(local["result"])
        relative = abs(actual - expected) / max(1.0, abs(expected))
        return {"status": "agreed" if relative <= tolerance else "disagreed",
                "actual": actual, "relative_error": relative, "detail": ""}
    except Exception as error:
        return {"status": "raised", "actual": None, "relative_error": None,
                "detail": repr(error)[:240]}


def bootstrap_delta(deltas: list[float], samples: int = 10_000) -> tuple[float, float]:
    random.seed(20260826)
    n = len(deltas)
    estimates = sorted(mean(deltas[random.randrange(n)] for _ in range(n)) for _ in range(samples))
    return estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]


def failure_label(table: str, pool: dict[str, dict], raw_rank: list[str],
                  pruned_rank: list[str], submitted: set[str]) -> str:
    if table in submitted:
        return "package_fallback_hit" if table not in pruned_rank else "submitted_hit"
    candidate = pool.get(table)
    if candidate is None:
        return "candidate_pool_miss"
    if table not in raw_rank and candidate.get("sparse_rank") is None:
        return "dense_only_absent_from_sparse_arm"
    if table not in pruned_rank:
        return "hard_prune_removed"
    return "budget_or_order_miss"


def fix_for(failures: Counter, false_positives: int, first_gold_rank: int | None,
            execution: str, labelled: bool) -> str:
    if not labelled:
        return ("repair_or_regenerate_answer_program; manual review only"
                if execution != "agreed" else "manual review only; hidden gold unavailable")
    fixes = []
    if failures["candidate_pool_miss"]:
        fixes.append("per-slot candidate generation / multi-page table recovery")
    elif failures["dense_only_absent_from_sparse_arm"]:
        fixes.append("selective dense recovery for uncovered report-item slots")
    elif failures["hard_prune_removed"]:
        fixes.append("remove hard lexical prune; use confidence-gated soft features")
    elif failures["budget_or_order_miss"]:
        fixes.append("confidence-gated slots with per-report cap, then preserve rerank order")
    elif first_gold_rank and first_gold_rank > 1:
        fixes.append("table-role/period rerank to move the first gold to rank 1")
    elif false_positives:
        fixes.append("stop forced padding; distinguish primary vs note/related-party tables")
    else:
        fixes.append("no retrieval error on reviewed labels")
    if execution != "agreed":
        fixes.append("repair or regenerate answer program")
    return "; ".join(fixes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=ROOT / "output/final_20260825_scored/submission_sparse_qwen_line_prune_auto/package")
    parser.add_argument("--strict-package", type=Path, default=ROOT / "output/final_20260825_scored/submission_sparse_qwen_line_prune_strict_auto/package")
    parser.add_argument("--raw-package", type=Path, default=ROOT / "output/final_20260825_scored/submission_sparse_qwen/package")
    parser.add_argument("--slots-zip", type=Path, default=ROOT / "output/submissions_table_ablation/zips/08_rrf_then_slots.zip")
    parser.add_argument("--pairs", type=Path, default=ROOT / "output/final_20260825_canonical/pairs_hybrid.jsonl")
    parser.add_argument("--graph-pairs", type=Path, default=ROOT / "output/final_20260825_canonical/pairs_hybrid_graph.jsonl")
    parser.add_argument("--scores", type=Path, nargs="+", default=[
        ROOT / "output/final_20260825_scored/scores_shard_00.jsonl",
        ROOT / "output/final_20260825_scored/scores_shard_01.jsonl",
    ])
    parser.add_argument("--raw-ranking", type=Path, default=ROOT / "output/final_20260825_scored/ranking_sparse_qwen.json")
    parser.add_argument("--pruned-ranking", type=Path, default=ROOT / "output/final_20260825_scored/ranking_sparse_qwen_line_prune.json")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/annotations/benchmark_v2/benchmark_v2.jsonl")
    parser.add_argument("--line-items", type=Path, default=ROOT / "data/derived/line_items.json")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "output/diagnostics/prune_auto_full_question_audit")
    parser.add_argument("--tolerance", type=float, default=2e-4)
    args = parser.parse_args()

    current = package_rows(args.package)
    strict = package_rows(args.strict_package)
    raw = package_rows(args.raw_package)
    slots = zip_rows(args.slots_zip)
    pairs = {int(row["id"]): row for row in jsonl(args.pairs)}
    graph_pairs = {int(row["id"]): row for row in jsonl(args.graph_pairs)}
    scores = {}
    for path in args.scores:
        scores.update({int(row["id"]): row["scores"] for row in jsonl(path)})
    raw_ranking, pruned_ranking = keyed_json(args.raw_ranking), keyed_json(args.pruned_ranking)
    benchmark = {int(row["id"]): row for row in jsonl(args.benchmark)}
    labels = load_line_items(args.line_items)
    tickers = tickers_of(ROOT / "data/raw/vifinqa")

    assert len(current) == 1012 and len(benchmark) == 233
    assert set(current) == set(pairs) == set(scores) == set(raw_ranking) == set(pruned_ranking)

    records, csv_rows = [], []
    for question_id in sorted(current):
        row = current[question_id]
        pair = pairs[question_id]
        candidate_by_id = {item["table_id"]: item for item in pair["candidates"]}
        graph_ids = {item["table_id"] for item in graph_pairs[question_id]["candidates"]}
        graph_added = graph_ids - candidate_by_id.keys()
        score_map = scores[question_id]
        score_order = sorted(score_map, key=lambda table: score_map[table], reverse=True)
        raw_position = {table: rank for rank, table in enumerate(raw_ranking[question_id], 1)}
        pruned_position = {table: rank for rank, table in enumerate(pruned_ranking[question_id], 1)}
        submitted = list(dict.fromkeys(row["relevant_tables"]))
        submitted_set = set(submitted)
        score_values = [score_map[table] for table in score_order]
        top_logit = logit(score_values[0]) if score_values else None
        low_confidence = [table for table in submitted if table in score_map and
                          top_logit - logit(score_map[table]) > 4]
        question_items = named_line_items(row["question"], labels)
        execution = replay(row, args.package, args.tolerance)
        kind, span = kind_of(row["question"], tickers)

        gold_row = benchmark.get(question_id)
        gold = set(gold_row["annotation"]["gold_tables_v2"]) if gold_row else set()
        hits = sorted(gold & submitted_set)
        misses = sorted(gold - submitted_set)
        false_positive_ids = [table for table in submitted if table not in gold] if gold_row else []
        failures = Counter(failure_label(table, candidate_by_id, raw_ranking[question_id],
                                         pruned_ranking[question_id], submitted_set)
                           for table in gold)
        current_metrics = f2(gold, submitted) if gold_row else (None,) * 5
        raw_metrics = f2(gold, raw[question_id]["relevant_tables"]) if gold_row else (None,) * 5
        strict_metrics = f2(gold, strict[question_id]["relevant_tables"]) if gold_row else (None,) * 5
        slot_metrics = f2(gold, slots[question_id]["relevant_tables"]) if gold_row else (None,) * 5
        _, precision, recall, mrr5, first_gold_rank = current_metrics

        years = set(part for part in fold(row["question"]).split() if len(part) == 4 and part.isdigit())
        false_positive_both = 0
        false_positive_details = []
        for table in false_positive_ids:
            text = normalize(candidate_by_id.get(table, {}).get("text", ""))
            has_year = not years or bool(years & set(text.split()))
            has_item = not question_items or any(f" {item} " in f" {text} " for item in question_items)
            false_positive_both += has_year and has_item
            false_positive_details.append({
                "table_id": table, "score": score_map.get(table),
                "raw_rank": raw_position.get(table), "pruned_rank": pruned_position.get(table),
                "matches_requested_year": has_year, "matches_named_line_item": has_item,
                "text_preview": candidate_by_id.get(table, {}).get("text", "")[:800],
            })

        flags = []
        if not gold_row:
            flags.append("unlabelled_no_correctness_claim")
        if len(row["relevant_docs"]) >= 3:
            flags.append("three_or_more_reports")
        if not question_items:
            flags.append("no_exact_catalogued_line_item")
        if low_confidence:
            flags.append("submitted_below_top_logit_gap4")
        if score_values and score_values[0] < 0.5:
            flags.append("low_top_reranker_score")
        if execution["status"] != "agreed":
            flags.append(f"execution_{execution['status']}")
        if gold_row and not hits:
            flags.append("zero_gold_tables_submitted")
        if gold_row and false_positive_both:
            flags.append("same_item_same_year_false_positive")

        top_candidates = []
        for table in raw_ranking[question_id][:10]:
            candidate = candidate_by_id.get(table, {})
            top_candidates.append({
                "table_id": table,
                "score": score_map.get(table),
                "sparse_rank": candidate.get("sparse_rank"),
                "dense_rank": candidate.get("dense_rank"),
                "survived_prune": table in pruned_ranking[question_id],
                "submitted": table in submitted_set,
                "gold": table in gold if gold_row else None,
                "text_preview": candidate.get("text", "")[:800],
            })

        recommended = fix_for(failures, len(false_positive_ids), first_gold_rank,
                              execution["status"], bool(gold_row))
        detail = {
            "id": question_id,
            "question": row["question"],
            "label_status": "reviewed_gold_v2" if gold_row else "unlabelled",
            "tier": gold_row.get("tier") if gold_row else tier_of(row["question"], tickers).lower(),
            "tier_is_proxy": not bool(gold_row),
            "source": gold_row.get("source") if gold_row else None,
            "taxonomy": gold_row.get("taxonomy") if gold_row else None,
            "query_shape": {"kind": kind, "span": span},
            "named_line_items": question_items,
            "selected_docs": row["relevant_docs"],
            "candidate_diagnostics": {
                "total": len(candidate_by_id),
                "sparse": sum(item.get("sparse_rank") is not None for item in candidate_by_id.values()),
                "dense": sum(item.get("dense_rank") is not None for item in candidate_by_id.values()),
                "dense_only": sum(item.get("dense_rank") is not None and item.get("sparse_rank") is None
                                  for item in candidate_by_id.values()),
                "graph_added": len(graph_added),
                "top_score": score_values[0] if score_values else None,
                "second_score": score_values[1] if len(score_values) > 1 else None,
                "top_two_logit_gap": (top_logit - logit(score_values[1])) if len(score_values) > 1 else None,
                "submitted_below_top_logit_gap4": low_confidence,
            },
            "submission": {
                "tables": submitted,
                "answer": row["answer"],
                "pandas_query": row.get("pandas_query", ""),
                "execution": execution,
            },
            "reviewed_error": None if not gold_row else {
                "gold_tables": sorted(gold),
                "hits": hits,
                "misses": misses,
                "false_positives": false_positive_ids,
                "false_positive_details": false_positive_details,
                "gold_table_outcomes": [{
                    "table_id": table,
                    "outcome": failure_label(table, candidate_by_id, raw_ranking[question_id],
                                             pruned_ranking[question_id], submitted_set),
                    "score": score_map.get(table),
                    "raw_rank": raw_position.get(table),
                    "pruned_rank": pruned_position.get(table),
                    "sparse_rank": candidate_by_id.get(table, {}).get("sparse_rank"),
                    "dense_rank": candidate_by_id.get(table, {}).get("dense_rank"),
                    "text_preview": candidate_by_id.get(table, {}).get("text", "")[:800],
                } for table in sorted(gold)],
                "failure_counts": dict(failures),
                "false_positives_matching_item_and_year": false_positive_both,
                "metrics": {"f2": current_metrics[0], "precision": precision,
                            "recall": recall, "mrr5": mrr5, "first_gold_rank": first_gold_rank},
                "comparisons": {
                    "raw_qwen_f2": raw_metrics[0],
                    "strict_auto_f2": strict_metrics[0],
                    "public_best_slots08_f2": slot_metrics[0],
                    "delta_vs_raw_qwen": current_metrics[0] - raw_metrics[0],
                    "delta_vs_slots08": current_metrics[0] - slot_metrics[0],
                },
            },
            "top_raw_qwen_candidates": top_candidates,
            "risk_flags": flags,
            "recommended_next_fix": recommended,
        }
        records.append(detail)
        csv_rows.append({
            "id": question_id, "question": row["question"], "label_status": detail["label_status"],
            "tier": detail["tier"], "tier_is_proxy": detail["tier_is_proxy"], "source": detail["source"],
            "query_kind": kind, "query_span": span, "named_line_items": json.dumps(question_items, ensure_ascii=False),
            "selected_doc_count": len(row["relevant_docs"]), "selected_docs": json.dumps(row["relevant_docs"], ensure_ascii=False),
            "candidate_count": len(candidate_by_id), "sparse_candidate_count": detail["candidate_diagnostics"]["sparse"],
            "dense_candidate_count": detail["candidate_diagnostics"]["dense"],
            "dense_only_candidate_count": detail["candidate_diagnostics"]["dense_only"],
            "graph_added_candidate_count": len(graph_added), "submitted_count": len(submitted),
            "submitted_tables": json.dumps(submitted, ensure_ascii=False), "answer": row["answer"],
            "execution_status": execution["status"], "execution_actual": execution["actual"],
            "execution_relative_error": execution["relative_error"], "top_score": detail["candidate_diagnostics"]["top_score"],
            "second_score": detail["candidate_diagnostics"]["second_score"],
            "top_two_logit_gap": detail["candidate_diagnostics"]["top_two_logit_gap"],
            "low_confidence_submitted_count": len(low_confidence),
            "gold_count": len(gold) if gold_row else "", "gold_tables": json.dumps(sorted(gold), ensure_ascii=False) if gold_row else "",
            "hit_count": len(hits) if gold_row else "", "hits": json.dumps(hits, ensure_ascii=False) if gold_row else "",
            "miss_count": len(misses) if gold_row else "", "misses": json.dumps(misses, ensure_ascii=False) if gold_row else "",
            "false_positive_count": len(false_positive_ids) if gold_row else "",
            "false_positives": json.dumps(false_positive_ids, ensure_ascii=False) if gold_row else "",
            "candidate_pool_miss_count": failures["candidate_pool_miss"] if gold_row else "",
            "dense_only_miss_count": failures["dense_only_absent_from_sparse_arm"] if gold_row else "",
            "hard_prune_removed_count": failures["hard_prune_removed"] if gold_row else "",
            "package_fallback_hit_count": failures["package_fallback_hit"] if gold_row else "",
            "budget_or_order_miss_count": failures["budget_or_order_miss"] if gold_row else "",
            "f2": current_metrics[0] if gold_row else "", "precision": precision if gold_row else "",
            "recall": recall if gold_row else "", "mrr5": mrr5 if gold_row else "",
            "first_gold_rank": first_gold_rank if first_gold_rank is not None else "", "raw_qwen_f2": raw_metrics[0] if gold_row else "",
            "strict_auto_f2": strict_metrics[0] if gold_row else "", "slots08_f2": slot_metrics[0] if gold_row else "",
            "delta_vs_raw_qwen": current_metrics[0] - raw_metrics[0] if gold_row else "",
            "risk_flags": json.dumps(flags, ensure_ascii=False), "recommended_next_fix": recommended,
        })

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    labelled = [row for row in csv_rows if row["label_status"] == "reviewed_gold_v2"]
    execution_counts = Counter(row["execution_status"] for row in csv_rows)
    failure_counts = Counter()
    for row in labelled:
        failure_counts.update({
            "candidate_pool_miss": row["candidate_pool_miss_count"],
            "dense_only_absent_from_sparse_arm": row["dense_only_miss_count"],
            "hard_prune_removed": row["hard_prune_removed_count"],
            "budget_or_order_miss": row["budget_or_order_miss_count"],
        })
    failure_counts["package_fallback_hit"] = sum(
        record["reviewed_error"]["failure_counts"].get("package_fallback_hit", 0)
        for record in records if record["reviewed_error"])
    failure_counts["submitted_hit"] = sum(row["hit_count"] for row in labelled) - failure_counts["package_fallback_hit"]
    deltas = [float(row["delta_vs_raw_qwen"]) for row in labelled]
    ci_low, ci_high = bootstrap_delta(deltas)
    zero = sorted((row for row in labelled if row["f2"] == 0),
                  key=lambda row: (-row["gold_count"], row["id"]))
    ranked = sorted(labelled, key=lambda row: (row["f2"], -row["gold_count"], row["id"]))
    wins = sum(delta > 1e-12 for delta in deltas)
    losses = sum(delta < -1e-12 for delta in deltas)
    false_positives = sum(row["false_positive_count"] for row in labelled)
    same_item_year_fp = sum(
        record["reviewed_error"]["false_positives_matching_item_and_year"]
        for record in records if record["reviewed_error"]
    )
    first_not_one = sum(isinstance(row["first_gold_rank"], int) and row["first_gold_rank"] > 1
                        for row in labelled)
    pool_miss_questions = sum(row["candidate_pool_miss_count"] > 0 for row in labelled)
    fixed_two_per_report = sum(row["submitted_count"] == 2 * row["selected_doc_count"]
                               for row in csv_rows)
    questions_with_fp = sum(row["false_positive_count"] > 0 for row in labelled)
    local_slots_f2 = mean(row["slots08_f2"] for row in labelled)
    unlabeled_risk = sorted((row for row in csv_rows if row["label_status"] == "unlabelled"),
                            key=lambda row: (-len(json.loads(row["risk_flags"])), row["id"]))[:50]

    lines = [
        "# Full question-level audit: sparse Qwen + prune-auto", "",
        "## Scope", "",
        f"- Audited all **{len(csv_rows):,}** submitted questions.",
        f"- **{len(labelled)}** have reviewed `benchmark_v2` table labels; only these receive table correctness/error labels.",
        f"- **{len(csv_rows) - len(labelled)}** have no available gold tables or gold answers; their rows contain observable risk and execution diagnostics only.",
        "- `benchmark_v2` is a reviewed reconstruction, not the organizer's hidden full-test gold. Local deltas are diagnostic, not leaderboard guarantees.", "",
        "## Aggregate findings", "",
        f"- Reviewed macro F2/P/R/MRR5: **{mean(row['f2'] for row in labelled):.4f} / {mean(row['precision'] for row in labelled):.4f} / {mean(row['recall'] for row in labelled):.4f} / {mean(row['mrr5'] for row in labelled):.4f}**.",
        "- Actual public result for this package: **F2 .5052 / P .3357 / R .5926 / MRR5 .6019 / answer-exec .1502**.",
        f"- Versus raw sparse-Qwen, mean paired F2 delta: **{mean(deltas):+.4f}**, question-bootstrap 95% CI **[{ci_low:+.4f}, {ci_high:+.4f}]**; {wins} wins, {losses} losses, {len(deltas)-wins-losses} ties.",
        f"- Gold accounting: **{failure_counts['submitted_hit']} normal hits + {failure_counts['package_fallback_hit']} fallback-recovered hits**, **{failure_counts['candidate_pool_miss']} pool misses**, **{failure_counts['dense_only_absent_from_sparse_arm']} dense-only omissions**, **{failure_counts['hard_prune_removed']} hard-prune removals that stayed lost**, **{failure_counts['budget_or_order_miss']} budget/order misses**.",
        f"- Pool misses are concentrated: **{failure_counts['candidate_pool_miss']} gold tables across only {pool_miss_questions} questions**. For most questions, generation is not the bottleneck.",
        f"- Submitted false positives: **{false_positives}**; at least **{same_item_year_fp}** match both a conservatively detected line item and requested year. Lexical pruning cannot distinguish their table role.",
        f"- **{fixed_two_per_report}/{len(csv_rows)}** questions submit exactly two tables per selected report. On the reviewed set, **{questions_with_fp}/{len(labelled)}** questions contain at least one false positive; 79 questions need only one gold table, so fixed `2 × reports` structurally caps their precision.",
        f"- **{first_not_one}** reviewed questions have a retrieved gold table below rank 1, leaving a direct MRR/rerank opportunity.",
        f"- Execution replay: **{execution_counts['agreed']} agreed**, **{execution_counts['disagreed']} disagreed**, **{execution_counts['raised']} raised**, **{execution_counts['no_query']} had no executable query**.",
        f"- Zero-table-hit reviewed questions: **{len(zero)}/{len(labelled)}**.", "",
        "The dominant actionable error is selection after retrieval: most missed reviewed gold tables were already in the pool, while 27 gold tables were touched by hard pruning (23 lost, 4 recovered only by fallback). The prune-auto delta is statistically inconclusive and its public F2 regressed, so it should not be the base architecture.", "",
        "## Performance slices", "",
        "| Slice | n | F2 | P | R | MRR5 | Zero-hit | Full recall | Delta vs raw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for field in ("tier", "query_kind"):
        for value in sorted({str(row[field]) for row in labelled}):
            group = [row for row in labelled if str(row[field]) == value]
            lines.append(
                f"| {field}={value} | {len(group)} | {mean(row['f2'] for row in group):.3f} | "
                f"{mean(row['precision'] for row in group):.3f} | {mean(row['recall'] for row in group):.3f} | "
                f"{mean(row['mrr5'] for row in group):.3f} | {sum(row['f2'] == 0 for row in group)} | "
                f"{sum(row['recall'] == 1 for row in group)} | {mean(row['delta_vs_raw_qwen'] for row in group):+.3f} |"
            )
    for label, predicate in (
        ("reports=1", lambda row: row["selected_doc_count"] == 1),
        ("reports=2", lambda row: row["selected_doc_count"] == 2),
        ("reports=3–4", lambda row: 3 <= row["selected_doc_count"] <= 4),
        ("reports=5+", lambda row: row["selected_doc_count"] >= 5),
    ):
        group = [row for row in labelled if predicate(row)]
        lines.append(
            f"| {label} | {len(group)} | {mean(row['f2'] for row in group):.3f} | "
            f"{mean(row['precision'] for row in group):.3f} | {mean(row['recall'] for row in group):.3f} | "
            f"{mean(row['mrr5'] for row in group):.3f} | {sum(row['f2'] == 0 for row in group)} | "
            f"{sum(row['recall'] == 1 for row in group)} | {mean(row['delta_vs_raw_qwen'] for row in group):+.3f} |"
        )
    lines += [
        "", "Key slice failures: peer questions are worst, 3+ report questions fall sharply, and intermediate questions regress under pruning. Easy single-report lookup is already strong locally; do not redesign it.", "",
        "## What the full audit changes", "",
        f"1. **Return to public-best `08_rrf_then_slots` as the base.** The reviewed set scores prune-auto {mean(row['f2'] for row in labelled):.4f} versus slots08 {local_slots_f2:.4f}, but public scoring reverses them: .5052 versus .5251. The reconstruction cannot choose close variants reliably.",
        "2. **Replace fixed `2 × reports` with slot-derived variable cardinality.** Build explicit `(report, year, operand/line-item, table-role)` slots. One-value questions normally emit one table; multi-year, peer, ratio, conditional, and superlative questions emit the union of their required slots.",
        "3. **Use Qwen per slot, then confidence-gate; do not hard-prune.** Keep the top table for each covered slot, admit additional tables only inside a calibrated score gap, preserve the original Qwen/RRF order for MRR, and never pad a slot with low-confidence tables.",
        "4. **Dense retrieval is recovery, not global fusion.** Invoke it only for an uncovered slot/report. It owns 23 reviewed gold tables but also thousands of non-gold candidates, so unconditional hybrid fusion destroys precision.",
        "5. **Add table-role evidence, not a broad bank/non-bank topic classifier.** The reranker must distinguish primary statement, note, movement schedule, related-party, segment, and cash-flow roles. Same item/year text appears in most distractors, which is why lexical pruning failed.",
        "6. **Route high-arity questions to structured decomposition.** The 99 pool misses occur in only 22 questions, led by multi-year/peer conditional questions. Decomposition is needed here for retrieval coverage; it is unnecessary overhead for easy single-table questions.",
        "7. **Repair answer programs independently.** Eighteen questions have no executable query and twelve disagree with their stored answer. Table gains cannot fix those thirty direct execution defects or the broader hidden-answer error rate.", "",
        "The next ablation should therefore be exactly one change over package 08: confidence-gated, slot-derived variable output size. Hold the candidates, Qwen scores, answer programs, and ordering constant. Because the local benchmark reverses close public variants, treat its slice analysis as causal diagnosis but require public confirmation for the final threshold.", "",
        "## Zero-hit reviewed questions", "",
        "| ID | Tier | Gold | Pool miss | Dense-only | Pruned | Budget/order | Question |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in zero:
        lines.append(f"| {row['id']} | {row['tier']} | {row['gold_count']} | {row['candidate_pool_miss_count']} | {row['dense_only_miss_count']} | {row['hard_prune_removed_count']} | {row['budget_or_order_miss_count']} | {row['question'].replace('|', '/')} |")
    lines += ["", "## All 233 reviewed questions (worst first)", "",
              "| ID | F2 | P | R | First gold | Misses | FP | Delta vs raw | Recommended fix |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in ranked:
        first = row["first_gold_rank"] if row["first_gold_rank"] != "" else "—"
        lines.append(f"| {row['id']} | {row['f2']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {first} | {row['miss_count']} | {row['false_positive_count']} | {row['delta_vs_raw_qwen']:+.3f} | {row['recommended_next_fix']} |")
    execution_failures = [row for row in csv_rows if row["execution_status"] != "agreed"]
    lines += ["", "## All execution failures", "",
              "These detect self-inconsistency or missing programs, not correctness against hidden answers.", "",
              "| ID | Status | Stored answer | Replayed answer | Question |",
              "|---:|---|---:|---:|---|"]
    for row in execution_failures:
        actual = row["execution_actual"] if row["execution_actual"] != "" else "—"
        lines.append(f"| {row['id']} | {row['execution_status']} | {row['answer']} | {actual} | {row['question'].replace('|', '/')} |")
    lines += ["", "## Highest observable-risk unlabelled questions", "",
              "These are a manual-review queue, not inferred errors.", "",
              "| ID | Reports | Candidates | Submitted | Execution | Flags | Question |",
              "|---:|---:|---:|---:|---|---|---|"]
    for row in unlabeled_risk:
        flags = ", ".join(json.loads(row["risk_flags"]))
        lines.append(f"| {row['id']} | {row['selected_doc_count']} | {row['candidate_count']} | {row['submitted_count']} | {row['execution_status']} | {flags} | {row['question'].replace('|', '/')} |")
    lines += ["", "## Files", "",
              f"- CSV: `{prefix.with_suffix('.csv').relative_to(ROOT)}`",
              f"- JSONL with top-10 candidate evidence per question: `{prefix.with_suffix('.jsonl').relative_to(ROOT)}`",
              "", "The CSV/JSONL contain every question. Blank correctness columns mean unavailable gold, not a correct result."]
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "questions": len(csv_rows), "reviewed": len(labelled), "unlabelled": len(csv_rows) - len(labelled),
        "reviewed_f2": mean(row["f2"] for row in labelled),
        "delta_vs_raw": mean(deltas), "delta_bootstrap_95ci": [ci_low, ci_high],
        "execution": execution_counts, "gold_accounting": failure_counts,
        "outputs": [str(prefix.with_suffix(suffix)) for suffix in (".csv", ".jsonl", ".md")],
    }, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
