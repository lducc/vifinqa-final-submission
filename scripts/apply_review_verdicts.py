#!/usr/bin/env python3
"""Fold adjudicated table-relevance verdicts into a new benchmark version.

The 233-question labels were discovered from one public submission's
predictions. They are the right size — 3.24 binding tables per question against
the organizer's 3.27 — but the wrong contents: on the same packages they score
precision 0.4412 where the organizer scores 0.3516, and recall 0.7634 where the
organizer scores 0.6148. They agree with our own retriever more than the
organizer does.

This script does not re-derive labels. It applies judgements made against the
source reports to produce a new versioned annotation file, and records what the
new version can and cannot be trusted to decide.

Only `relevant` verdicts widen the gold set by default. `uncertain` verdicts are
carried through as a separate reviewable list rather than being silently
resolved in either direction; `--include-uncertain` folds them in for a
sensitivity run.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


VERDICTS = {"relevant", "not_relevant", "uncertain"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verdicts(path: Path) -> list[dict]:
    records, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record["verdict"] not in VERDICTS:
                raise SystemExit(f"{path}:{number}: unknown verdict {record['verdict']!r}")
            key = (record["question_id"], record["table"])
            if key in seen:
                raise SystemExit(f"{path}:{number}: duplicate verdict for {key}")
            seen.add(key)
            records.append(record)
    return records


def bound_tables(annotation: dict) -> list[str]:
    bound = list(dict.fromkeys(
        binding["table"] for binding in annotation.get("row_column_bindings", [])
        if binding.get("table")
    ))
    return bound or list(dict.fromkeys(annotation["gold_tables"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path,
                        default=Path("attic/annotations/benchmark.jsonl"))
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--include-uncertain", action="store_true",
                        help="also widen gold with uncertain verdicts (sensitivity run)")
    args = parser.parse_args()

    base = [json.loads(line) for line in
            args.benchmark.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {record["id"]: record for record in base}

    verdicts = load_verdicts(args.verdicts)
    unknown = {v["question_id"] for v in verdicts} - by_id.keys()
    if unknown:
        raise SystemExit(f"verdicts reference {len(unknown)} unknown question ids: "
                         f"{sorted(unknown)[:5]}")

    widen = {"relevant"} | ({"uncertain"} if args.include_uncertain else set())
    additions: dict[int, list[dict]] = defaultdict(list)
    deferred: dict[int, list[dict]] = defaultdict(list)
    counts = defaultdict(int)
    for record in verdicts:
        counts[record["verdict"]] += 1
        counts[(record["verdict"], record["tier"])] += 1
        if record["verdict"] in widen:
            additions[record["question_id"]].append(record)
        elif record["verdict"] == "uncertain":
            deferred[record["question_id"]].append(record)

    added = 0
    with args.output.open("w", encoding="utf-8") as out:
        for record in base:
            annotation = record["annotation"]
            gold = bound_tables(annotation)
            new = [entry for entry in additions.get(record["id"], [])
                   if entry["table"] not in set(gold)]
            annotation["gold_tables_v2"] = gold + [entry["table"] for entry in new]
            annotation["added_in_v2"] = [
                {"table": entry["table"], "confidence": entry["confidence"],
                 "reason": entry["reason"]}
                for entry in new
            ]
            annotation["deferred_uncertain"] = [
                {"table": entry["table"], "reason": entry["reason"]}
                for entry in deferred.get(record["id"], [])
            ]
            added += len(new)
            record["label_version"] = "benchmark-v2-disputed-pool"
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    sizes = [len(json.loads(line)["annotation"]["gold_tables_v2"])
             for line in args.output.read_text(encoding="utf-8").splitlines()]
    manifest = {
        "label_version": "benchmark-v2-disputed-pool",
        "base": {"path": str(args.benchmark), "sha256": sha256(args.benchmark),
                 "questions": len(base)},
        "verdicts": {"path": str(args.verdicts), "sha256": sha256(args.verdicts),
                     "judged": len(verdicts)},
        "verdict_counts": {key: value for key, value in counts.items()
                           if isinstance(key, str)},
        "verdict_counts_by_tier": {f"{key[0]}/{key[1]}": value
                                   for key, value in counts.items()
                                   if isinstance(key, tuple)},
        "uncertain_included": args.include_uncertain,
        "labels_added": added,
        "gold_tables_per_question": round(sum(sizes) / len(sizes), 4),
        "organizer_gold_tables_per_question": 3.27,
        "output_sha256": sha256(args.output),
        "scope": (
            "Judged pool is the set of tables submission 08 retrieves and "
            "submission 00 does not, on the 233 benchmark questions. It is not a "
            "symmetric pool over all retrievers."
        ),
        "known_limitations": [
            "Pooled from one arm only, so labels favour tables that arm retrieves.",
            "With the pooled arm held out, correlation against public scores does "
            "not improve; this version repairs specific labels, it does not make "
            "the benchmark able to rank close arms.",
            "Use as a harm gate and for the 00-versus-08 comparison, not to select "
            "among half-point variants.",
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")

    print(f"questions written : {len(base)}", file=sys.stderr)
    print(f"labels added      : {added}", file=sys.stderr)
    print(f"gold tables/q     : {manifest['gold_tables_per_question']} "
          f"(organizer 3.27)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
