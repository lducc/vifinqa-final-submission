#!/usr/bin/env python3
"""Measure whether a benchmark version can reproduce known public rankings.

Nine full packages have recorded organizer scores. That makes the benchmark
itself falsifiable: score the same packages locally, and correlate. A label set
that ranks those nine backwards cannot be trusted to rank a tenth.

The distinction that matters is between the wide spread and the close spread.
Across all nine arms the spread in public F2 is about 0.10, and almost any label
set separates the weak arm from the rest. Among arms 02 to 08 the spread is
0.016, which is the scale of every decision the benchmark is actually asked to
make. Report both, and treat only the close-arm number as the acceptance gate.

Public values come from `output/submissions_table_ablation/MANIFEST.md`. They
are recorded here rather than parsed so that a manifest edit cannot silently
change a gate result.
"""

import argparse
import json
from pathlib import Path
from statistics import mean
import sys
import zipfile


PUBLIC_F2 = {
    "00": 0.5221, "01": 0.4203, "02": 0.5090, "03": 0.5210, "04": 0.5204,
    "05": 0.5214, "06": 0.5210, "07": 0.5197, "08": 0.5251,
}
CLOSE_ARMS = ["02", "03", "04", "05", "06", "07", "08"]


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    out = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for index in range(position, end + 1):
            out[order[index]] = average
        position = end + 1
    return out


def correlate(x: list[float], y: list[float], spearman: bool = False) -> float:
    if spearman:
        x, y = ranks(x), ranks(y)
    mx, my = mean(x), mean(y)
    covariance = sum((a - mx) * (b - my) for a, b in zip(x, y))
    spread = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return covariance / spread if spread else float("nan")


def gold_sets(path: Path, field: str) -> dict[int, set[str]]:
    gold = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        annotation = record["annotation"]
        if field in annotation:
            tables = annotation[field]
        else:
            tables = [binding["table"] for binding
                      in annotation.get("row_column_bindings", []) if binding.get("table")]
            tables = tables or annotation["gold_tables"]
        gold[record["id"]] = set(tables)
    return gold


def submitted(zip_path: Path) -> dict[int, list[str]]:
    archive = zipfile.ZipFile(zip_path)
    name = next(entry for entry in archive.namelist() if entry.endswith("submission.json"))
    return {row["id"]: list(dict.fromkeys(row["relevant_tables"]))
            for row in json.loads(archive.read(name))}


def score(gold: dict[int, set[str]], predictions: dict[int, list[str]]) -> tuple[float, float, float]:
    """Macro F2 over questions, alongside macro precision and recall.

    The organizer averages a per-question F2, so the F2 has to be computed
    inside the loop and averaged afterwards. Combining the averaged precision
    and recall instead is a different statistic: it lets a long submission on
    one question offset a short one on another, which the real metric never
    does. Per question the identity is 5h / (4G + k), so a question with gold
    and nothing submitted scores zero rather than being skipped.
    """
    f2s, precisions, recalls = [], [], []
    for question_id, tables in gold.items():
        predicted = set(predictions.get(question_id, []))
        hits = len(predicted & tables)
        precisions.append(hits / len(predicted) if predicted else 0.0)
        recalls.append(hits / len(tables) if tables else 0.0)
        denominator = 4 * len(tables) + len(predicted)
        f2s.append(5 * hits / denominator if denominator else 0.0)
    return mean(f2s), mean(precisions), mean(recalls)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--gold-field", default="gold_tables_v2",
                        help="annotation field holding the gold set; falls back to bindings")
    parser.add_argument("--zips", type=Path,
                        default=Path("output/submissions_table_ablation/zips"))
    parser.add_argument("--hold-out", nargs="*", default=[],
                        help="arms to exclude, for checking pooling circularity")
    parser.add_argument("--min-close-spearman", type=float, default=None,
                        help="exit non-zero if the close-arm Spearman falls below this")
    args = parser.parse_args()

    gold = gold_sets(args.benchmark, args.gold_field)
    local = {}
    for path in sorted(args.zips.glob("*.zip")):
        arm = path.name[:2]
        if arm in PUBLIC_F2:
            local[arm] = score(gold, submitted(path))

    missing = set(PUBLIC_F2) - local.keys()
    if missing:
        raise SystemExit(f"missing packages for arms {sorted(missing)}")

    sizes = [len(tables) for tables in gold.values()]
    print(f"benchmark        : {args.benchmark}")
    print(f"gold field       : {args.gold_field}")
    print(f"questions        : {len(gold)}")
    print(f"gold tables/q    : {mean(sizes):.4f}   (organizer 3.27)")
    print()
    print(f"{'arm':>4} {'local F2':>9} {'local P':>8} {'local R':>8} | {'public F2':>9}")
    for arm in sorted(PUBLIC_F2):
        f2, precision, recall = local[arm]
        print(f"{arm:>4} {f2:9.4f} {precision:8.4f} {recall:8.4f} | {PUBLIC_F2[arm]:9.4f}")
    print()

    held = set(args.hold_out)
    if held:
        print(f"holding out arms : {sorted(held)}")
    close_result = None
    for label, arms in (("all arms", sorted(PUBLIC_F2)), ("close arms", CLOSE_ARMS)):
        arms = [arm for arm in arms if arm not in held]
        if len(arms) < 3:
            continue
        x = [local[arm][0] for arm in arms]
        y = [PUBLIC_F2[arm] for arm in arms]
        pearson, spearman = correlate(x, y), correlate(x, y, spearman=True)
        print(f"  {label:>10} (n={len(arms)}): Pearson {pearson:+.4f}   Spearman {spearman:+.4f}")
        if label == "close arms":
            close_result = spearman

    if args.min_close_spearman is not None:
        if close_result is None:
            print("\nno close-arm result to gate on", file=sys.stderr)
            return 2
        if close_result < args.min_close_spearman:
            print(f"\nFAIL: close-arm Spearman {close_result:+.4f} is below "
                  f"{args.min_close_spearman:+.4f}. This label set must not be used to "
                  f"choose between close variants.", file=sys.stderr)
            return 1
        print(f"\nPASS: close-arm Spearman {close_result:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
