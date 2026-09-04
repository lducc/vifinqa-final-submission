#!/usr/bin/env python3
"""Measure the released questions, and print the constants sample_groups.py uses.

Every proportion the sampler conditions on is measured here, over the 1,012
public questions and the public corpus, with nothing but the raw data as input.
No annotation of ours is read, and no released question is ever copied into the
training data — only its surface features are counted.

The point is reproducibility. A constant pasted into a sampler is an assertion;
a constant printed by a script an organizer can run is a measurement. Run this,
and the block it prints is what belongs at the top of sample_groups.py.

    python scripts/measure_questions.py            # question features only
    python scripts/measure_questions.py --items    # also count named line items

The line-item count needs the corpus, because the vocabulary of statutory item
names is the corpus's own row labels: a phrase that appears as the first cell of
a row in many different issuers' filings is a statutory label, and one that
appears in a single filing is that company's own wording. That threshold is the
only judgement in this file and it is stated as a flag.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
import json
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.retrieval import load_reports, report_tables

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
UPPER = re.compile(r"\b[A-Z]{3}\b")

# What the question does with the figures. Three shapes, because the organizers'
# own difficulty tiers turn on exactly these: their Intermediate tier is
# repeated or grouped arithmetic (mean, margin), and their Hard tier is
# "multi-hop dependent", an intermediate result deciding what to read next —
# which in question wording is a filter applied before the lookup.
SHAPE = {
    "superlative": re.compile(r"cao nhất|thấp nhất|lớn nhất|nhỏ nhất|nhiều nhất"
                              r"|ít nhất|dẫn đầu", re.I),
    "conditional": re.compile(r"nếu|trong số|những công ty|công ty nào|thỏa"
                              r"|vượt|lớn hơn|nhỏ hơn|trên mức|dưới mức|chỉ tính", re.I),
    "aggregate": re.compile(r"trung bình|trung vị|bình quân", re.I),
}

# Mutually exclusive, first match wins, because a question asks its answer in
# one unit. Ordered so the longer phrase is tested before the shorter one it
# contains.
UNITS = (("tỷ đồng", re.compile(r"tỷ đồng", re.I)),
         ("phần trăm", re.compile(r"phần trăm|%", re.I)),
         ("triệu đồng", re.compile(r"triệu đồng", re.I)),
         ("đồng", re.compile(r"\bđồng\b", re.I)),
         ("cổ phiếu", re.compile(r"cổ phiếu", re.I)),
         ("tỷ lệ", re.compile(r"tỷ lệ", re.I)))

# The organizers' Hard tier is "multi-hop dependent: an intermediate result
# decides the next table, company or reporting period" (their p.32, 21.3% of the
# task). In Vietnamese that is a relative clause: one figure picks the year or
# the company, and a second figure is then asked of whatever was picked. Every
# released question this matches also carries a superlative, so it is not a new
# shape — it is what half of the superlatives actually are.
DEPENDENT = re.compile(
    r"(tại|ở|vào|trong)\s+(năm|quý|thời điểm|kỳ)\s+(có|đạt|ghi nhận)"
    r"|(công ty|doanh nghiệp|đơn vị)\s+(có|đạt|ghi nhận)[^,\.]{0,60}(cao nhất|thấp nhất|lớn nhất|nhỏ nhất)"
    r"|(năm|công ty|doanh nghiệp)\s+(nào)[^,\.]{0,60}(thì|,)\s", re.I)

SEPARATE = re.compile(r"công ty mẹ|riêng lẻ|báo cáo riêng", re.I)
CONSOLIDATED = re.compile(r"hợp nhất", re.I)
PERIOD_END = re.compile(r"cuối năm|cuối kỳ|31[/. ]?12|tại ngày|thời điểm", re.I)

OPERATIONS = (("tổng", re.compile(r"\btổng\b", re.I)),
              ("chênh lệch", re.compile(r"chênh lệch|tăng trưởng|thay đổi|biến động", re.I)),
              ("tỷ trọng", re.compile(r"tỷ trọng|chiếm bao nhiêu", re.I)))


def fold(text: str) -> str:
    stripped = unicodedata.normalize("NFD", text.lower())
    return " ".join("".join(c for c in stripped if not unicodedata.combining(c)).split())


def tickers_of(data_root: Path) -> set[str]:
    codes = set()
    with open(data_root / "code_stock.csv", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row and re.fullmatch(r"[A-Z]{3}", row[0].strip()):
                codes.add(row[0].strip())
    return codes


def kind_of(question: str, tickers: set[str]) -> tuple[str, int]:
    """What the question ranges over, and how far.

    A question naming two tickers compares issuers; one naming two years and one
    issuer compares periods; anything else is a single filing. Nothing else is
    needed, and nothing else would be checkable by someone holding only the
    public file.
    """
    named = set(UPPER.findall(question)) & tickers
    years = set(YEAR.findall(question))
    if len(named) > 1:
        return "peers", len(named)
    if len(years) > 1:
        return "years", len(years)
    return "single", 1


def statutory_labels(data_root: Path, minimum: int) -> list[str]:
    """Row labels that recur across issuers, longest first."""
    seen = Counter()
    reports = load_reports(data_root)
    for number, report in enumerate(reports, 1):
        labels = set()
        try:
            tables = report_tables(str(report.path), report.identity)
        except Exception:
            continue
        for table in tables:
            for row in table.rows:
                if not row:
                    continue
                label = fold(row[0])
                if len(label) >= 10 and not any(c.isdigit() for c in label):
                    labels.add(label)
        seen.update(labels)
        if number % 200 == 0:
            print(f"  scanned {number}/{len(reports)} reports", file=sys.stderr, flush=True)
    return sorted((label for label, count in seen.items() if count >= minimum),
                  key=lambda label: (-len(label), label))


def count_items(question: str, labels: list[str]) -> int:
    """How many distinct statutory labels the question names.

    Longest first, and each match is cut out before the next is tried, so
    "chi phí lãi vay" is not also counted as "chi phí".
    """
    text = f" {fold(question)} "
    found = 0
    for label in labels:
        if label in text:
            text = text.replace(label, " ")
            found += 1
    return found


def share(counter: Counter, total: int) -> dict:
    return {key: round(count / max(1, total), 3) for key, count in counter.most_common()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/vifinqa")
    parser.add_argument("--questions", type=Path,
                        default=ROOT / "data/raw/vifinqa/questions/questions.jsonl")
    parser.add_argument("--items", action="store_true",
                        help="also count named line items, which needs a corpus scan")
    parser.add_argument("--item-minimum", type=int, default=20,
                        help="a row label counts as statutory at this many reports")
    args = parser.parse_args()

    questions = [json.loads(line)["question"]
                 for line in open(args.questions, encoding="utf-8") if line.strip()]
    tickers = tickers_of(args.data_root)
    total = len(questions)

    kinds, spans = Counter(), {"years": Counter(), "peers": Counter()}
    superlative, dependent = Counter(), Counter()
    shapes = {kind: Counter() for kind in ("single", "years", "peers")}
    units, operations = Counter(), Counter()
    scope, period = Counter(), 0
    for question in questions:
        kind, span = kind_of(question, tickers)
        kinds[kind] += 1
        if kind in spans:
            spans[kind][span] += 1
        shapes[kind][tuple(sorted(name for name, pattern in SHAPE.items()
                                  if pattern.search(question)))] += 1
        if SHAPE["superlative"].search(question):
            superlative[kind] += 1
            dependent[kind] += bool(DEPENDENT.search(question))
        units[next((name for name, pattern in UNITS if pattern.search(question)), "")] += 1
        operations[next((name for name, pattern in OPERATIONS if pattern.search(question)), "")] += 1
        if SEPARATE.search(question):
            scope["separate"] += 1
        if CONSOLIDATED.search(question):
            scope["consolidated"] += 1
        period += bool(PERIOD_END.search(question))

    print(f"{total} released questions, {len(tickers)} tickers\n")
    print(f"KIND_MIX = {tuple((k, round(v / total, 3)) for k, v in kinds.most_common())}\n")
    for kind in ("years", "peers"):
        name = "YEAR_SPAN_MIX" if kind == "years" else "PEER_COUNT_MIX"
        n = kinds[kind]
        print(f"{name} = "
              f"{tuple((s, round(c / max(1, n), 3)) for s, c in sorted(spans[kind].items()))}\n")
    print("SHAPE_MIX = {")
    for kind in ("single", "years", "peers"):
        n = kinds[kind]
        combos = {("plain" if not combo else "+".join(combo)): round(count / max(1, n), 3)
                  for combo, count in shapes[kind].most_common()}
        print(f"    {kind!r}: {combos},   # n={n}")
    print("}\n")
    print("DEPENDENT_GIVEN_SUPERLATIVE = {")
    for kind in ("single", "years", "peers"):
        print(f"    {kind!r}: {dependent[kind] / max(1, superlative[kind]):.3f},"
              f"   # {dependent[kind]}/{superlative[kind]}")
    print("}\n")
    print(f"UNIT_MIX = {tuple((u, round(c / total, 3)) for u, c in units.most_common())}\n")
    print(f"OPERATION_MIX = {tuple((o, round(c / total, 3)) for o, c in operations.most_common())}\n")
    print(f"say separate: {scope['separate'] / total:.3f}   "
          f"say consolidated: {scope['consolidated'] / total:.3f}   "
          f"period end: {period / total:.3f}")

    if args.items:
        print("\nscanning the corpus for statutory row labels", file=sys.stderr, flush=True)
        labels = statutory_labels(args.data_root, args.item_minimum)
        print(f"\n{len(labels)} row labels in {args.item_minimum}+ reports")
        counts = Counter(min(3, count_items(q, labels)) for q in questions)
        named = sum(count for size, count in counts.items() if size >= 1)
        print(f"TABLE_MIX = "
              f"{tuple((s, round(c / max(1, named), 3)) for s, c in sorted(counts.items()) if s)}")
        print("named nothing:", counts[0], f"({counts[0] / total:.3f})")


if __name__ == "__main__":
    main()
