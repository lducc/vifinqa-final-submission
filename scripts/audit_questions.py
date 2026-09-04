#!/usr/bin/env python3
"""Compare generated questions to the released ones on the axes we conditioned.

Promptagator validates its generator by plotting the first-word distribution of
generated queries against the real ones (Figure 3): the closer the two, the more
the synthetic task resembles the real one. This does the same, plus the surface
features sample_groups.py conditions on, so a drift shows up as a number rather
than as a hunch.

Nothing here decides anything. It reports, and the report goes in the paper.

    python scripts/audit_questions.py --questions output/rerank/questions.jsonl
"""

import argparse
from collections import Counter
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_questions import DEPENDENT, OPERATIONS, SHAPE, kind_of, tickers_of
from vifinqa.answers import fold
from vifinqa.jsonl import load_jsonl


# The organizers' four tiers, by their definitions on p.32:
#
#   Easy          one value from one table                            35.7%
#   Medium        two values and one arithmetic operation             23.2%
#   Intermediate  more than two values, repeated or grouped           19.8%
#   Hard          multi-hop dependent: an intermediate result picks
#                 the next table, company or period                   21.3%
#
# The proxy below reads wording rather than counting the cells an answer needs,
# so on the released questions it returns 0.298 / 0.333 / 0.249 / 0.120 — it
# under-reads Hard and over-reads Medium. That bias is why the audit compares
# the proxy against itself on the released set instead of against those shares:
# what is being measured is the drift between two sets of questions, and a
# constant bias applied to both cancels.


def tier_of(question: str, tickers: set) -> str:
    """Which tier a question looks like, by the organizers' own definitions.

    A proxy, and a coarse one: it reads the wording rather than counting the
    cells an answer would actually need. Read it for drift between two sets of
    questions, never as a label.
    """
    if DEPENDENT.search(question):
        return "Hard"
    if SHAPE["aggregate"].search(question):
        return "Intermediate"
    kind, span = kind_of(question, tickers)
    values = span if kind != "single" else 1
    operation = any(pattern.search(question) for _, pattern in OPERATIONS)
    if values > 2:
        return "Intermediate"
    if values == 2 or operation:
        return "Medium"
    return "Easy"


def unit(folded: str) -> str:
    for pattern, name in (
        (r"ty dong", "tỷ đồng"), (r"trieu dong", "triệu đồng"),
        (r"phan tram|%", "phần trăm"), (r"ty le", "tỷ lệ"),
        (r"co phieu", "cổ phiếu"), (r"\bdong\b", "đồng"),
    ):
        if re.search(pattern, folded):
            return name
    return "(none)"


def scope(folded: str) -> str:
    if re.search(r"cong ty me|bao cao rieng|rieng le", folded):
        return "công ty mẹ"
    if re.search(r"hop nhat", folded):
        return "hợp nhất"
    return "(unstated)"


FEATURES = {
    "unit": unit,
    "scope": scope,
    "period": lambda f: "end" if re.search(r"cuoi nam|31 thang 12|cuoi ky|tai ngay", f) else "during",
    "first word": lambda f: f.split()[0] if f.split() else "",
    "ends with ?": lambda f: str(f.strip().endswith("?")),
}

# Read on the unfolded question, because the shape and kind detectors are the
# ones in measure_questions.py and they match diacritics. Sharing them is the
# point: the sampler conditions on what that script measured, so the audit has
# to ask the same question of the generated text.
RAW_FEATURES = {
    "shape": lambda q, tickers: "+".join(
        sorted(name for name, pattern in SHAPE.items() if pattern.search(q))) or "plain",
    "kind": lambda q, tickers: kind_of(q, tickers)[0],
    "span": lambda q, tickers: str(kind_of(q, tickers)[1]),
    "tier": lambda q, tickers: tier_of(q, tickers),
}


def profile(questions: list[str], tickers: set[str]) -> dict:
    folded = [fold(q) for q in questions]
    total = max(1, len(questions))
    report = {name: {k: v / total for k, v in Counter(extract(f) for f in folded).most_common(8)}
              for name, extract in FEATURES.items()}
    report.update({name: {k: v / total for k, v
                          in Counter(extract(q, tickers) for q in questions).most_common(8)}
                   for name, extract in RAW_FEATURES.items()})
    words = [len(q.split()) for q in questions]
    report["length"] = {
        "median chars": statistics.median(len(q) for q in questions),
        "median words": statistics.median(words),
        "p90 words": sorted(words)[int(0.9 * (len(words) - 1))],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "output/rerank/questions.jsonl")
    parser.add_argument("--released", type=Path,
                        default=ROOT / "data/raw/vifinqa/questions/questions.jsonl")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/vifinqa")
    args = parser.parse_args()

    generated = [record["question"] for record in load_jsonl(args.questions)]
    released = [record["question"] for record in load_jsonl(args.released)]
    tickers = tickers_of(args.data_root)
    ours, theirs = profile(generated, tickers), profile(released, tickers)

    print(f"generated {len(generated)}   released {len(released)}\n")
    for name in list(FEATURES) + list(RAW_FEATURES) + ["length"]:
        print(name)
        keys = list(dict.fromkeys(list(theirs[name]) + list(ours[name])))
        for key in keys[:8]:
            a, b = ours[name].get(key, 0), theirs[name].get(key, 0)
            if name == "length":
                print(f"  {str(key):18s} generated {a:>7}   released {b:>7}")
            else:
                print(f"  {str(key):18s} generated {a:6.3f}   released {b:6.3f}   Δ {a-b:+.3f}")
        print()

    # One number for the whole comparison: total variation distance over the
    # features that are distributions, so a regression is visible at a glance.
    drift = {}
    for name in list(FEATURES) + list(RAW_FEATURES):
        keys = set(ours[name]) | set(theirs[name])
        drift[name] = 0.5 * sum(abs(ours[name].get(k, 0) - theirs[name].get(k, 0)) for k in keys)
    print("total variation distance (0 identical, 1 disjoint):")
    for name, value in sorted(drift.items(), key=lambda pair: -pair[1]):
        print(f"  {name:18s} {value:.3f}")


if __name__ == "__main__":
    main()
