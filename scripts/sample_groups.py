#!/usr/bin/env python3
"""Sample tables to write questions about. No rules, no thresholds, no parsing.

This is the whole of our contribution to the labels: pick some tables. A model
then writes a question those tables answer, and because we chose them, the
positive set is known without anyone deciding what counts as evidence. That is
the standard query-generation recipe for retrieval — Doc2Query, InPars,
Promptagator — applied to Vietnamese filings.

Earlier versions of this file decided which rows were line items, which figures
were significant, and which tables restated which. Every one of those rules was
ours, and each one measured wrong when it was finally checked. There is nothing
here to be wrong about: a table is drawn, and it is a positive because it is the
table the question was written from.

Groups come in three kinds, because the released questions do. Most ask about
one filing; a quarter compare several years of one issuer; a sixth compare
several issuers in one year. The proportions and the shapes that go with them
are measured below, not chosen.

    python scripts/sample_groups.py
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from docs import load_companies
from vifinqa.jsonl import write_jsonl
from vifinqa.rerank import INVENTORY, table_representation
from vifinqa.retrieval import (
    carries_figures, load_reports, report_tables, score_bm25, unicode_tokenize,
)

# Everything below is printed by `python scripts/measure_questions.py`, which
# counts the 1,012 released questions and the corpus and nothing else. Paste, do
# not invent: a constant an organizer can reprint is a measurement, one typed in
# here is an assertion.
#
# A generator left to its own devices writes the question it finds most natural,
# and that is not the question this task asks: it names a unit almost always,
# names the scope only a third of the time, and says "hợp nhất" in 1.4% of
# cases. Sampling the surface features from what was observed is how
# the generated distribution ends up looking like the real one without any
# released question being copied.

# What a question ranges over. A question naming two or more years needs tables
# from two or more filings, and one naming a group of companies needs tables
# from several issuers; neither can be written from a single report, which is
# why an earlier version of this file produced only the first kind and reported
# the other two as a coverage gap. Every issuer in the corpus files six to
# eleven years, so the gap was in the sampler rather than in the data.
KIND_MIX = (("single", 0.539), ("years", 0.288), ("peers", 0.174))

# Line items named per question, within a single filing, counted as the corpus's
# own row labels — a phrase appearing as the first cell of a row in twenty or
# more filings is statutory wording rather than one company's. The count is
# capped at three and errs slightly high, because a few statutory labels are
# scope phrases ("của công ty mẹ") that a question uses for a different purpose.
TABLE_MIX = ((1, 0.329), (2, 0.387), (3, 0.284))

# Distinct years named, among the questions that name more than one.
YEAR_SPAN_MIX = ((2, 0.416), (3, 0.172), (4, 0.227), (5, 0.172), (6, 0.014))

# Tickers named, among the questions that compare a group of companies. The
# tail is real: 10.2% of them name seven or more.
PEER_COUNT_MIX = ((2, 0.091), (3, 0.335), (4, 0.335), (5, 0.080), (6, 0.057),
                  (7, 0.051), (8, 0.028), (9, 0.006), (10, 0.017))

# What the question does with the figures, as a joint distribution rather than
# three independent rates, because the shapes are not independent: half the
# cross-company questions take an average, and only 1% of them take an average
# and a maximum and a filter at once. Drawing the three separately would put a
# three-clause instruction in front of the generator eleven times as often as it
# occurs, and no 20-55 word question can honour all three.
SHAPE_MIX = {
    "single": (((), 0.930), (("conditional", "superlative"), 0.022),
               (("conditional",), 0.020), (("aggregate",), 0.017),
               (("superlative",), 0.009), (("aggregate", "superlative"), 0.002)),
    "years": ((("superlative",), 0.460), ((), 0.323), (("aggregate",), 0.103),
              (("conditional", "superlative"), 0.052), (("conditional",), 0.027),
              (("aggregate", "superlative"), 0.021),
              (("aggregate", "conditional", "superlative"), 0.007),
              (("aggregate", "conditional"), 0.007)),
    "peers": ((("aggregate",), 0.358), ((), 0.182), (("superlative",), 0.170),
              (("aggregate", "superlative"), 0.142),
              (("conditional", "superlative"), 0.057), (("conditional",), 0.051),
              (("aggregate", "conditional"), 0.028),
              (("aggregate", "conditional", "superlative"), 0.011)),
}

# The unit the answer is asked in, mutually exclusive, "" meaning none stated.
UNIT_MIX = (("tỷ đồng", 0.386), ("phần trăm", 0.246), ("triệu đồng", 0.213),
            ("", 0.083), ("cổ phiếu", 0.028), ("đồng", 0.026), ("tỷ lệ", 0.018))

# 36.0% of questions say "công ty mẹ" and 48.4% of reports are separate, so about
# three in four separate-scope questions say it. 1.4% say "hợp nhất" against a
# 48.5% consolidated share, so consolidated questions almost never name a scope.
SAY_SEPARATE = 0.360 / 0.484
SAY_CONSOLIDATED = 0.014 / 0.485

# 43.9% ask as at a date ("cuối năm", "31/12", "tại ngày"); the rest ask over
# the year.
PERIOD_END = 0.439

# Arithmetic across the tables of one filing: sum 37.9%, difference or growth
# 12.2%, share-of 1.1%, and the rest name no operation at all.
OPERATION_MIX = (("tổng", 0.379), ("chênh lệch", 0.122), ("tỷ trọng", 0.011),
                 ("", 0.488))


def draw(rng, mix):
    values, weights = zip(*mix)
    return rng.choices(values, weights=weights)[0]


def counterpart(anchor_tokens, tables):
    """The table in another filing that the anchor most resembles.

    A question comparing 2021 with 2022 has to compare the same thing in both,
    so the second table cannot be drawn at random. Nothing here knows what a
    balance sheet is: it is the same BM25 the retriever runs, asking which table
    of the other filing reads most like this one. Where the counterpart does not
    exist the top match is merely the least bad, and the round-trip filter in
    assemble_training.py drops the question that results.
    """
    scores = score_bm25(anchor_tokens, [tokens for _, _, tokens in tables])
    return max(range(len(tables)), key=lambda i: scores[i])


# Half of the released questions that ask for a maximum do not stop there: the
# maximum picks a year or a company, and a different figure is then asked of
# whatever it picked. That is the organizers' Hard tier, 21.3% of the task and
# the one where accuracy collapses. Every released question matching it also
# carries a superlative, so this splits an existing draw rather than adding a
# proportion. Measured by measure_questions.py.
DEPENDENT_GIVEN_SUPERLATIVE = {"single": 0.667, "years": 0.401, "peers": 0.687}


def shape_for(rng, kind):
    """Which shapes this question takes, drawn as one combination."""
    combos, weights = zip(*SHAPE_MIX[kind])
    drawn = rng.choices(combos, weights=weights)[0]
    shapes = {name: name in drawn
              for name in ("superlative", "conditional", "aggregate")}
    shapes["dependent"] = (shapes["superlative"]
                           and rng.random() < DEPENDENT_GIVEN_SUPERLATIVE[kind])
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/vifinqa")
    parser.add_argument("--holdout", type=float, default=0.2,
                        help="share of reports withheld from training, for evaluation")
    parser.add_argument("--side", choices=("train", "holdout"), default="train",
                        help="which side of the split to sample. The held-out side is "
                             "what makes the holdout worth its cost: a question written "
                             "from known tables carries its own gold, so questions "
                             "sampled here are an evaluation set on reports the reranker "
                             "never saw — the only instrument this task otherwise has")
    parser.add_argument("--per-report", type=int, default=8)
    parser.add_argument("--reports", type=int)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path, default=ROOT / "output/rerank/tasks.jsonl")
    args = parser.parse_args()

    companies = {ticker: company.name for ticker, company
                 in load_companies(args.data_root / "code_stock.csv").items()}

    rng = random.Random(args.seed)
    # Hold reports out by name rather than by label, so the split needs nothing
    # but the corpus and reproduces from the seed alone. Splitting on the report
    # keeps every table of a filing on one side, which splitting on the table
    # would not.
    reports = sorted(load_reports(args.data_root), key=lambda r: r.identity.report_id)
    rng.shuffle(reports)
    held = reports[:round(args.holdout * len(reports))]
    if args.side == "train":
        write_jsonl(args.output.with_name("held_out_reports.jsonl"),
                    [{"report_id": r.identity.report_id} for r in held])
    reports = held if args.side == "holdout" else reports[len(held):]
    if args.reports:
        reports = reports[:args.reports]

    # Parsed once and kept, because a cross-filing group needs several reports
    # open at the same time and the corpus does not fit that pattern otherwise.
    # `carries_figures` is the gate the retriever puts every live candidate
    # through, so using it here makes the tables we write questions about the
    # same population the reranker meets at inference. Testing the rendered text
    # instead would test the wrong thing: the representation carries the title,
    # row 0 and a label inventory, never the numeric body.
    corpus, identity = {}, {}
    for number, report in enumerate(sorted(reports, key=lambda r: r.identity.report_id), 1):
        try:
            tables = report_tables(str(report.path), report.identity)
        except Exception as error:
            print(f"skip {report.identity.report_id}: {error}", flush=True)
            continue
        usable = []
        for table in tables:
            if not carries_figures(table):
                continue
            text = table_representation(table, 0, inventory=INVENTORY)
            usable.append((f"{report.identity.report_id}|{table.start_line}",
                           text, unicode_tokenize(text)))
        if usable:
            corpus[report.identity.report_id] = usable
            identity[report.identity.report_id] = report.identity
        if number % 200 == 0:
            print(f"parsed {number}/{len(reports)} reports", flush=True)
    print(f"{len(corpus)} reports parsed, "
          f"{sum(len(v) for v in corpus.values())} tables usable", flush=True)

    # Which other filings a group can reach: the same issuer in another year at
    # the same scope, and other issuers in the same year at the same scope.
    siblings, peers = defaultdict(list), defaultdict(list)
    for report_id, ident in identity.items():
        siblings[(ident.ticker, ident.scope)].append(report_id)
        peers[(ident.year, ident.scope)].append(report_id)

    def described(report_id, table_id, text):
        ident = identity[report_id]
        return {"table_id": table_id, "text": text, "ticker": ident.ticker,
                "company": companies.get(ident.ticker, ident.ticker),
                "year": ident.year}

    tasks = []
    order = sorted(corpus)
    rng.shuffle(order)
    for number, report_id in enumerate(order, 1):
        ident = identity[report_id]
        usable = list(corpus[report_id])
        rng.shuffle(usable)
        cursor = 0
        for _ in range(args.per_report):
            kind = draw(rng, KIND_MIX)
            # One filing per distinct year, or per distinct issuer, so a group
            # cannot compare a year with itself. Twenty filings in the corpus
            # arrive split across two files under the same ticker, year and
            # scope, and without this both halves can land in one group.
            axis = ((lambda i: i.year) if kind == "years"
                    else (lambda i: i.ticker) if kind == "peers" else None)
            reachable = (siblings[(ident.ticker, ident.scope)] if kind == "years"
                         else peers[(ident.year, ident.scope)] if kind == "peers" else [])
            distinct = {}
            for other in reachable:
                if axis and axis(identity[other]) != axis(ident):
                    distinct.setdefault(axis(identity[other]), other)
            others = sorted(distinct.values())
            if kind != "single" and not others:
                kind = "single"

            if kind == "single":
                wanted = draw(rng, TABLE_MIX)
                if cursor + wanted > len(usable):
                    break
                picked = [described(report_id, table_id, text)
                          for table_id, text, _ in usable[cursor:cursor + wanted]]
                cursor += wanted
            else:
                if cursor >= len(usable):
                    break
                anchor_id, anchor_text, anchor_tokens = usable[cursor]
                cursor += 1
                span = draw(rng, YEAR_SPAN_MIX if kind == "years" else PEER_COUNT_MIX)
                chosen = rng.sample(others, min(span - 1, len(others)))
                picked = [described(report_id, anchor_id, anchor_text)]
                for other in chosen:
                    index = counterpart(set(anchor_tokens), corpus[other])
                    table_id, text, _ = corpus[other][index]
                    picked.append(described(other, table_id, text))

            says = SAY_SEPARATE if ident.scope == "separate" else SAY_CONSOLIDATED
            style = {
                "unit": draw(rng, UNIT_MIX),
                "name_scope": rng.random() < says,
                "period_end": rng.random() < PERIOD_END,
                "operation": (draw(rng, OPERATION_MIX)
                              if kind == "single" and len(picked) > 1 else ""),
            }
            style.update(shape_for(rng, kind))
            tasks.append({
                "style": style,
                "kind": kind,
                "task_id": f"{report_id}#{len(tasks)}",
                "report_id": report_id,
                "report_ids": sorted({t["table_id"].split("|")[0] for t in picked}),
                "company": companies.get(ident.ticker, ident.ticker),
                "ticker": ident.ticker,
                "year": ident.year,
                "scope": ident.scope,
                "tables": picked,
            })
        if number % 200 == 0:
            print(f"{number}/{len(order)} reports, {len(tasks)} groups", flush=True)

    write_jsonl(args.output, tasks)
    print(f"{len(tasks)} groups from {len(corpus)} reports -> {args.output}")
    total = max(1, len(tasks))
    for name, key in (("kind", lambda t: t["kind"]),
                      ("tables per group", lambda t: len(t["tables"])),
                      ("reports per group", lambda t: len(t["report_ids"]))):
        counts = Counter(key(task) for task in tasks)
        print(f"{name}:", {k: f"{v} ({v/total:.3f})" for k, v in sorted(counts.items(), key=str)})


if __name__ == "__main__":
    main()
