#!/usr/bin/env python3
"""Join generated questions to their tables, then keep only the ones that hold up.

The positives need no decision: they are the tables the question was written
from. What does need checking is whether the model wrote a question about them
at all — a generation that drifted off the table produces a query whose real
answer lives somewhere else, and training on it teaches the reranker to prefer
the wrong table.

The check is the round-trip used by Promptagator: retrieve with the generated
question and require the source tables to come back. It costs nothing, it needs
no annotation, and it uses the retriever the system already has.

Negatives are the highest-ranked tables the question was not written from, drawn
from the same filings the group spans. For a cross-year group that means the
same statement in the wrong year, which is the discrimination the reranker
actually has to make and the one a single-filing corpus never teaches.

    python scripts/assemble_training.py --questions output/rerank/questions.jsonl
"""

import argparse
from collections import Counter
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.jsonl import load_jsonl, write_jsonl
from vifinqa.rerank import INVENTORY, table_representation
from vifinqa.retrieval import (
    carries_figures, load_reports, report_tables, score_bm25, unicode_tokenize,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/vifinqa")
    parser.add_argument("--tasks", type=Path, default=ROOT / "output/rerank/tasks.jsonl")
    parser.add_argument("--questions", type=Path, default=ROOT / "output/rerank/questions.jsonl")
    parser.add_argument("--negatives", type=int, default=8)
    parser.add_argument("--foreign-negatives", type=int, default=2,
                        help="of those, how many come from a filing the group does not span")
    parser.add_argument("--foreign-refresh", type=int, default=200,
                        help="tasks between redraws of the foreign filings, so the "
                             "distractor companies are not the same four all run")
    parser.add_argument("--negative-pool", type=int, default=50,
                        help="per filing the group spans, sample negatives from this many")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--round-trip", type=int, default=10,
                        help="a source table must rank this high for its question to be kept")
    parser.add_argument("--exclude", type=Path,
                        default=ROOT / "output/rerank/held_out_reports.jsonl",
                        help="report_id list written by sample_groups.py")
    parser.add_argument("--output", type=Path, default=ROOT / "output/rerank/training_qwen.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    excluded = ({record["report_id"] for record in load_jsonl(args.exclude)}
                if args.exclude and args.exclude.exists() else set())

    asked = {record["task_id"]: record["question"] for record in load_jsonl(args.questions)}
    tasks = [task for task in load_jsonl(args.tasks)
             if task["task_id"] in asked
             and not (excluded & set(task.get("report_ids", [task["report_id"]])))]
    # Ordered by the filings a group spans, so consecutive tasks reuse the cache.
    tasks.sort(key=lambda task: task.get("report_ids", [task["report_id"]]))

    paths = {r.identity.report_id: r for r in load_reports(args.data_root)}
    cache, failed = {}, set()

    def filing(report_id):
        """The gated tables of one filing, rendered and tokenized once.

        The same gate the retriever applies, so a negative is a table the
        reranker could actually be handed rather than a signature block the
        retriever would have dropped before it ever ran. Tokenized here rather
        than per question because a filing answers several.
        """
        if report_id in cache:
            return cache[report_id]
        report = paths.get(report_id)
        if report is None or report_id in failed:
            failed.add(report_id)
            return None
        try:
            tables = [t for t in report_tables(str(report.path), report.identity)
                      if carries_figures(t)]
        except Exception:
            failed.add(report_id)
            return None
        # Rendered through row 0 here only to rank and to round-trip filter, which
        # needs one text per table and no question. What is written as a training
        # document is rendered again below, through the row the question matches.
        cache[report_id] = [
            (f"{report_id}|{table.start_line}",
             unicode_tokenize(table_representation(table, 0, inventory=INVENTORY,
                                                   identity=report.identity)),
             table, report.identity)
            for table in tables
        ]
        if len(cache) > 24:
            cache.pop(next(iter(cache)))
        return cache[report_id]

    # Negatives drawn only from the filings a group spans teach one distinction:
    # which table inside the right report. That was the whole job while a gate
    # settled the company, the year and the scope before any ranking ran. A dense
    # first stage does not go through that gate, so the reranker now also has to
    # reject the right statement from the wrong issuer, and nothing in a
    # same-filing negative ever asks it to.
    #
    # The foreign filings are redrawn periodically rather than fixed, or a run
    # would teach the model four particular companies rather than the idea.
    corpus = sorted(set(paths) - excluded)
    foreign_cache: dict = {}

    def foreign(seed_number: int):
        if seed_number % max(1, args.foreign_refresh) == 0 or not foreign_cache:
            foreign_cache.clear()
            for report_id in rng.sample(corpus, min(4, len(corpus))):
                loaded = filing(report_id)
                if loaded:
                    foreign_cache[report_id] = loaded
        return [entry for entries in foreign_cache.values() for entry in entries]

    def document(entry, question_tokens):
        """A candidate as the reranker will actually be handed it at inference.

        export_rerank_pairs renders a candidate through the row the sparse ranker
        matched — usually the line the question asked about. Rendering training
        documents through row 0 instead, which is header text as often as not,
        would train the model on documents whose most discriminative field is
        blank and serve it documents where that field carries the answer. So the
        row is chosen the same way on both sides.
        """
        _, _, table, ident = entry
        if not table.rows:
            return table_representation(table, 0, inventory=INVENTORY, identity=ident)
        scored = score_bm25(question_tokens,
                            [unicode_tokenize(" ".join(row)) for row in table.rows])
        best = max(range(len(table.rows)), key=lambda index: scored[index])
        return table_representation(table, best, inventory=INVENTORY, identity=ident)

    rows, kept = [], 0
    reasons, sizes, kinds = Counter(), Counter(), Counter()
    for number, task in enumerate(tasks, 1):
        report_ids = task.get("report_ids", [task["report_id"]])
        loaded = [filing(report_id) for report_id in report_ids]
        if any(one is None for one in loaded):
            reasons["filing would not parse"] += 1
            continue
        pooled = [entry for one in loaded for entry in one]

        question = asked[task["task_id"]]
        if len(question) < 25:
            reasons["too short to be a question"] += 1
            continue
        position = {table_id: index for index, (table_id, _, _, _) in enumerate(pooled)}
        sources = [position[t["table_id"]] for t in task["tables"]
                   if t["table_id"] in position]
        if len(sources) != len(task["tables"]):
            reasons["source table not found in its filing"] += 1
            continue

        asked_tokens = set(unicode_tokenize(question))
        ranked = score_bm25(asked_tokens, [tokens for _, tokens, _, _ in pooled])
        order = sorted(range(len(pooled)), key=lambda i: -ranked[i])
        # Scale the depth with the group: a question written from three tables
        # cannot put all three first, and a fixed cut-off would drop multi-table
        # groups hardest — reintroducing the single-table skew this corpus
        # exists to remove.
        top = set(order[:args.round_trip * len(sources)])
        if any(source not in top for source in sources):
            reasons["question does not retrieve its own tables"] += 1
            continue
        # Promptagator samples its 31 negatives from the top 200 the retriever
        # returns, "which approximates the inference time distribution". Ours
        # reranks to depth 130 across the gated filings, so each filing
        # contributes roughly fifty candidates — sample from that many rather
        # than always taking the eight hardest. Taking the hardest also
        # concentrates exactly where a false negative is most likely: a table
        # that ranks second is the one most apt to hold the item too.
        chosen = set(sources)
        near = max(0, args.negatives - args.foreign_negatives)
        pool = [i for i in order if i not in chosen][:args.negative_pool * len(report_ids)]
        if len(pool) < near:
            reasons["group too small for negatives"] += 1
            continue
        hard = rng.sample(pool, near)
        elsewhere = []
        if args.foreign_negatives:
            outside = [entry for entry in foreign(number)
                       if not entry[0].startswith(tuple(f"{r}|" for r in report_ids))]
            if len(outside) >= args.foreign_negatives:
                elsewhere = rng.sample(outside, args.foreign_negatives)

        kept += 1
        sizes[len(sources)] += 1
        kinds[task.get("kind", "single")] += 1
        for index in sources:
            rows.append({"id": task["task_id"], "query": question, "label": 1,
                         "table_id": pooled[index][0],
                         "document": document(pooled[index], asked_tokens)})
        for index in hard:
            rows.append({"id": task["task_id"], "query": question, "label": 0,
                         "table_id": pooled[index][0],
                         "document": document(pooled[index], asked_tokens)})
        for entry in elsewhere:
            rows.append({"id": task["task_id"], "query": question, "label": 0,
                         "table_id": entry[0],
                         "document": document(entry, asked_tokens)})
        if number % 500 == 0:
            print(f"{number}/{len(tasks)} groups, {kept} kept", flush=True)

    write_jsonl(args.output, rows)
    total = kept + sum(reasons.values())
    print(f"{kept}/{total} groups kept ({kept/max(1,total):.3f}), {len(rows)} rows -> {args.output}")
    print("tables per kept group:",
          {k: f"{v} ({v/max(1,kept):.3f})" for k, v in sorted(sizes.items())})
    print("kind of kept group:",
          {k: f"{v} ({v/max(1,kept):.3f})" for k, v in sorted(kinds.items())})
    for reason, count in reasons.most_common():
        print(f"  dropped, {reason}: {count}")


if __name__ == "__main__":
    main()
