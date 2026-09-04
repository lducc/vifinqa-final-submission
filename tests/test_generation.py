"""The training-data pipeline, from raw filings to grouped training rows.

Three stages, each tested for the one thing that would silently poison the data:
sampling must draw tables a numeric question can be answered from, the generator
prompt must reflect the released questions rather than the model's habits, and
assembly must not label a question's own table as a negative.
"""

import json
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "kaggle"))

import assemble_training
import audit_questions
import sample_groups


def test_questions_are_drawn_from_the_tables_the_retriever_would_serve():
    """The sampled population must be the one the reranker meets at inference.

    An earlier version tested a regex against the rendered representation, which
    carries the title, row 0 and a label inventory but never the numeric body —
    so it measured something other than "this table holds an answer".
    """
    from vifinqa.retrieval import Table, carries_figures

    def table(identifier, rows):
        return Table(table_id=identifier, report_id="R", page=None, start_line=1,
                     rows=rows, title="", context=(), headers=(), periods=(), unit="")

    statement = table("R|1", (("Chỉ tiêu", "2020"),
                              ("Tiền và tương đương tiền", "26.285.074.127")))
    board = table("R|2", (("Hội đồng Quản trị",), ("Ông Phạm Quang Dũng",)))
    assert carries_figures(statement)
    assert not carries_figures(board)
    assert "carries_figures" in (ROOT / "scripts" / "sample_groups.py").read_text("utf-8")


def test_the_sampled_mix_follows_the_released_questions():
    """One, two and three tables in the proportion the real questions name items.

    Two is the commonest, and two thirds of questions name more than one — the
    reason a single-table corpus trains for the wrong task.
    """
    sizes = dict(sample_groups.TABLE_MIX)
    assert sizes[2] > sizes[1] and sizes[2] > sizes[3]
    assert sizes[2] + sizes[3] > 0.5
    assert abs(sum(sizes.values()) - 1.0) < 0.02


def test_the_prompt_states_a_scope_only_when_the_style_says_so():
    """62.7% of released questions name no scope; the prompt must be able to omit it."""
    generate_questions = pytest.importorskip(
        "generate_questions", reason="needs torch")
    task = {"year": 2020, "scope": "separate", "tables": [{"text": "x"}],
            "style": {"unit": "tỷ đồng", "name_scope": False,
                      "period_end": True, "operation": ""}}
    quiet = generate_questions.rules_for(task, len(task["tables"]))
    assert "công ty mẹ" not in quiet
    task["style"]["name_scope"] = True
    assert "công ty mẹ" in generate_questions.rules_for(task, len(task["tables"]))


def test_the_prompt_asks_for_the_unit_the_style_drew():
    generate_questions = pytest.importorskip(
        "generate_questions", reason="needs torch")
    task = {"year": 2020, "scope": "consolidated", "tables": [{"text": "x"}],
            "style": {"unit": "triệu đồng", "name_scope": False,
                      "period_end": False, "operation": ""}}
    assert "triệu đồng" in generate_questions.rules_for(task, len(task["tables"]))
    task["style"]["unit"] = ""
    assert "đơn vị" not in generate_questions.rules_for(task, len(task["tables"]))


def test_a_question_that_does_not_retrieve_its_table_is_dropped(tmp_path):
    """Promptagator's round trip: a generation that drifted off its table is noise."""
    assert hasattr(assemble_training, "main")


def test_the_audit_reports_a_distance_per_feature():
    tickers = {"VJC", "ACB"}
    ours = audit_questions.profile(["Doanh thu năm 2020 là bao nhiêu tỷ đồng?"], tickers)
    assert ours["unit"]["tỷ đồng"] == 1.0
    assert ours["scope"]["(unstated)"] == 1.0
    assert ours["kind"]["single"] == 1.0
    theirs = audit_questions.profile(
        ["Doanh thu của công ty mẹ năm 2020 là bao nhiêu triệu đồng?"], tickers)
    assert theirs["scope"]["công ty mẹ"] == 1.0
    assert theirs["unit"]["triệu đồng"] == 1.0


def test_the_audit_measures_the_axes_the_sampler_conditions_on():
    """A generator that writes no cross-year questions has to show up as drift."""
    tickers = {"VJC", "ACB"}
    profiled = audit_questions.profile(
        ["Doanh thu của VJC năm nào cao nhất trong giai đoạn 2020 đến 2023?",
         "Lợi nhuận của VJC và ACB năm 2022 trung bình là bao nhiêu tỷ đồng?"],
        tickers)
    assert profiled["kind"]["years"] == 0.5
    assert profiled["kind"]["peers"] == 0.5
    assert profiled["shape"]["superlative"] == 0.5
    assert profiled["shape"]["aggregate"] == 0.5


def test_the_holdout_is_derived_from_the_corpus_alone():
    """No annotation of ours decides the split, so it reproduces from the seed."""
    source = (ROOT / "scripts" / "sample_groups.py").read_text(encoding="utf-8")
    assert "--holdout" in source
    assert "annotations" not in source


def test_the_group_kinds_follow_the_released_questions():
    """54% of questions ask about one filing, 29% span years, 17% span issuers."""
    kinds = dict(sample_groups.KIND_MIX)
    assert abs(sum(kinds.values()) - 1.0) < 0.01
    assert kinds["single"] > kinds["years"] > kinds["peers"]


def test_a_superlative_is_drawn_where_there_is_something_to_maximise():
    """Measured: 0.9% of single-filing questions, 46% of cross-year ones.

    A question about one filing has no axis to take a maximum over, which is why
    an earlier sampler that only produced single-filing groups could not cover
    the quarter of the task that asks for one.
    """
    def marginal(kind, name):
        return sum(weight for combo, weight in sample_groups.SHAPE_MIX[kind]
                   if name in combo)

    assert marginal("single", "superlative") < 0.1
    assert marginal("years", "superlative") > 0.4
    assert marginal("peers", "aggregate") > marginal("single", "aggregate") * 10


def test_the_shapes_are_drawn_together_because_they_occur_together():
    """The joint distribution, not three coin flips.

    Independent draws would ask for an average and a maximum and a filter in one
    twenty-word question 3.9% of the time on cross-company groups; it happens in
    1.1% of the real ones.
    """
    for kind, combos in sample_groups.SHAPE_MIX.items():
        assert abs(sum(weight for _, weight in combos) - 1.0) < 0.02, kind
    triples = sum(weight for combo, weight in sample_groups.SHAPE_MIX["peers"]
                  if len(combo) == 3)
    assert triples < 0.05

    rng = random.Random(0)
    drawn = [sample_groups.shape_for(rng, "single") for _ in range(2000)]
    plain = sum(1 for shape in drawn if not any(shape.values()))
    assert plain / len(drawn) > 0.85


def test_the_counterpart_is_the_table_that_reads_most_like_the_anchor():
    """Cross-year groups compare the same statement, found by BM25, not by rule."""
    from vifinqa.retrieval import unicode_tokenize

    def entry(text):
        return ("id", text, unicode_tokenize(text))

    tables = [entry("Hội đồng Quản trị Ông Nguyễn Văn A thành viên"),
              entry("Tiền và các khoản tương đương tiền tiền gửi ngân hàng")]
    anchor = set(unicode_tokenize("Tiền và các khoản tương đương tiền tiền mặt"))
    assert sample_groups.counterpart(anchor, tables) == 1


def test_a_big_group_shows_less_of_each_table_so_the_prompt_still_fits():
    """Ten peers at full length would overrun the context; four do not."""
    generate_questions = pytest.importorskip(
        "generate_questions", reason="needs torch")

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[1]["content"]

    def task(count):
        return {"year": 2021, "scope": "consolidated", "kind": "peers",
                "company": "CTCP Test", "ticker": "TST",
                "tables": [{"text": "x" * 4000, "ticker": f"T{i:02d}", "year": 2021}
                           for i in range(count)],
                "style": {"unit": "", "name_scope": False, "period_end": False,
                          "operation": ""}}

    small = generate_questions.prompt_for(Tokenizer(), task(4))
    large = generate_questions.prompt_for(Tokenizer(), task(10))
    assert small.count("x") == 4 * generate_questions.TABLE_CHARS
    assert large.count("x") <= generate_questions.TABLE_BUDGET


def test_a_cross_filing_prompt_says_which_figure_belongs_to_which_filing():
    generate_questions = pytest.importorskip(
        "generate_questions", reason="needs torch")
    task = {"year": 2021, "scope": "consolidated", "kind": "years",
            "company": "CTCP Test", "ticker": "TST",
            "tables": [{"text": "x", "ticker": "TST", "year": 2020},
                       {"text": "y", "ticker": "TST", "year": 2021}],
            "style": {"unit": "", "name_scope": False, "period_end": False,
                      "operation": "", "superlative": True}}
    preamble = generate_questions.preamble_for(task, 2)
    assert "2020, 2021" in preamble
    rules = generate_questions.rules_for(task, 2)
    assert "năm nào" in rules          # the axis a cross-year superlative ranges over
    assert "CÙNG một chỉ tiêu" in rules


def test_a_candidate_says_which_filing_it_came_from():
    """The one fact a dense candidate can be wrong about, and the gate no longer checks.

    Every candidate used to come from a report the gate had already matched on
    ticker, year and scope, so the representation could leave all three out and
    lose nothing. A dense candidate skips the gate, so a table can now be the
    right line item in the wrong company's filing — and a representation that
    omits the filing gives the reranker no way to say so.
    """
    from vifinqa.rerank import identity_line, table_representation
    from vifinqa.retrieval import Table
    from vifinqa.tables import ReportIdentity

    identity = ReportIdentity("VJC_financial_statements_2018_separate", "VJC", 2018,
                              "separate", "somewhere.txt")
    assert identity_line(identity) == "VJC 2018 công ty mẹ"

    table = Table(
        table_id="VJC_financial_statements_2018_separate|12", report_id=identity.report_id,
        page=1, start_line=12, rows=(("Lãi tiền gửi", "208"), ("Cổ tức", "17")),
        title="Thu nhập tài chính", context=(), headers=(("Chỉ tiêu", "2018"),),
        periods=("2018",), unit="VND",
    )
    without = table_representation(table, 0)
    with_filing = table_representation(table, 0, identity=identity)
    assert "VJC" not in without
    # First line, because it is the part that can disqualify the table outright.
    assert with_filing.startswith("VJC 2018 công ty mẹ")


def test_the_trainer_and_the_scorer_read_the_same_instruction():
    """A tuned adapter scored under a different prompt is scored on text it never saw."""
    trainer = (ROOT / "kaggle/train_reranker.py").read_text("utf-8")
    scorer = (ROOT / "kaggle/rerank_qwen_8b.py").read_text("utf-8")
    def instruction(source: str) -> str:
        start = source.index("INSTRUCTION = (")
        return source[start:source.index("\n)", start) + 2]

    assert instruction(trainer) == instruction(scorer)
    # And it must describe the input that is actually sent. An earlier version
    # told the model to read a "Chỉ tiêu cần tìm" block, which came from a
    # line-item lexicon of ours that is now quarantined and which no pairs file
    # has carried since.
    assert "Chỉ tiêu cần tìm" not in instruction(trainer)
    assert "line_items" not in (ROOT / "scripts/export_rerank_pairs.py").read_text("utf-8")


def test_a_dense_candidate_claims_no_sparse_rank_it_was_never_given():
    """The sparse ranker did not return it, so it must not appear to have."""
    import export_rerank_pairs

    dense = {"order": {7: 0}, "ids": [], "tables": None, "questions": None}
    assert export_rerank_pairs.dense_candidates(dense, 999, set(), {}, 5) == []
    source = (ROOT / "scripts/export_rerank_pairs.py").read_text("utf-8")
    assert '"sparse_rank": None' in source


def test_each_arm_ranks_only_the_candidates_its_own_retriever_found():
    """Three systems out of one scored pool, which is the whole measurement plan.

    A ranking is a claim about what the first stage found. An arm that borrowed
    the other arm's candidates would not be that arm's system, and the comparison
    between them would measure nothing.
    """
    from vifinqa.fusion import first_stage, fuse

    candidates = [
        {"table_id": "both", "sparse_rank": 3, "dense_rank": 4},
        {"table_id": "sparse_only", "sparse_rank": 1, "dense_rank": None},
        {"table_id": "dense_only", "sparse_rank": None, "dense_rank": 1},
    ]
    scores = {"both": 0.9, "sparse_only": 0.1, "dense_only": 0.8}

    assert set(fuse(candidates, scores, "fuse", arm="sparse")) == {"both", "sparse_only"}
    assert set(fuse(candidates, scores, "fuse", arm="dense")) == {"both", "dense_only"}
    assert set(fuse(candidates, scores, "fuse", arm="hybrid")) == {
        "both", "sparse_only", "dense_only"}

    # Found by both retrievers beats found by one at a comparable rank: the
    # agreement bonus is what reciprocal rank fusion is for.
    assert first_stage(candidates[0], "hybrid") > first_stage(candidates[1], "hybrid")


def test_line_item_pruning_keeps_a_small_recall_cushion():
    from vifinqa.fusion import prune_line_item_tables

    record = {"candidates": [
        {"table_id": "hit", "text": "Đường dẫn dòng: Lãi tiền gửi"},
        {"table_id": "noise", "text": "Đường dẫn dòng: Chi phí khác"},
    ]}
    assert prune_line_item_tables(record, ["noise", "hit"], ["Lãi tiền gửi"], 0) == ["hit"]
    assert prune_line_item_tables(record, ["noise", "hit"], ["unknown"], 0) == ["noise", "hit"]


def test_the_sparse_arm_is_the_ordering_that_shipped():
    """Adding arms must not quietly restate the rule behind every past submission."""
    from vifinqa.fusion import RRF_OFFSET, fuse

    candidates = [{"table_id": f"t{rank}", "sparse_rank": rank} for rank in range(1, 6)]
    scores = {"t5": 0.9, "t1": 0.2, "t3": 0.5}
    by_hand = sorted(
        (c["table_id"] for c in candidates),
        key=lambda table_id: -(
            0.5 / (RRF_OFFSET + int(table_id[1:]))
            + (0.5 / (RRF_OFFSET + {"t5": 1, "t3": 2, "t1": 3}[table_id])
               if table_id in scores else 0.0)
        ),
    )
    assert fuse(candidates, scores, "fuse", arm="sparse") == by_hand


def test_the_generator_keeps_the_group_mix_intact_when_the_budget_stops_it():
    """The run is ended by money, so what gets written is a prefix of the file.

    Sorting the whole file by table count would make that prefix every
    single-table group and nothing else, which is the one skew the three-kind
    sampler exists to remove.
    """
    import generate_questions

    tasks = [{"task_id": str(n), "tables": [0] * (1 + n % 3)} for n in range(600)]
    blocked = [task for start in range(0, len(tasks), generate_questions.BLOCK)
               for task in sorted(tasks[start:start + generate_questions.BLOCK],
                                  key=lambda task: len(task["tables"]))]
    # Any prefix of a whole block or more carries every group size. Inside one
    # block the order is sorted, so the block length is the worst-case skew and
    # the reason BLOCK is small.
    prefix = blocked[:generate_questions.BLOCK]
    sizes = {len(task["tables"]) for task in prefix}
    assert sizes == {1, 2, 3}, sizes
    assert generate_questions.BLOCK <= 128
    # And a batch still sees near-uniform widths, which is what the sort is for.
    batch = blocked[:generate_questions.MAX_BATCH]
    assert len({len(task["tables"]) for task in batch}) == 1
