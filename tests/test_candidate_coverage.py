"""Candidate coverage keeps sparse and dense arms separate."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import json

import pytest

from evaluate_candidate_coverage import arm_tables, coverage, load_pairs
from select_candidate_arm import select_candidates
from vifinqa.fusion import fuse_orders
from vifinqa.retrieval import table_budget


RECORD = {"candidates": [
    {"table_id": "A", "sparse_rank": 1},
    {"table_id": "B", "sparse_rank": 2, "dense_rank": 1},
    {"table_id": "C", "sparse_rank": None, "dense_rank": 2},
]}


def test_each_arm_uses_only_its_own_candidates():
    assert arm_tables(RECORD, "sparse") == ["A", "B"]
    assert arm_tables(RECORD, "dense") == ["B", "C"]
    # B appears in both rankings and receives RRF's agreement bonus.
    assert arm_tables(RECORD, "hybrid") == ["B", "A", "C"]


def test_coverage_reports_strict_all_gold_separately_from_any_gold():
    report = coverage({1: {"A", "C"}}, {1: RECORD}, "sparse")

    assert report["mean_gold_recall"] == 0.5
    assert report["mean_gold_recall_at_5"] == 0.5
    assert report["mrr5"] == 1.0
    assert report["questions_with_any_gold"] == 1.0
    assert report["questions_with_all_gold"] == 0.0


def test_sparse_scoring_file_can_be_derived_without_changing_the_record():
    record = {"id": 1, "question": "q", "selected_docs": ["R"], **RECORD}
    selected = select_candidates(record, "sparse")

    assert [candidate["table_id"] for candidate in selected["candidates"]] == ["A", "B"]
    assert selected["question"] == "q"
    assert len(record["candidates"]) == 3


def test_default_table_budget_is_the_live_validated_two_per_report():
    assert table_budget(1) == 2
    assert table_budget(4) == 8
    assert table_budget(20) == 30


def test_equal_rrf_rewards_tables_that_independent_rankings_agree_on():
    assert fuse_orders([["A", "B"], ["B", "C"]]) == ["B", "A", "C"]


def test_pair_loading_rejects_incomplete_exports(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({"id": 1, **RECORD}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing 1 expected questions"):
        load_pairs(path, {1, 2})


def test_pair_loading_rejects_duplicate_questions(tmp_path):
    path = tmp_path / "pairs.jsonl"
    record = json.dumps({"id": 1, **RECORD})
    path.write_text(f"{record}\n{record}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate question id 1"):
        load_pairs(path, {1})
