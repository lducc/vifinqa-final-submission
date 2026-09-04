"""The benchmark is itself a measuring instrument, so it needs its own checks.

Two things are worth pinning. The correlation maths must be right, because a
sign error here would silently reverse every promotion decision. And the gold
set the fidelity script reads must be the one the caller asked for, because the
whole failure this work addresses came from two label definitions being confused
for each other.
"""

import json

import pytest

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fidelity = load("evaluate_benchmark_fidelity")


def test_correlation_is_one_for_an_identical_ordering():
    values = [0.1, 0.2, 0.3, 0.4]
    assert fidelity.correlate(values, values) == pytest.approx(1.0)
    assert fidelity.correlate(values, values, spearman=True) == pytest.approx(1.0)


def test_correlation_is_minus_one_for_a_reversed_ordering():
    assert fidelity.correlate([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert fidelity.correlate([1, 2, 3, 4], [4, 3, 2, 1], spearman=True) == pytest.approx(-1.0)


def test_spearman_ignores_spacing_that_pearson_reacts_to():
    """A monotone but non-linear relation is a perfect rank match, not a perfect fit."""
    x, y = [1, 2, 3, 4], [1, 2, 4, 100]
    assert fidelity.correlate(x, y, spearman=True) == pytest.approx(1.0)
    assert fidelity.correlate(x, y) < 0.95


def test_ranks_average_over_ties():
    assert fidelity.ranks([5.0, 5.0, 9.0]) == [1.5, 1.5, 3.0]


def test_f2_weights_recall_above_precision():
    gold = {1: {"r|1", "r|2", "r|3", "r|4"}}
    recall_heavy = fidelity.score(gold, {1: ["r|1", "r|2", "r|3", "r|9"]})
    precision_heavy = fidelity.score(gold, {1: ["r|1"]})
    assert recall_heavy[0] > precision_heavy[0]


def test_f2_is_averaged_per_question_not_derived_from_averaged_precision():
    # One short exact answer and one long half-right one. Averaging the rates
    # first lets the short question's precision subsidise the long question's,
    # which the organizer's per-question average never does.
    gold = {1: {"r|1"}, 2: {"r|1", "r|2", "r|3", "r|4"}}
    predictions = {
        1: ["r|1"],
        2: ["r|1", "r|2", "r|8", "r|9", "r|10", "r|11", "r|12", "r|13"],
    }
    f2, precision, recall = fidelity.score(gold, predictions)
    assert f2 == pytest.approx((1.0 + 10 / 24) / 2)
    from_averages = 5 * precision * recall / (4 * precision + recall)
    assert f2 != pytest.approx(from_averages)


def test_missing_question_scores_as_a_miss_rather_than_being_skipped():
    gold = {1: {"r|1"}, 2: {"r|2"}}
    f2, precision, recall = fidelity.score(gold, {1: ["r|1"]})
    assert recall == pytest.approx(0.5)
    assert precision == pytest.approx(0.5)


def test_gold_field_is_used_when_present(tmp_path):
    path = tmp_path / "bench.jsonl"
    path.write_text(json.dumps({
        "id": 1,
        "annotation": {
            "gold_tables": ["r|1", "r|2"],
            "gold_tables_v2": ["r|1", "r|2", "r|3"],
            "row_column_bindings": [{"table": "r|1"}],
        },
    }) + "\n", encoding="utf-8")
    assert fidelity.gold_sets(path, "gold_tables_v2") == {1: {"r|1", "r|2", "r|3"}}


def test_gold_field_falls_back_to_bindings_not_to_the_wide_set(tmp_path):
    """The wide set overshoots; an absent v2 field must not silently select it."""
    path = tmp_path / "bench.jsonl"
    path.write_text(json.dumps({
        "id": 1,
        "annotation": {
            "gold_tables": ["r|1", "r|2"],
            "row_column_bindings": [{"table": "r|1"}],
        },
    }) + "\n", encoding="utf-8")
    assert fidelity.gold_sets(path, "gold_tables_v2") == {1: {"r|1"}}


def test_gold_falls_back_to_the_wide_set_only_when_there_are_no_bindings(tmp_path):
    path = tmp_path / "bench.jsonl"
    path.write_text(json.dumps({
        "id": 1,
        "annotation": {"gold_tables": ["r|1", "r|2"], "row_column_bindings": []},
    }) + "\n", encoding="utf-8")
    assert fidelity.gold_sets(path, "gold_tables_v2") == {1: {"r|1", "r|2"}}
