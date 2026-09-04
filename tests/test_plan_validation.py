from vifinqa.plan_validation import validate_and_repair_plan


def _plan(**changes):
    plan = {
        "operation": "filter_then_argmax_then_lookup",
        "selection": {"metric": "revenue", "direction": "max", "group_by": "year"},
        "filters": [], "target_metric": "ratio", "formula": "", "evidence_entities": [],
    }
    plan.update(changes)
    return plan


def test_count_question_repairs_argmax_operation():
    result = validate_and_repair_plan("Có bao nhiêu doanh nghiệp đạt điều kiện?", _plan())
    assert result["operation"] == "filter_then_count"


def test_year_equalities_become_membership_filter():
    result = validate_and_repair_plan("Giá trị lớn nhất qua nhiều năm", _plan(filters=["year == 2016", "year == 2020"]))
    assert result["filters"] == ["year in [2016, 2020]"]


def test_invalid_plan_is_rejected():
    assert validate_and_repair_plan("question", {"operation": "run_code"}) is None
