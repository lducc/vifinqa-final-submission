"""Small guardrail for optional LLM operation plans."""

from __future__ import annotations

import re
from typing import Any


OPERATIONS = {
    "direct_lookup",
    "compute_ratio",
    "filter_then_average",
    "filter_then_argmax_then_lookup",
    "filter_then_count",
}


def validate_and_repair_plan(question: str, plan: Any) -> dict[str, Any] | None:
    """Return a typed, conservative plan or ``None``.

    The repair rules handle only structural errors observed in the smoke test;
    values and evidence are never invented.
    """
    if not isinstance(plan, dict) or plan.get("operation") not in OPERATIONS:
        return None
    selection = plan.get("selection")
    if not isinstance(selection, dict):
        return None
    direction = selection.get("direction")
    group_by = selection.get("group_by")
    if direction not in {"max", "min", "none"} or group_by not in {"entity", "year", "none"}:
        return None

    repaired = dict(plan)
    repaired["filters"] = [str(item) for item in plan.get("filters", []) if str(item).strip()]
    repaired["target_metric"] = str(plan.get("target_metric", ""))
    repaired["formula"] = str(plan.get("formula", ""))
    repaired["evidence_entities"] = [str(item) for item in plan.get("evidence_entities", [])]

    folded = question.casefold()
    asks_count = bool(re.search(r"\b(bao nhiêu|có bao nhiêu)\s+(doanh nghiệp|công ty|ngân hàng)", folded))
    if asks_count:
        repaired["operation"] = "filter_then_count"

    # Multiple equality filters on years are an AND trap; a question listing
    # years means membership in that set.
    years = re.findall(r"year\s*==\s*(\d{4})", " ".join(repaired["filters"]))
    if len(years) > 1:
        filters = [item for item in repaired["filters"] if not re.search(r"year\s*==\s*\d{4}", item)]
        filters.append(f"year in [{', '.join(years)}]")
        repaired["filters"] = filters
    return repaired
