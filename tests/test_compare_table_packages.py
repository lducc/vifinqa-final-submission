from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_table_packages import compare


def row(identifier, tables, answer=1.0, query="result = 1"):
    return {
        "id": identifier,
        "relevant_docs": ["R"],
        "relevant_tables": tables,
        "answer": answer,
        "pandas_query": query,
    }


def test_comparison_separates_table_changes_from_answer_changes():
    report = compare(
        {1: row(1, ["R|1", "R|2"])},
        {1: row(1, ["R|2", "R|3"])},
    )

    assert report["same_docs"] == 1
    assert report["same_table_set"] == 0
    assert report["same_answers"] == 1
    assert report["same_pandas_queries"] == 1
    assert report["tables_added"] == report["tables_removed"] == 1
