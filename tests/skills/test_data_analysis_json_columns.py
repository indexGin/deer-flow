import csv
import json
import runpy
from pathlib import Path

import duckdb
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills/public/data-analysis/scripts/analyze.py"
)
action_query = runpy.run_path(str(SCRIPT))["action_query"]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 AS value, 2 AS value",
        "SELECT a.*, b.* FROM (VALUES (1)) AS a(value) CROSS JOIN (VALUES (2)) AS b(value)",
    ],
)
def test_json_export_rejects_duplicate_columns_without_overwriting(
    tmp_path, existing, sql
):
    output = tmp_path / "result.json"
    if existing:
        output.write_bytes(b"existing report")

    with (
        duckdb.connect() as connection,
        pytest.raises(ValueError, match="unique column names.*SQL aliases"),
    ):
        action_query(connection, sql, {}, str(output))

    if existing:
        assert output.read_bytes() == b"existing report"
    else:
        assert not output.exists()


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        (
            "SELECT 1 AS left_value, 2 AS right_value",
            [{"left_value": 1, "right_value": 2}],
        ),
        ("SELECT 1 AS value, 2 AS Value", [{"value": 1, "Value": 2}]),
        ("SELECT 1 AS value WHERE false", []),
    ],
)
def test_json_export_preserves_unique_names(tmp_path, sql, expected):
    output = tmp_path / "result.json"
    with duckdb.connect() as connection:
        action_query(connection, sql, {}, str(output))
    assert json.loads(output.read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize("extension", ["csv", "md"])
def test_tabular_exports_preserve_duplicate_columns(tmp_path, extension):
    output = tmp_path / f"result.{extension}"
    with duckdb.connect() as connection:
        action_query(connection, "SELECT 1 AS value, 2 AS value", {}, str(output))
    if extension == "csv":
        with output.open(encoding="utf-8", newline="") as handle:
            assert list(csv.reader(handle)) == [["value", "value"], ["1", "2"]]
    else:
        assert (
            output.read_text(encoding="utf-8")
            == "| value | value |\n| --- | --- |\n| 1 | 2 |\n"
        )
