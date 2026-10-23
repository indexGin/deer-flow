import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills/public/data-analysis/scripts/analyze.py"
)


@pytest.fixture
def cached_cli(tmp_path):
    analysis = runpy.run_path(str(SCRIPT))
    source = tmp_path / "data.csv"
    source.write_text("value\n7\n", encoding="utf-8")
    key = analysis["compute_files_hash"]([str(source)])
    cache = tmp_path / ".data-analysis-cache"
    cache.mkdir()
    with duckdb.connect(str(cache / f"{key}.duckdb")) as con:
        table_map = {}
        analysis["_load_csv"](con, str(source), table_map)
    (cache / f"{key}.table_map.json").write_text(
        json.dumps(table_map), encoding="utf-8"
    )
    return [sys.executable, str(SCRIPT), "--files", str(source), "--action", "query"]


@pytest.mark.parametrize(
    "sql",
    [
        "SELEC value FROM data",
        "SELECT missing FROM data",
        "SELECT CAST('bad' AS INTEGER) FROM data",
        "DELETE FROM data",
    ],
)
def test_failed_query_exits_nonzero_without_overwriting_export(
    cached_cli, tmp_path, sql
):
    output = tmp_path / "report.json"
    output.write_text("existing report", encoding="utf-8")
    env = dict(os.environ, TMP=str(tmp_path), TEMP=str(tmp_path), TMPDIR=str(tmp_path))

    result = subprocess.run(
        [*cached_cli, "--sql", sql, "--output-file", str(output)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "SQL Error:" in result.stderr
    assert '"data" (data): value' in result.stderr
    assert "Results exported" not in result.stdout
    assert output.read_text(encoding="utf-8") == "existing report"


@pytest.mark.parametrize(
    "predicate, expected", [("true", [{"value": 7}]), ("false", [])]
)
def test_successful_query_keeps_zero_exit_and_exports_rows(
    cached_cli, tmp_path, predicate, expected
):
    output = tmp_path / "report.json"
    env = dict(os.environ, TMP=str(tmp_path), TEMP=str(tmp_path), TMPDIR=str(tmp_path))
    result = subprocess.run(
        [
            *cached_cli,
            "--sql",
            f"SELECT value FROM data WHERE {predicate}",
            "--output-file",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Results exported" in result.stdout
    assert json.loads(output.read_text(encoding="utf-8")) == expected
