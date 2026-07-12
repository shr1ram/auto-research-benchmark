"""Library-surface + CLI tests: get_benchmark wiring and the two commands
(`arbench tasks`, `arbench grade`) that survived the 2026-07 prune."""
from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from click.testing import CliRunner

import arbench
from arbench.cli import main

TASK = "wine_quality"


def _stage_openml(root):
    """A tiny fake prepared openml task in the split public/private layout."""
    prep = root / "public" / TASK / "prepared"
    prep.mkdir(parents=True)
    priv = root / "private" / TASK
    priv.mkdir(parents=True)
    # SPEC-CONSISTENT with tasks.py (wine_quality: regression, rmse,
    # target "quality") — the CLI constructs the bench with enforce_spec on,
    # which is exactly what these tests should exercise
    meta = {"task_id": TASK, "dataset_id": 0, "metric": "rmse",
            "higher_better": False, "kind": "regression", "target": "quality",
            "id_col": "row_id", "classes": None, "n_train": 3, "n_test": 3}
    (prep / "meta.json").write_text(json.dumps(meta))
    (prep / "description.md").write_text("# toy task\n")
    pd.DataFrame({"row_id": [0, 1, 2], "x": [1, 2, 3],
                  "quality": [5.0, 6.0, 5.0]}).to_csv(prep / "train.csv", index=False)
    pd.DataFrame({"row_id": [3, 4, 5], "x": [4, 5, 6]}).to_csv(prep / "test.csv", index=False)
    pd.DataFrame({"row_id": [3, 4, 5],
                  "prediction": [0.0, 0.0, 0.0]}).to_csv(prep / "sample_submission.csv", index=False)
    pd.DataFrame({"row_id": [3, 4, 5],
                  "quality": [4.0, 5.0, 6.0]}).to_csv(priv / "answers.csv", index=False)


def test_get_benchmark_unknown_name_raises():
    with pytest.raises(KeyError):
        arbench.get_benchmark("nope")


def test_get_benchmark_constructs_openml(tmp_path):
    bench = arbench.get_benchmark("openml_tabular", data_dir=str(tmp_path))
    assert bench.name == "openml_tabular"
    assert TASK in list(bench.list_tasks())


def test_cli_tasks_lists_ids():
    result = CliRunner().invoke(main, ["tasks", "--benchmark", "openml_tabular"])
    assert result.exit_code == 0
    assert TASK in result.output.splitlines()


def test_cli_grade_roundtrip(tmp_path):
    _stage_openml(tmp_path)
    sub = tmp_path / "submission.csv"
    pd.DataFrame({"row_id": [3, 4, 5],
                  "prediction": [4.0, 5.0, 6.0]}).to_csv(sub, index=False)
    result = CliRunner().invoke(main, [
        "grade", "--benchmark", "openml_tabular", "--task", TASK,
        "--data-dir", str(tmp_path / "public"),
        "--private-data-dir", str(tmp_path / "private"),
        str(sub),
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["valid"] is True
    assert payload["value"] == 0.0          # exact predictions -> rmse 0


def test_cli_grade_invalid_submission_exits_2(tmp_path):
    _stage_openml(tmp_path)
    sub = tmp_path / "submission.csv"
    sub.write_text("row_id,prediction\n")  # empty predictions -> invalid
    result = CliRunner().invoke(main, [
        "grade", "--benchmark", "openml_tabular", "--task", TASK,
        "--data-dir", str(tmp_path / "public"),
        "--private-data-dir", str(tmp_path / "private"),
        str(sub),
    ])
    assert result.exit_code == 2, result.output
