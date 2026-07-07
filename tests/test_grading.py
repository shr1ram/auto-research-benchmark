"""Grading firewall tests.

The held-out answers must resolve OUTSIDE the agent-visible data dir (openml:
<task>/private/answers.csv, a sibling of prepared/; mlebench: a separate private
data root), the Task handed to adapters must never name the answers, and grading
must still work from the relocated paths.
"""
from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from arbench.benchmarks.openml_tabular.benchmark import OpenMLTabular
from arbench.benchmarks.mlebench_lite.benchmark import MLEBenchLite

# Any real task id from tasks.BY_ID (load_task rejects unknown ids); the data
# itself is a tiny fake staged into tmp_path.
TASK = "wine_quality"


def _stage_openml(root, legacy_answers: bool = False):
    """Fake a prepared openml task in the split public/private layout
    (answers in a separate private tree — see the 2026-07-07 firewall split)."""
    prep = root / "public" / TASK / "prepared"
    prep.mkdir(parents=True)
    priv = root / "private" / TASK
    priv.mkdir(parents=True)
    meta = {"task_id": TASK, "dataset_id": 0, "metric": "accuracy",
            "higher_better": True, "kind": "multiclass", "target": "label",
            "id_col": "row_id", "n_train": 3, "n_test": 3}
    (prep / "meta.json").write_text(json.dumps(meta))
    (prep / "description.md").write_text("# toy task\n")
    pd.DataFrame({"row_id": [0, 1, 2], "x": [1, 2, 3],
                  "label": ["a", "b", "a"]}).to_csv(prep / "train.csv", index=False)
    pd.DataFrame({"row_id": [3, 4, 5], "x": [4, 5, 6]}).to_csv(prep / "test.csv", index=False)
    pd.DataFrame({"row_id": [3, 4, 5],
                  "prediction": ["a", "a", "a"]}).to_csv(prep / "sample_submission.csv", index=False)
    answers = pd.DataFrame({"row_id": [3, 4, 5], "label": ["a", "b", "b"]})
    answers.to_csv(priv / "answers.csv", index=False)
    if legacy_answers:  # the pre-firewall layout: answers inside the agent dir
        answers.to_csv(prep / "answers.csv", index=False)
    return prep, priv


def test_openml_answers_outside_agent_dir(tmp_path):
    prep, priv = _stage_openml(tmp_path)
    bench = OpenMLTabular(data_dir=str(tmp_path / "public"),
                          private_data_dir=str(tmp_path / "private"))
    task = bench.load_task(TASK)
    # the agent-visible tree contains no answers file anywhere
    assert task.data_dir == prep
    assert not list(task.data_dir.rglob("answers.csv"))
    # and the Task never names the answers path (metadata included)
    joined = " ".join(str(v) for v in task.metadata.values()) + task.goal + task.eval
    assert "answers.csv" not in joined
    assert str(priv) not in joined


def test_openml_grades_from_private_answers(tmp_path):
    _stage_openml(tmp_path)
    bench = OpenMLTabular(data_dir=str(tmp_path / "public"),
                          private_data_dir=str(tmp_path / "private"))
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.csv"
    pd.DataFrame({"row_id": [3, 4, 5], "prediction": ["a", "b", "b"]}).to_csv(sub, index=False)
    score = bench.grade(task, sub)
    assert score.valid and score.value == 1.0
    pd.DataFrame({"row_id": [3, 4, 5], "prediction": ["a", "a", "a"]}).to_csv(sub, index=False)
    score = bench.grade(task, sub)
    assert score.valid and abs(score.value - (1 / 3)) < 1e-9


def test_openml_refuses_legacy_leaky_layout(tmp_path):
    _stage_openml(tmp_path, legacy_answers=True)
    bench = OpenMLTabular(data_dir=str(tmp_path / "public"),
                          private_data_dir=str(tmp_path / "private"))
    with pytest.raises(RuntimeError, match="FIREWALL"):
        bench.load_task(TASK)


def test_mlebench_private_root_resolution(tmp_path, monkeypatch):
    """Private-root resolution: env var > `<data_dir>-private` sibling > legacy
    single root. (No mlebench install needed — resolution is pure path logic.)"""
    monkeypatch.delenv("MLEBENCH_PRIVATE_DATA_DIR", raising=False)
    pub = tmp_path / "mlebench-data"
    pub.mkdir()

    legacy = MLEBenchLite(data_dir=str(pub))       # no sibling, no env: legacy
    assert legacy.private_data_dir == legacy.data_dir
    assert not legacy._firewalled()

    sibling = tmp_path / "mlebench-data-private"   # sibling appears: autodetect
    sibling.mkdir()
    fw = MLEBenchLite(data_dir=str(pub))
    assert fw.private_data_dir == sibling
    assert fw._firewalled()

    other = tmp_path / "elsewhere"                 # env var wins over sibling
    other.mkdir()
    monkeypatch.setenv("MLEBENCH_PRIVATE_DATA_DIR", str(other))
    env_fw = MLEBenchLite(data_dir=str(pub))
    assert env_fw.private_data_dir == other
    assert env_fw._firewalled()
