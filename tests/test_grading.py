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


def _stage_openml(root, legacy_answers: bool = False, metric: str = "accuracy",
                  kind: str = "multiclass", classes=("a", "b")):
    """Fake a prepared openml task in the split public/private layout
    (answers in a separate private tree — see the 2026-07-07 firewall split).
    Classification submissions carry one probability column per class."""
    prep = root / "public" / TASK / "prepared"
    prep.mkdir(parents=True)
    priv = root / "private" / TASK
    priv.mkdir(parents=True)
    classes = None if kind == "regression" else sorted(classes)
    meta = {"task_id": TASK, "dataset_id": 0, "metric": metric,
            "higher_better": metric in ("accuracy", "roc_auc"), "kind": kind,
            "target": "label", "id_col": "row_id", "classes": classes,
            "n_train": 3, "n_test": 3}
    (prep / "meta.json").write_text(json.dumps(meta))
    (prep / "description.md").write_text("# toy task\n")
    labels = ["a", "b", "a"] if kind != "regression" else [1.0, 2.0, 3.0]
    pd.DataFrame({"row_id": [0, 1, 2], "x": [1, 2, 3],
                  "label": labels}).to_csv(prep / "train.csv", index=False)
    pd.DataFrame({"row_id": [3, 4, 5], "x": [4, 5, 6]}).to_csv(prep / "test.csv", index=False)
    sample = pd.DataFrame({"row_id": [3, 4, 5]})
    if kind == "regression":
        sample["prediction"] = 0.0
    else:
        for c in classes:
            sample[c] = 1.0 / len(classes)
    sample.to_csv(prep / "sample_submission.csv", index=False)
    ans_labels = ["a", "b", "b"] if kind != "regression" else [4.0, 5.0, 6.0]
    answers = pd.DataFrame({"row_id": [3, 4, 5], "label": ans_labels})
    answers.to_csv(priv / "answers.csv", index=False)
    if legacy_answers:  # the pre-firewall layout: answers inside the agent dir
        answers.to_csv(prep / "answers.csv", index=False)
    return prep, priv


def _bench(tmp_path):
    return OpenMLTabular(data_dir=str(tmp_path / "public"),
                         private_data_dir=str(tmp_path / "private"))


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
    _stage_openml(tmp_path)   # accuracy over classes a/b; answers = a, b, b
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.csv"
    pd.DataFrame({"row_id": [3, 4, 5],
                  "a": [0.9, 0.1, 0.2], "b": [0.1, 0.9, 0.8]}).to_csv(sub, index=False)
    score = bench.grade(task, sub)
    assert score.valid and score.value == 1.0
    pd.DataFrame({"row_id": [3, 4, 5],
                  "a": [0.9, 0.9, 0.9], "b": [0.1, 0.1, 0.1]}).to_csv(sub, index=False)
    score = bench.grade(task, sub)
    assert score.valid and abs(score.value - (1 / 3)) < 1e-9


def test_openml_auc_orientation_is_structural(tmp_path):
    """Regression (smoke-e2e MagicTelescope): AUC is scored from the positive
    class's OWN named column, so no orientation convention exists to guess."""
    _stage_openml(tmp_path, metric="roc_auc", kind="binary")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.csv"
    # a perfect model, expressed only via named per-class columns
    pd.DataFrame({"row_id": [3, 4, 5],
                  "a": [0.9, 0.2, 0.1], "b": [0.1, 0.8, 0.9]}).to_csv(sub, index=False)
    score = bench.grade(task, sub)
    assert score.valid and score.value == 1.0
    # the same model with columns swapped = genuinely inverted predictions
    pd.DataFrame({"row_id": [3, 4, 5],
                  "b": [0.9, 0.2, 0.1], "a": [0.1, 0.8, 0.9]}).to_csv(sub, index=False)
    score = bench.grade(task, sub)
    assert score.valid and score.value == 0.0


def test_openml_multiclass_log_loss(tmp_path):
    """Single-prediction-column log_loss was structurally ungradeable for >2
    classes; per-class columns make it exact (and rows may be unnormalised)."""
    import math
    _stage_openml(tmp_path, metric="log_loss", kind="multiclass",
                  classes=("a", "b", "c"))
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.csv"
    pd.DataFrame({"row_id": [3, 4, 5],           # answers: a, b, b
                  "a": [0.8, 0.1, 0.1], "b": [0.1, 0.8, 0.8],
                  "c": [0.1, 0.1, 0.1]}).to_csv(sub, index=False)
    score = bench.grade(task, sub)
    assert score.valid and abs(score.value - (-math.log(0.8))) < 1e-9
    # unnormalised rows are tolerated (renormalised), not rejected
    pd.DataFrame({"row_id": [3, 4, 5],
                  "a": [8.0, 1.0, 1.0], "b": [1.0, 8.0, 8.0],
                  "c": [1.0, 1.0, 1.0]}).to_csv(sub, index=False)
    score = bench.grade(task, sub)
    assert score.valid and abs(score.value - (-math.log(0.8))) < 1e-9


def test_openml_regression_contract_unchanged(tmp_path):
    _stage_openml(tmp_path, metric="rmse", kind="regression")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.csv"
    pd.DataFrame({"row_id": [3, 4, 5],
                  "prediction": [4.0, 5.0, 6.0]}).to_csv(sub, index=False)
    score = bench.grade(task, sub)
    assert score.valid and score.value == 0.0


def test_openml_rejects_wrong_columns(tmp_path):
    """The old single-column classification format is now an invalid Score
    (never an exception), naming the expected columns."""
    _stage_openml(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.csv"
    pd.DataFrame({"row_id": [3, 4, 5], "prediction": ["a", "b", "b"]}).to_csv(sub, index=False)
    score = bench.grade(task, sub)
    assert not score.valid and "missing" in score.details["reason"]


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


def test_openml_rejects_negative_probabilities(tmp_path):
    """A row like [-0.5, 1.5] sums to 1 but is not a distribution — reject,
    don't renormalise (cubic review, PR #8)."""
    _stage_openml(tmp_path, metric="roc_auc", kind="binary")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.csv"
    pd.DataFrame({"row_id": [3, 4, 5],
                  "a": [-0.5, 0.5, 0.5], "b": [1.5, 0.5, 0.5]}).to_csv(sub, index=False)
    score = bench.grade(task, sub)
    assert not score.valid and "non-negative" in score.details["reason"]


def test_openml_validate_submission_agrees_with_grade(tmp_path):
    """validate_submission is the public-only mirror of grade()'s validity
    gates: same verdict on every failure mode, no answers file needed."""
    _stage_openml(tmp_path, metric="roc_auc", kind="binary")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.csv"

    cases = {
        "valid": pd.DataFrame({"row_id": [3, 4, 5],
                               "a": [0.9, 0.2, 0.1], "b": [0.1, 0.8, 0.9]}),
        "wrong_columns": pd.DataFrame({"row_id": [3, 4, 5],
                                       "prediction": [0.1, 0.8, 0.9]}),
        "missing_rows": pd.DataFrame({"row_id": [3, 4],
                                      "a": [0.9, 0.2], "b": [0.1, 0.8]}),
        "negative_probs": pd.DataFrame({"row_id": [3, 4, 5],
                                        "a": [-0.5, 0.5, 0.5], "b": [1.5, 0.5, 0.5]}),
    }
    for name, frame in cases.items():
        frame.to_csv(sub, index=False)
        ok, reason = bench.validate_submission(task, sub)
        score = bench.grade(task, sub)
        assert ok == score.valid, f"{name}: validate={ok} but grade.valid={score.valid}"
        if not ok:
            assert reason, name

    # and it truly never opens the answers: delete them, valid file still validates
    (tmp_path / "private" / TASK / "answers.csv").unlink()
    cases["valid"].to_csv(sub, index=False)
    assert bench.validate_submission(task, sub) == (True, None)


def test_validate_submission_default_strips_score(tmp_path):
    """The ABC default (grade-and-strip) returns ONLY (ok, reason) — the tuple
    shape is the firewall."""
    _stage_openml(tmp_path, metric="rmse", kind="regression")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.csv"
    pd.DataFrame({"row_id": [3, 4, 5], "prediction": [4.0, 5.0, 6.0]}).to_csv(sub, index=False)
    from arbench.core.benchmark import Benchmark
    result = Benchmark.validate_submission(bench, task, sub)   # force the default
    assert result == (True, None)
    sub.write_text("row_id,prediction\n")
    ok, reason = Benchmark.validate_submission(bench, task, sub)
    assert not ok and isinstance(reason, str)
    # the default NEVER forwards grade()'s own reason/exception text — it can
    # embed private detail (e.g. the answers path in an IOError); cubic P2
    assert reason == Benchmark.GENERIC_INVALID
    assert "answers" not in reason and "/" not in reason
