"""Grading firewall tests.

The held-out answers must resolve OUTSIDE the agent-visible data dir (openml:
<task>/private/answers.csv, a sibling of prepared/), the Task handed to adapters
must never name the answers, and grading must still work from the relocated
paths.
"""
from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from arbench.benchmarks.openml_tabular.benchmark import OpenMLTabular

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
    # enforce_spec=False: fixtures stage SYNTHETIC metas for the real
    # wine_quality id across every kind; the spec guard is pinned by its
    # own test below
    return OpenMLTabular(data_dir=str(tmp_path / "public"),
                         private_data_dir=str(tmp_path / "private"),
                         enforce_spec=False)


def test_openml_answers_outside_agent_dir(tmp_path):
    prep, priv = _stage_openml(tmp_path)
    bench = OpenMLTabular(data_dir=str(tmp_path / "public"),
                          private_data_dir=str(tmp_path / "private"),
                          enforce_spec=False)
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
    # non-numeric predictions: validate must agree with grade (cubic, PR #9)
    pd.DataFrame({"row_id": [3, 4, 5],
                  "prediction": ["high", "low", "mid"]}).to_csv(sub, index=False)
    ok, reason = bench.validate_submission(task, sub)
    score = bench.grade(task, sub)
    assert not ok and "numeric" in reason
    assert not score.valid


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
                          private_data_dir=str(tmp_path / "private"),
                          enforce_spec=False)
    with pytest.raises(RuntimeError, match="FIREWALL"):
        bench.load_task(TASK)


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


def test_openml_infinite_predictions_rejected_by_validate_and_grade(tmp_path):
    """inf converted to float cleanly, so validate said ok while grade's
    sklearn call raised -> Score.invalid: a validate/grade disagreement
    (subagent review find). Both must now reject, in agreement."""
    _stage_openml(tmp_path, metric="rmse", kind="regression")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.csv"
    pd.DataFrame({"row_id": [3, 4, 5],
                  "prediction": [float("inf"), 5.0, 6.0]}).to_csv(sub, index=False)
    ok, reason = bench.validate_submission(task, sub)
    assert not ok and "finite" in reason
    assert not bench.grade(task, sub).valid


def test_openml_infinite_probabilities_rejected(tmp_path):
    _stage_openml(tmp_path, metric="roc_auc", kind="binary")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.csv"
    pd.DataFrame({"row_id": [3, 4, 5], "a": [float("inf"), 0.5, 0.5],
                  "b": [0.5, 0.5, 0.5]}).to_csv(sub, index=False)
    ok, reason = bench.validate_submission(task, sub)
    assert not ok and "finite" in reason


def test_openml_nan_metric_is_invalid_not_valid_nan(tmp_path):
    """roc_auc on single-class truth WARNS and returns nan (no exception), so
    it bypassed the except-guard into Score(valid=True, value=nan), silently
    poisoning downstream rankings (subagent review find)."""
    _stage_openml(tmp_path, metric="roc_auc", kind="binary")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    # overwrite answers with a single-class truth vector
    pd.DataFrame({"row_id": [3, 4, 5],
                  "label": ["a", "a", "a"]}).to_csv(
        tmp_path / "private" / TASK / "answers.csv", index=False)
    sub = tmp_path / "submission.csv"
    pd.DataFrame({"row_id": [3, 4, 5], "a": [0.9, 0.2, 0.4],
                  "b": [0.1, 0.8, 0.6]}).to_csv(sub, index=False)
    score = bench.grade(task, sub)
    assert not score.valid
    assert "non-finite" in score.details.get("reason", "")


def test_openml_duplicate_row_ids_rejected_by_validate_and_grade(tmp_path):
    """Conflicting predictions for one row = broken agent pipeline; silently
    keeping the first masked it (2026-07-12). Both sides must agree."""
    _stage_openml(tmp_path, metric="roc_auc", kind="binary")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.csv"
    pd.DataFrame({"row_id": [3, 3, 4, 5], "a": [0.9, 0.1, 0.5, 0.5],
                  "b": [0.1, 0.9, 0.5, 0.5]}).to_csv(sub, index=False)
    ok, reason = bench.validate_submission(task, sub)
    score = bench.grade(task, sub)
    assert not ok and "duplicate row_ids" in reason
    assert not score.valid
    assert "duplicate row_ids" in score.details["reason"]


def test_openml_multiclass_metric_is_log_loss():
    """Metric assignment is convention-derived (AMLB): binary->AUC,
    multiclass->log_loss (LOWER better), regression->RMSE — never authored."""
    from arbench.benchmarks.openml_tabular.tasks import BY_ID
    kinds = {}
    for spec in BY_ID.values():
        kinds.setdefault(spec.kind, set()).add((spec.metric, spec.higher_better))
    assert kinds["binary"] == {("roc_auc", True)}
    assert kinds["multiclass"] == {("log_loss", False)}
    assert kinds["regression"] == {("rmse", False)}


def test_prepare_rejects_class_collisions(tmp_path, monkeypatch):
    """Mixed-type targets (raw 1 and '1') stringify to colliding class
    columns — prep must fail loudly, not desync the probability matrix."""
    from arbench.benchmarks.openml_tabular import prepare as prep
    df = pd.DataFrame({"x": [1, 2, 3, 4], "label": [1, "1", 0, 0]})
    monkeypatch.setattr(prep, "_download_openml_csv",
                        lambda dataset_id: (df, "label", "toy prose"))
    spec = type("S", (), {"task_id": TASK, "dataset_id": 0, "target": "label",
                          "metric": "roc_auc", "higher_better": True,
                          "kind": "binary", "provenance": ""})()
    with pytest.raises(ValueError, match="collide"):
        prep.prepare_one(spec, tmp_path / "pub", tmp_path / "priv")


def test_load_task_refuses_stale_prepared_metadata(tmp_path):
    """The migration tripwire (cubic P1 on PR #11): prepared meta.json that
    disagrees with tasks.py (e.g. multiclass tasks still advertising accuracy
    after the log_loss switch) must refuse to serve, pointing at the refresh
    script — deployed data can never silently run on stale terms."""
    _stage_openml(tmp_path, metric="roc_auc", kind="binary")   # wine_quality
    bench = OpenMLTabular(data_dir=str(tmp_path / "public"),   # is regression
                          private_data_dir=str(tmp_path / "private"))
    with pytest.raises(RuntimeError, match="disagrees with tasks.py"):
        bench.load_task(TASK)


def test_refresh_syncs_metric_from_tasks_py(tmp_path, monkeypatch):
    """The stale-meta guard names refresh as the migration, so refresh must
    actually migrate (cubic P1 on PR #11): stale meta -> refreshed to the
    spec's metric/direction -> loadable again under the default guard."""
    import importlib.util
    from pathlib import Path as _P
    spec = importlib.util.spec_from_file_location(
        "refresh_openml_task_text",
        _P(__file__).resolve().parents[1] / "scripts"
        / "refresh_openml_task_text.py")
    refresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refresh)

    # stage wine_quality (spec: regression/rmse/"quality") with a STALE metric
    _stage_openml(tmp_path, metric="mse", kind="regression")
    prep = tmp_path / "public" / TASK / "prepared"
    meta = json.loads((prep / "meta.json").read_text())
    meta["target"] = "quality"
    (prep / "meta.json").write_text(json.dumps(meta))
    pd.DataFrame({"row_id": [0, 1], "x": [1, 2],
                  "quality": [5.0, 6.0]}).to_csv(prep / "train.csv", index=False)
    (prep / ".data_version").unlink(missing_ok=True)

    bench = _bench(tmp_path)
    bench.enforce_spec = True
    with pytest.raises(RuntimeError, match="disagrees with tasks.py"):
        bench.load_task(TASK)                          # stale: refused

    monkeypatch.setattr(refresh, "fetch_openml_description",
                        lambda dataset_id: "upstream prose")
    refresh.refresh_one(prep, tmp_path / "private" / TASK / "answers.csv")

    refreshed = json.loads((prep / "meta.json").read_text())
    assert refreshed["metric"] == "rmse"               # synced from BY_ID
    assert refreshed["higher_better"] is False
    task = bench.load_task(TASK)                       # guard now passes
    assert task.metadata["metric"] == "rmse"


def test_get_benchmark_refuses_the_enforce_spec_seam():
    import arbench
    with pytest.raises(TypeError, match="not a production knob"):
        arbench.get_benchmark("openml_tabular", enforce_spec=False)
