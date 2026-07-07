"""Harness contract tests using fakes — no real adapter/mlebench needed. Proves the
entry/exit contracts compose: a fake autoresearch system + fake benchmark run
end-to-end through the runner, and failures degrade to invalid scores.
"""
from __future__ import annotations

from pathlib import Path

from arbench.core.task import Task
from arbench.core.result import Score
from arbench.core.adapter import AutoResearchAdapter
from arbench.core.benchmark import Benchmark
from arbench.core.runner import run_one


class FakeBenchmark(Benchmark):
    name = "fake"

    def list_tasks(self):
        return ["t1"]

    def load_task(self, task_id):
        return Task(task_id=task_id, benchmark=self.name,
                    goal="predict the number 42", eval="exact match",
                    submission_filename="submission.csv")

    def grade(self, task, submission_path):
        if not Path(submission_path).exists():
            return Score.invalid("missing")
        content = Path(submission_path).read_text().strip()
        return Score(value=1.0 if content == "42" else 0.0, valid=True,
                     is_higher_better=True)


class GoodAdapter(AutoResearchAdapter):
    name = "good"

    def run(self, task, workspace):
        p = task.submission_path(workspace)
        p.write_text("42")
        return p


class EmptyAdapter(AutoResearchAdapter):
    name = "empty"

    def run(self, task, workspace):
        # produces nothing
        return task.submission_path(workspace)


class CrashAdapter(AutoResearchAdapter):
    name = "crash"

    def run(self, task, workspace):
        raise RuntimeError("boom")


class CrashAfterSubmissionAdapter(AutoResearchAdapter):
    """Produces a good submission, THEN dies (e.g. a transport failure in a late
    iteration). The submission must still be graded."""
    name = "crash_late"

    def run(self, task, workspace):
        task.submission_path(workspace).write_text("42")
        raise RuntimeError("boom-late")


def test_good_run_grades_high(tmp_path):
    r = run_one(GoodAdapter(), FakeBenchmark(), "t1", tmp_path / "good")
    assert r.score.valid and r.score.value == 1.0
    assert r.adapter_error is None
    assert r.submission_path and r.submission_path.exists()


def test_missing_submission_is_invalid_not_crash(tmp_path):
    r = run_one(EmptyAdapter(), FakeBenchmark(), "t1", tmp_path / "empty")
    assert not r.score.valid
    assert r.adapter_error is None  # didn't crash, just produced nothing


def test_adapter_crash_becomes_invalid_score(tmp_path):
    r = run_one(CrashAdapter(), FakeBenchmark(), "t1", tmp_path / "crash")
    assert not r.score.valid
    assert r.adapter_error is not None and "boom" in r.adapter_error


def test_crash_after_submission_is_still_graded(tmp_path):
    r = run_one(CrashAfterSubmissionAdapter(), FakeBenchmark(), "t1", tmp_path / "late")
    assert r.adapter_error is not None and "boom-late" in r.adapter_error
    assert r.score.valid and r.score.value == 1.0  # graded the best it produced
    assert "note" in r.score.details               # ...with the crash flagged


def test_rerun_clears_stale_adapter_outputs(tmp_path):
    ws = tmp_path / "reused"
    # Simulate a previous attempt's leftovers: adapter output trees + submission.
    for d in ("car_iters/iter_000", "car_best"):
        (ws / d).mkdir(parents=True)
        (ws / d / "submission.csv").write_text("42")
    (ws / "submission.csv").write_text("42")
    (ws / "llm_calls.jsonl").write_text('{"model": "stale"}\n')

    r = run_one(EmptyAdapter(), FakeBenchmark(), "t1", ws)
    # The failed re-run must NOT be graded on the previous attempt's files...
    assert not r.score.valid
    # ...because the stale trees were cleared before the adapter ran.
    assert not (ws / "car_iters").exists()
    assert not (ws / "car_best").exists()
    assert not (ws / "submission.csv").exists()


def test_result_serialises(tmp_path):
    r = run_one(GoodAdapter(), FakeBenchmark(), "t1", tmp_path / "good")
    d = r.to_dict()
    assert d["adapter"] == "good" and d["benchmark"] == "fake"
    assert d["score"]["value"] == 1.0
