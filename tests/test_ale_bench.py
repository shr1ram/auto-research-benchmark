"""ale_bench_lite plugin: task view, format verdicts, judge-backed grading
(the judge itself is always faked — no ale-bench import, Docker, or network
anywhere in CI)."""
from __future__ import annotations

import json

import pytest

from arbench.benchmarks.ale_bench_lite.benchmark import (
    MAX_SUBMISSION_BYTES, ALEBenchLite,
)
from arbench.benchmarks.ale_bench_lite.tasks import LITE_PROBLEMS

TASK = "ahc008"


def _stage(root, score_type="maximize"):
    prep = root / TASK / "prepared"
    (prep / "cases").mkdir(parents=True)
    (prep / "problem.md").write_text(
        "# Toy AHC\n\nPlace things well. Score = sum of goodness.\n")
    (prep / "meta.json").write_text(json.dumps(
        {"problem_id": TASK, "score_type": score_type,
         "judge_version": "202301", "time_limit_s": 2,
         "n_public_cases": 2, "public_seeds": [0, 1]}))
    (prep / "cases" / "0000.txt").write_text("3 3\n")
    (prep / "cases" / "0001.txt").write_text("4 4\n")
    return prep


class _FakeResult:
    def __init__(self, score):
        self.overall_absolute_score = score
        self.overall_judge_result = "JudgeResult.ACCEPTED"
        self.case_results = [object(), object()]


class _FakeSession:
    """Stands in for ale_bench.start(...): records calls, serves a score."""

    def __init__(self, score=1234):
        self.score = score
        self.evaluated = None
        self.closed = False

    def private_eval(self, code, code_language):
        self.evaluated = (code, code_language)
        return _FakeResult(self.score), 42, 1500

    def close(self):
        self.closed = True


def test_lite_task_list_is_the_official_subset(tmp_path):
    bench = ALEBenchLite(data_dir=str(tmp_path))
    assert tuple(bench.list_tasks()) == LITE_PROBLEMS
    assert len(LITE_PROBLEMS) == 10


def test_load_task_view_and_contract(tmp_path):
    prep = _stage(tmp_path)
    bench = ALEBenchLite(data_dir=str(tmp_path))
    task = bench.load_task(TASK)
    assert task.submission_filename == "submission.py"
    assert "Place things well" in task.goal            # official statement
    assert "submission.py" in task.goal                # our contract
    assert str(prep / "cases") in task.goal            # public cases path
    assert task.metadata["score_type"] == "maximize"
    assert task.metadata["data_version"]
    # nothing private exists anywhere under the staged tree
    names = [p.name for p in prep.rglob("*")]
    assert not any("private" in n or "answer" in n for n in names)


def test_load_task_refuses_unknown_and_unprepared(tmp_path):
    bench = ALEBenchLite(data_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="unknown"):
        bench.load_task("ahc999")
    with pytest.raises(RuntimeError, match="not prepared"):
        bench.load_task(TASK)


def test_validate_submission_format_only(tmp_path):
    _stage(tmp_path)
    bench = ALEBenchLite(data_dir=str(tmp_path))
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"

    ok, reason = bench.validate_submission(task, sub)
    assert not ok and "not found" in reason
    sub.write_text("")
    ok, reason = bench.validate_submission(task, sub)
    assert not ok and "empty" in reason
    sub.write_text("def f(:\n")
    ok, reason = bench.validate_submission(task, sub)
    assert not ok and "not valid Python" in reason
    sub.write_text("# big\n" + "x = 1\n" * (MAX_SUBMISSION_BYTES // 6 + 1))
    ok, reason = bench.validate_submission(task, sub)
    assert not ok and "caps submissions" in reason
    sub.write_text("print(input())\n")
    ok, reason = bench.validate_submission(task, sub)
    assert ok and reason is None


def test_grade_runs_the_official_judge_and_closes_the_session(tmp_path):
    _stage(tmp_path)
    fake = _FakeSession(score=98765)
    bench = ALEBenchLite(data_dir=str(tmp_path),
                         session_factory=lambda pid: fake)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(input())\n")
    score = bench.grade(task, sub)
    assert score.valid and score.value == 98765.0
    assert score.is_higher_better is True              # maximize problem
    assert fake.evaluated == ("print(input())\n", "python")
    assert fake.closed                                 # session released
    assert score.details["rank"] == 42


def test_grade_direction_follows_score_type(tmp_path):
    _stage(tmp_path, score_type="minimize")
    bench = ALEBenchLite(data_dir=str(tmp_path),
                         session_factory=lambda pid: _FakeSession(score=10))
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(1)\n")
    score = bench.grade(task, sub)
    assert score.valid and score.is_higher_better is False


def test_grade_without_judge_is_invalid_not_a_crash(tmp_path):
    _stage(tmp_path)

    def no_judge(pid):
        raise RuntimeError("docker daemon not running")

    bench = ALEBenchLite(data_dir=str(tmp_path), session_factory=no_judge)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(1)\n")
    score = bench.grade(task, sub)
    assert not score.valid
    assert "judge unavailable" in score.details["reason"]


def test_grade_judge_failure_mid_eval_is_invalid_and_closes(tmp_path):
    _stage(tmp_path)

    class Exploding(_FakeSession):
        def private_eval(self, code, code_language):
            raise TimeoutError("judge container hung")

    fake = Exploding()
    bench = ALEBenchLite(data_dir=str(tmp_path),
                         session_factory=lambda pid: fake)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(1)\n")
    score = bench.grade(task, sub)
    assert not score.valid and "private_eval failed" in score.details["reason"]
    assert fake.closed


def test_split_facts_cover_every_lite_problem():
    from arbench.core.splits import load_split_meta
    meta = load_split_meta("ale_bench_lite")
    assert set(meta) == set(LITE_PROBLEMS)
    assert {m["family"] for m in meta.values()} == {"code"}
