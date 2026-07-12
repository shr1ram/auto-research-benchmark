"""ale_bench plugin: task view, format verdicts, judge-backed grading (the
judge itself is always faked — no ale-bench import, Docker, or network
anywhere in CI)."""
from __future__ import annotations

import json

import pytest

from arbench.benchmarks.ale_bench.benchmark import (
    MAX_SUBMISSION_BYTES, ALEBench,
)
from arbench.benchmarks.ale_bench.tasks import ALL_PROBLEMS, LITE_PROBLEMS

TASK = "ahc008"


def _stage(root, score_type="maximize"):
    prep = root / TASK / "prepared"
    (prep / "cases").mkdir(parents=True)
    (prep / "problem.md").write_text(
        "# Toy AHC\n\nPlace things well. Score = sum of goodness.\n")
    (prep / "meta.json").write_text(json.dumps(
        {"problem_id": TASK, "score_type": score_type,
         "judge_version": "202301", "time_limit_s": 2,
         "n_public_cases": 2, "public_seeds": [0, 1],
         "seed_regime": "lite"}))
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


def test_task_list_is_the_full_official_set(tmp_path):
    bench = ALEBench(data_dir=str(tmp_path))
    assert tuple(bench.list_tasks()) == ALL_PROBLEMS
    assert len(ALL_PROBLEMS) == 40
    assert len(LITE_PROBLEMS) == 10 and set(LITE_PROBLEMS) <= set(ALL_PROBLEMS)


def test_private_data_dir_is_refused_loudly():
    """get_benchmark(**kwargs) parity (cubic P1 on PR #12): the kwarg is
    accepted by signature but there IS no private tree — passing one is a
    config error, not a silent ignore."""
    with pytest.raises(ValueError, match="no private data tree"):
        ALEBench(data_dir="/tmp/x", private_data_dir="/tmp/private")
    ALEBench(data_dir="/tmp/x", private_data_dir=None)   # None is fine


def test_load_task_view_and_contract(tmp_path):
    prep = _stage(tmp_path)
    bench = ALEBench(data_dir=str(tmp_path))
    task = bench.load_task(TASK)
    assert task.submission_filename == "submission.py"
    assert "Place things well" in task.goal            # official statement
    assert "submission.py" in task.goal                # our contract
    assert str(prep / "cases") in task.goal            # public cases path
    assert task.metadata["score_type"] == "maximize"
    assert task.metadata["staged_seed_regime"] == "lite"
    assert task.metadata["data_version"]
    # nothing private exists anywhere under the staged tree
    names = [p.name for p in prep.rglob("*")]
    assert not any("private" in n or "answer" in n for n in names)


def test_load_task_pins_data_version(tmp_path):
    """Trust-on-first-use (cubic P2 on PR #12): first load stamps, a later
    load after silent data change fails LOUDLY."""
    prep = _stage(tmp_path)
    bench = ALEBench(data_dir=str(tmp_path))
    bench.load_task(TASK)                              # stamps .data_version
    assert (prep / ".data_version").exists()
    (prep / "cases" / "0000.txt").write_text("999 999\n")   # silent drift
    with pytest.raises(RuntimeError, match="data_version mismatch"):
        bench.load_task(TASK)


def test_load_task_refuses_unknown_and_unprepared(tmp_path):
    bench = ALEBench(data_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="unknown"):
        bench.load_task("ahc999")
    with pytest.raises(RuntimeError, match="not prepared"):
        bench.load_task(TASK)


def test_validate_submission_format_only(tmp_path):
    _stage(tmp_path)
    bench = ALEBench(data_dir=str(tmp_path))
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"

    ok, reason = bench.validate_submission(task, sub)
    assert not ok and "missing" in reason
    (tmp_path / "subdir.py").mkdir()                   # a DIRECTORY (cubic P3)
    ok, reason = bench.validate_submission(task, tmp_path / "subdir.py")
    assert not ok and "regular file" in reason
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
    bench = ALEBench(data_dir=str(tmp_path),
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
    assert score.details["seed_regime"] == "lite"      # default regime


def test_grade_direction_follows_score_type(tmp_path):
    _stage(tmp_path, score_type="minimize")
    bench = ALEBench(data_dir=str(tmp_path),
                     session_factory=lambda pid: _FakeSession(score=10))
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(1)\n")
    score = bench.grade(task, sub)
    assert score.valid and score.is_higher_better is False


def test_grade_unreadable_submission_is_invalid_not_a_crash(tmp_path):
    """cubic P2 on PR #12: a directory (or unreadable path) at the submission
    location must grade invalid BEFORE the external judge is started."""
    _stage(tmp_path)
    calls = []
    bench = ALEBench(data_dir=str(tmp_path),
                     session_factory=lambda pid: calls.append(pid))
    task = bench.load_task(TASK)
    (tmp_path / "submission.py").mkdir()               # directory, not a file
    score = bench.grade(task, tmp_path / "submission.py")
    assert not score.valid
    assert "unreadable" in score.details["reason"]
    assert calls == []                                 # judge never started


def test_grade_without_judge_is_invalid_not_a_crash(tmp_path):
    _stage(tmp_path)

    def no_judge(pid):
        raise RuntimeError("docker daemon not running")

    bench = ALEBench(data_dir=str(tmp_path), session_factory=no_judge)
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
    bench = ALEBench(data_dir=str(tmp_path),
                     session_factory=lambda pid: fake)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(1)\n")
    score = bench.grade(task, sub)
    assert not score.valid and "private_eval failed" in score.details["reason"]
    assert fake.closed


def test_split_facts_cover_every_problem_and_mark_the_lite_subset():
    from arbench.core.splits import load_split_meta
    meta = load_split_meta("ale_bench")
    assert set(meta) == set(ALL_PROBLEMS)
    assert {m["family"] for m in meta.values()} == {"code"}
    lite_marked = {tid for tid, m in meta.items() if m.get("subset") == "lite"}
    assert lite_marked == set(LITE_PROBLEMS)
