"""ale_bench plugin: task view, format verdicts, and SELF-CONTAINED grading.

Grading tests are real: the submission genuinely executes as a subprocess
over staged case files, and the 'official scorer' is a stub python script
speaking the vis interface (`vis <input> <output>` -> 'Score = N'). No Rust,
Docker, or network anywhere; sandbox=False for portability (bwrap is
box-only)."""
from __future__ import annotations

import json
import stat
import sys

import pytest

from arbench.benchmarks.ale_bench.benchmark import (
    MAX_SUBMISSION_BYTES, PYTHON_TIME_SCALE, ALEBench,
    ReactiveTesterUnavailable, SandboxUnavailable, assert_sandbox_works,
    assert_tester_executes,
)
from arbench.benchmarks.ale_bench.tasks import ALL_PROBLEMS, LITE_PROBLEMS

TASK = "ahc011"


#: vis stub: score = the integer the solver printed, doubled; reject 'BAD'
VIS_STUB = f"""#!{sys.executable}
import sys
out = open(sys.argv[2]).read().strip()
if out == "BAD":
    print("invalid output")
else:
    print(f"Score = {{int(out) * 2}}")
"""


#: tester stub: reads the case N from ITS stdin, feeds N to the solver
#: (argv[1:] = the solver command), reads the solver's reply, scores it
#: (reply * 10) to stderr like the real tester. A reply of 'X' = invalid.
TESTER_STUB = f"""#!{sys.executable}
import sys, subprocess
n = sys.stdin.readline().strip()
p = subprocess.Popen(sys.argv[1:], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, text=True)
p.stdin.write(n + chr(10)); p.stdin.flush()
reply = p.stdout.readline().strip()
p.wait()
if reply == "X" or not reply:
    sys.stderr.write("invalid interaction" + chr(10))
else:
    sys.stderr.write(f"Score = {{int(reply) * 10}}" + chr(10))
"""


def _stage(root, score_type="maximize", problem_type="batch",
           n_private=3, time_limit=1.0):
    pub = root / "public" / TASK / "prepared"
    (pub / "cases").mkdir(parents=True)
    pub.joinpath("problem.md").write_text(
        "# Toy AHC\n\nMaximise goodness. Score = sum of goodness.\n")
    pub.joinpath("meta.json").write_text(json.dumps(
        {"problem_id": TASK, "problem_type": problem_type,
         "score_type": score_type, "time_limit_s": time_limit,
         "n_public_cases": 2, "n_private_cases": n_private,
         "public_seeds": [0, 1], "seed_regime": "lite"}))
    (pub / "cases" / "case_0000.txt").write_text("3\n")
    (pub / "cases" / "case_0001.txt").write_text("4\n")

    priv = root / "private" / TASK
    (priv / "cases").mkdir(parents=True)
    (priv / "bin").mkdir()
    for i in range(n_private):
        (priv / "cases" / f"case_{i:04d}.txt").write_text(f"{i + 10}\n")
    if problem_type == "reactive":
        (pub / "bin").mkdir()
        for base in (pub, priv):
            t = base / "bin" / "tester"
            t.write_text(TESTER_STUB)
            t.chmod(t.stat().st_mode | stat.S_IEXEC)
    else:
        vis = priv / "bin" / "vis"
        vis.write_text(VIS_STUB)
        vis.chmod(vis.stat().st_mode | stat.S_IEXEC)
    return pub, priv


def _bench(tmp_path, **kw):
    return ALEBench(data_dir=str(tmp_path / "public"),
                    private_data_dir=str(tmp_path / "private"),
                    sandbox=False, **kw)


# ------------------------------------------------------------------ tasks

def test_task_list_is_the_full_official_set(tmp_path):
    bench = _bench(tmp_path)
    assert tuple(bench.list_tasks()) == ALL_PROBLEMS
    assert len(ALL_PROBLEMS) == 40
    assert len(LITE_PROBLEMS) == 10 and set(LITE_PROBLEMS) <= set(ALL_PROBLEMS)


def test_private_root_defaults_to_sibling(tmp_path):
    bench = ALEBench(data_dir=str(tmp_path / "public"), sandbox=False)
    assert bench.private_dir == tmp_path / "private"


def test_load_task_view_and_contract(tmp_path):
    pub, priv = _stage(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    assert task.submission_filename == "submission.py"
    assert "Maximise goodness" in task.goal            # official statement
    assert "submission.py" in task.goal                # our contract
    assert str(pub / "cases") in task.goal             # public cases path
    assert task.metadata["score_type"] == "maximize"
    assert task.metadata["seed_regime"] == "lite"
    assert task.metadata["data_version"]
    # the goal and the public tree never leak the private side (macOS tmp
    # paths contain "/private/", so check real markers, not the substring)
    assert str(priv) not in task.goal
    assert "private_seeds" not in (pub / "meta.json").read_text()
    names = [p.name for p in pub.rglob("*")]
    assert "seeds.json" not in names and "bin" not in names


def test_load_task_pins_data_version(tmp_path):
    pub, _ = _stage(tmp_path)
    bench = _bench(tmp_path)
    bench.load_task(TASK)                              # stamps .data_version
    (pub / "cases" / "case_0000.txt").write_text("999\n")   # silent drift
    with pytest.raises(RuntimeError, match="data_version mismatch"):
        bench.load_task(TASK)


def test_load_task_refuses_unknown_and_unprepared(tmp_path):
    bench = _bench(tmp_path)
    with pytest.raises(RuntimeError, match="unknown"):
        bench.load_task("ahc999")
    with pytest.raises(RuntimeError, match="not prepared"):
        bench.load_task(TASK)


def test_reactive_task_view_carries_the_interactive_contract(tmp_path):
    _stage(tmp_path, problem_type="reactive")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    assert task.metadata["problem_type"] == "reactive"
    assert "INTERACTIVE" in task.goal and "flush" in task.goal.lower()
    assert "tester" in task.goal            # the self-eval command names it


def test_reactive_without_public_tester_is_refused(tmp_path):
    """A reactive problem staged without the public tester (old batch-only
    staging) can't run its self-eval — refuse with a re-prepare hint."""
    _stage(tmp_path, problem_type="reactive")
    import shutil as _sh
    _sh.rmtree(tmp_path / "public" / TASK / "prepared" / "bin")
    (tmp_path / "public" / TASK / "prepared" / ".data_version").unlink(missing_ok=True)
    bench = _bench(tmp_path)
    with pytest.raises(RuntimeError, match="tester is not staged"):
        bench.load_task(TASK)


def test_grade_reactive_runs_the_tester_and_scores(tmp_path):
    """Echo solver: reads N, prints N -> tester scores N*10. Private cases
    10/11/12 -> 100+110+120 = 330."""
    _stage(tmp_path, problem_type="reactive", time_limit=10.0)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("import sys\nprint(sys.stdin.readline().strip()); "
                   "sys.stdout.flush()\n")
    score = bench.grade(task, sub)
    assert score.valid and score.value == 330.0
    assert score.details["problem_type"] == "reactive"
    assert score.details["n_rejected"] == 0


def test_grade_reactive_invalid_interaction_is_rejected(tmp_path):
    """A solver that replies 'X' (invalid move) on case 11 -> that case
    rejected, others score."""
    _stage(tmp_path, problem_type="reactive", time_limit=10.0)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("import sys\nn=int(sys.stdin.readline())\n"
                   "print('X' if n==11 else n); sys.stdout.flush()\n")
    score = bench.grade(task, sub)
    assert score.valid and score.value == 220.0     # 100 + 120
    assert score.details["n_rejected"] == 1


def test_grade_reactive_solver_cannot_forge_a_score_via_stderr(tmp_path):
    """SECURITY: the solver inherits the tester's stderr, so a solver that
    prints 'Score = <huge>' to stderr could forge the official score. The
    solver's stderr is /dev/null'd (only the tester's reaches the grader), so
    the forgery scores nothing. Verified exploitable before the fix."""
    _stage(tmp_path, problem_type="reactive")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    # never completes the protocol; just forges a score on stderr and exits
    sub.write_text("import sys\n"
                   "sys.stderr.write('Score = 999999999' + chr(10))\n"
                   "sys.exit(1)\n")
    score = bench.grade(task, sub)
    # the forged 999999999 must NOT appear as the score
    assert not score.valid or score.value != 999999999.0 * len(
        list((tmp_path / "private" / TASK / "cases").glob("case_*.txt")))
    assert score.details.get("reason", "").startswith("no case") or \
        score.details.get("n_rejected") == score.details.get("n_cases")


# --------------------------------------------------------------- verdicts

def test_validate_submission_format_only(tmp_path):
    _stage(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"

    ok, reason = bench.validate_submission(task, sub)
    assert not ok and "missing" in reason
    (tmp_path / "subdir.py").mkdir()
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


# ---------------------------------------------------------------- grading

def test_grade_executes_and_scores_with_the_official_interface(tmp_path):
    """Echo solver over cases 10/11/12 -> stub vis doubles -> 20+22+24."""
    _stage(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(input().strip())\n")
    score = bench.grade(task, sub)
    assert score.valid and score.value == 66.0
    assert score.is_higher_better is True
    d = score.details
    assert d["n_cases"] == 3 and d["n_tle"] == d["n_error"] == d["n_rejected"] == 0
    assert d["python_time_scale"] == PYTHON_TIME_SCALE
    assert d["seed_regime"] == "lite"


def test_grade_counts_tle_error_and_rejected_cases(tmp_path):
    """Case 10 answers, case 11 crashes, case 12 emits a rejected output —
    maximize: failures contribute 0, run stays valid."""
    _stage(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text(
        "n = int(input())\n"
        "if n == 11: raise SystemExit(3)\n"
        "print('BAD' if n == 12 else n)\n")
    score = bench.grade(task, sub)
    assert score.valid and score.value == 20.0          # only case 10 scores
    assert score.details["n_error"] == 1
    assert score.details["n_rejected"] == 1


def test_grade_tle_is_killed_and_scored_zero(tmp_path):
    _stage(tmp_path, time_limit=0.2)                    # 0.2s * scale = 0.6s
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text(
        "import time\n"
        "n = int(input())\n"
        "if n == 10: time.sleep(60)\n"
        "print(n)\n")
    score = bench.grade(task, sub)
    assert score.valid and score.value == 46.0          # 22 + 24
    assert score.details["n_tle"] == 1


def test_grade_minimize_with_failures_is_invalid_not_flattering(tmp_path):
    """On a minimize problem a failed case contributing 0 would IMPROVE the
    total — must grade invalid instead of reporting a misleading number."""
    _stage(tmp_path, score_type="minimize")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("n = int(input())\n"
                   "if n == 11: raise SystemExit(3)\n"
                   "print(n)\n")
    score = bench.grade(task, sub)
    assert not score.valid
    assert "minimize" in score.details["reason"]
    # clean minimize submissions still grade, lower-better
    sub.write_text("print(input().strip())\n")
    score = bench.grade(task, sub)
    assert score.valid and score.is_higher_better is False


def test_grade_all_cases_failing_is_invalid(tmp_path):
    _stage(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("raise SystemExit(1)\n")
    score = bench.grade(task, sub)
    assert not score.valid and "no case produced" in score.details["reason"]


def test_grade_without_staged_private_tree_is_invalid(tmp_path):
    pub, priv = _stage(tmp_path)
    import shutil as _sh
    _sh.rmtree(priv)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(1)\n")
    score = bench.grade(task, sub)
    assert not score.valid
    assert "not staged" in score.details["reason"]


def test_grade_unreadable_submission_is_invalid(tmp_path):
    _stage(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    (tmp_path / "submission.py").mkdir()               # directory, not a file
    score = bench.grade(task, tmp_path / "submission.py")
    assert not score.valid


# ----------------------------------------------------------------- splits

def test_all_40_problems_are_in_the_draw():
    """Reactive support (PR #14) means ALL 40 problems — batch AND reactive —
    are runnable, so none are excluded and the draw covers the full set."""
    from arbench.core.splits import families, assign_roles
    fams = families(("ale_bench",))
    assert len(fams["code"]) == 40
    roles = assign_roles({"bank": 0.55, "val": 0.30, "test": 0.15}, seed=0)
    drawn = {t for (b, t), r in roles.items() if b == "ale_bench"}
    assert len(drawn) == 40


def test_split_facts_cover_every_problem_and_mark_the_lite_subset():
    from arbench.core.splits import load_split_meta
    meta = load_split_meta("ale_bench")
    assert set(meta) == set(ALL_PROBLEMS)
    assert {m["family"] for m in meta.values()} == {"code"}
    lite_marked = {tid for tid, m in meta.items() if m.get("subset") == "lite"}
    assert lite_marked == set(LITE_PROBLEMS)


def test_grade_runaway_stdout_is_rejected_not_ooM(tmp_path):
    """A solver streaming unbounded stdout must become a rejected case, not
    buffer into grader RAM (cubic P1 on PR #12). MAX_OUTPUT_BYTES is patched
    down so the test writes KBs, not MBs."""
    import arbench.benchmarks.ale_bench.benchmark as B
    _stage(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    # case 10 floods stdout; 11/12 answer normally
    sub.write_text(
        "n = int(input())\n"
        "print('X' * 100000 if n == 10 else n)\n")
    orig = B.MAX_OUTPUT_BYTES
    B.MAX_OUTPUT_BYTES = 1024
    try:
        score = bench.grade(task, sub)
    finally:
        B.MAX_OUTPUT_BYTES = orig
    assert score.valid and score.value == 46.0          # 22 + 24, case 10 out
    assert score.details["n_rejected"] == 1


def test_grade_detects_private_case_count_mismatch(tmp_path):
    """A partially-staged private tree must not grade silently on fewer
    cases (a truncated grade reads as a genuine low score)."""
    pub, priv = _stage(tmp_path, n_private=3)          # meta says 3
    (priv / "cases" / "case_0002.txt").unlink()        # now only 2 on disk
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(input().strip())\n")
    score = bench.grade(task, sub)
    assert not score.valid and "expects 3" in score.details["reason"]


def test_grade_detects_private_tree_drift(tmp_path):
    """The private cases/scorer are grade inputs — silent drift must be
    caught, same as public data_version drift."""
    from arbench.core.data_version import verify_data_version
    pub, priv = _stage(tmp_path)
    verify_data_version(priv)                          # stamp .data_version
    (priv / "cases" / "case_0000.txt").write_text("999\n")   # silent drift
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(input().strip())\n")
    score = bench.grade(task, sub)
    assert not score.valid and "drift" in score.details["reason"]


# --------------------------------------------------------- per-case details
# The SUM alone cannot reconstruct the official ALE-Bench metric: the relative
# contests re-normalise per case, and there a REJECTED case is not a zero-cost
# answer. These pin the per-case record that makes the metric reconstructible
# without re-running agent code over the private cases.
def test_details_carry_one_row_per_case_in_case_order(tmp_path):
    """Cases 10/11/12 -> stub vis doubles -> 20/22/24, in the case order."""
    _stage(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(input().strip())\n")
    score = bench.grade(task, sub)
    assert score.details["cases"] == [
        {"case": "case_0000.txt", "status": "ok", "score": 20},
        {"case": "case_0001.txt", "status": "ok", "score": 22},
        {"case": "case_0002.txt", "status": "ok", "score": 24}]
    # the per-case scores must reconstruct the aggregate exactly
    assert sum(c["score"] for c in score.details["cases"]) == score.value


def test_per_case_statuses_distinguish_error_from_rejected(tmp_path):
    """Case 10 answers, case 11 crashes, case 12 emits a rejected output.
    An invalid answer is REJECTED, not a zero score — conflating the two is
    what inverts a minimize task."""
    _stage(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text(
        "n = int(input())\n"
        "if n == 11: raise SystemExit(3)\n"
        "print('BAD' if n == 12 else n)\n")
    score = bench.grade(task, sub)
    assert score.details["cases"] == [
        {"case": "case_0000.txt", "status": "ok", "score": 20},
        {"case": "case_0001.txt", "status": "error", "score": None},
        {"case": "case_0002.txt", "status": "rejected", "score": None}]
    # the rows must agree with the aggregate counters they sit beside
    d = score.details
    assert d["n_error"] == sum(1 for c in d["cases"] if c["status"] == "error")
    assert d["n_rejected"] == sum(1 for c in d["cases"]
                                  if c["status"] == "rejected")


def test_a_timed_out_case_is_recorded_as_tle(tmp_path):
    _stage(tmp_path, time_limit=0.3)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("import time\n"
                   "n = int(input())\n"
                   "if n == 11: time.sleep(30)\n"
                   "print(n)\n")
    score = bench.grade(task, sub)
    rows = {c["case"]: c["status"] for c in score.details["cases"]}
    assert rows["case_0001.txt"] == "tle"
    assert score.details["n_tle"] == 1


def test_an_invalid_minimize_grade_still_carries_per_case_rows(tmp_path):
    """A minimize task with any failed case returns invalid — but the
    per-case record is exactly what official relative scoring needs, so it
    must survive the invalid return."""
    _stage(tmp_path, score_type="minimize")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("n = int(input())\n"
                   "print('BAD' if n == 12 else n)\n")
    score = bench.grade(task, sub)
    assert not score.valid and score.value is None
    assert [c["status"] for c in score.details["cases"]] == [
        "ok", "ok", "rejected"]
    # the reason stays human-sized: the per-case list is NOT interpolated
    assert "case_0000.txt" not in score.details["reason"]


def test_an_all_failed_grade_carries_per_case_rows(tmp_path):
    _stage(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("raise SystemExit(1)\n")
    score = bench.grade(task, sub)
    assert not score.valid
    assert [c["status"] for c in score.details["cases"]] == ["error"] * 3
    assert "case_0000.txt" not in score.details["reason"]


def test_early_invalid_returns_carry_no_case_rows(tmp_path):
    """Nothing was graded, so there is no per-case record to report — the
    key must be ABSENT rather than an empty list that reads as '0 cases'."""
    _stage(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    score = bench.grade(task, tmp_path / "does-not-exist.py")
    assert not score.valid and "cases" not in score.details


def test_per_case_rows_are_json_serialisable(tmp_path):
    """The rows travel to the consuming loop's manifest as JSON."""
    _stage(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(input().strip())\n")
    score = bench.grade(task, sub)
    assert json.loads(json.dumps(score.details["cases"])) \
        == score.details["cases"]


def test_reactive_grading_records_per_case_rows(tmp_path):
    _stage(tmp_path, problem_type="reactive")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(input().strip(), flush=True)\n")
    score = bench.grade(task, sub)
    assert len(score.details["cases"]) == 3
    assert all(c["status"] in ("ok", "tle", "error", "rejected")
               for c in score.details["cases"])


# ------------------------------------------------------- concurrent grading
# Cases grade concurrently. What must hold: the SCORE is unchanged, the rows
# stay in case order whatever order they finish in, and a bwrap setup race
# (the sandbox misfiring, not the submission) does not become a failed case.
# Measured evidence for the ceiling is in the MAX_GRADE_WORKERS comment.

def test_grading_workers_derive_from_the_core_share(monkeypatch):
    """P is derived from this process's core share, not hardcoded: grading
    runs inside one of `max_workers` concurrent cells, and the loop publishes
    that cell's share in the same variable it caps agent threads with."""
    from arbench.benchmarks.ale_bench.benchmark import (
        CORES_PER_SOLVER, MAX_GRADE_WORKERS, grade_workers,
    )
    monkeypatch.delenv("ARBENCH_ALE_GRADE_WORKERS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    assert grade_workers() == 4 // CORES_PER_SOLVER
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert grade_workers() == 1                 # a 1-core share never forks
    monkeypatch.setenv("OMP_NUM_THREADS", "256")
    assert grade_workers() == MAX_GRADE_WORKERS  # the measured ceiling holds
    monkeypatch.setenv("OMP_NUM_THREADS", "not-a-number")
    assert grade_workers() >= 1                  # junk must not raise
    # an explicit share argument bypasses the environment entirely
    assert grade_workers(core_share=2) == 1


def test_grading_workers_env_override_is_bounded(monkeypatch):
    from arbench.benchmarks.ale_bench.benchmark import (
        MAX_GRADE_WORKERS, grade_workers,
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("ARBENCH_ALE_GRADE_WORKERS", "6")
    assert grade_workers() == 6                  # overrides the derivation
    monkeypatch.setenv("ARBENCH_ALE_GRADE_WORKERS", "64")
    assert grade_workers() == MAX_GRADE_WORKERS  # never above the ceiling
    monkeypatch.setenv("ARBENCH_ALE_GRADE_WORKERS", "junk")
    assert grade_workers() == 1                  # falls back to the share


def test_workers_argument_is_clamped_to_the_measured_ceiling(tmp_path):
    from arbench.benchmarks.ale_bench.benchmark import MAX_GRADE_WORKERS
    _stage(tmp_path)
    assert _bench(tmp_path, workers=4).workers == 4
    assert _bench(tmp_path, workers=99).workers == MAX_GRADE_WORKERS
    assert _bench(tmp_path, workers=0).workers == 1
    assert _bench(tmp_path).workers is None      # None = derive per grade


def test_parallel_and_sequential_grades_are_identical(tmp_path):
    """A deterministic submission must score the same at P=1 and P=8, with
    identical per-case rows — the whole safety claim in one assertion."""
    _stage(tmp_path, n_private=12)
    task = _bench(tmp_path).load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(input().strip())\n")
    seq = _bench(tmp_path, workers=1).grade(task, sub)
    par = _bench(tmp_path, workers=8).grade(task, sub)
    assert seq.valid and par.valid
    assert seq.value == par.value
    assert seq.details["cases"] == par.details["cases"]
    assert par.details["grade_workers"] == 8
    assert seq.details["grade_workers"] == 1


def test_case_rows_stay_in_case_order_when_completion_order_differs(tmp_path):
    """Earlier cases sleep LONGER, so they finish last. The rows must still
    be in case_files order: official relative scoring consumes them
    positionally, so a completion-ordered list silently mis-scores."""
    _stage(tmp_path, n_private=8, time_limit=10.0)
    task = _bench(tmp_path).load_task(TASK)
    sub = tmp_path / "submission.py"
    # case values are 10..17; smaller value = longer sleep = later finish
    sub.write_text("import time\n"
                   "n = int(input())\n"
                   "time.sleep((18 - n) * 0.05)\n"
                   "print(n)\n")
    score = _bench(tmp_path, workers=8).grade(task, sub)
    assert score.valid
    assert [c["case"] for c in score.details["cases"]] == [
        f"case_{i:04d}.txt" for i in range(8)]
    assert [c["score"] for c in score.details["cases"]] == [
        (i + 10) * 2 for i in range(8)]          # vis stub doubles


def test_counters_agree_with_the_rows_under_concurrency(tmp_path):
    """Mixed outcomes graded concurrently: the aggregate counters are
    recomputed from the collected rows, so they cannot drift apart."""
    _stage(tmp_path, n_private=8, time_limit=0.4)
    task = _bench(tmp_path).load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("import time\n"
                   "n = int(input())\n"
                   "if n == 11: raise SystemExit(3)\n"        # error
                   "if n == 12: time.sleep(30)\n"             # tle
                   "print('BAD' if n == 13 else n)\n")        # rejected
    score = _bench(tmp_path, workers=8).grade(task, sub)
    d = score.details
    assert d["n_error"] == sum(1 for c in d["cases"] if c["status"] == "error")
    assert d["n_tle"] == sum(1 for c in d["cases"] if c["status"] == "tle")
    assert d["n_rejected"] == sum(1 for c in d["cases"]
                                 if c["status"] == "rejected")
    assert d["n_error"] == d["n_tle"] == d["n_rejected"] == 1
    assert score.value == sum(c["score"] for c in d["cases"]
                              if c["score"] is not None)


def test_concurrent_solvers_do_not_share_a_cwd(tmp_path):
    """Each case runs in its OWN cwd. A solver writing a fixed-name scratch
    file beside itself would otherwise be clobbered by every other case
    (measured: 21/24 corrupted at P=8 with no sandbox)."""
    _stage(tmp_path, n_private=8, time_limit=10.0)
    task = _bench(tmp_path).load_task(TASK)
    sub = tmp_path / "submission.py"
    # write my own value to a FIXED name, pause, read it back: a shared cwd
    # returns some other case's value and the vis stub scores the wrong case
    sub.write_text("import time\n"
                   "n = int(input())\n"
                   "open('scratch.txt', 'w').write(str(n))\n"
                   "time.sleep(0.3)\n"
                   "print(open('scratch.txt').read().strip())\n")
    score = _bench(tmp_path, workers=8).grade(task, sub)
    assert score.valid
    assert [c["score"] for c in score.details["cases"]] == [
        (i + 10) * 2 for i in range(8)]


def test_a_bwrap_setup_race_is_retried_not_recorded_as_a_failure(tmp_path,
                                                                monkeypatch):
    """bwrap's mount setup racing under concurrency exits nonzero with no
    output — which reads exactly like a crashed submission and cost one
    measured run -174,065 points. The narrow signature is retried; the
    retry is counted so it is visible."""
    import arbench.benchmarks.ale_bench.benchmark as B
    _stage(tmp_path)
    task = _bench(tmp_path).load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(input().strip())\n")

    fails = {"left": 1}          # fail the FIRST attempt of each case only
    real = B._run_case

    def flaky(submission, case_file, out_path, timeout_s, sandbox):
        if fails["left"] > 0:
            fails["left"] -= 1
            return "error", (b"bwrap: Can't bind mount /oldroot/ on /newroot/: "
                             b"Unable to remount recursively with correct "
                             b"flags: No such file or directory\n")
        return real(submission, case_file, out_path, timeout_s, sandbox)

    monkeypatch.setattr(B, "_run_case", flaky)
    # the signature is only consulted when a sandbox is actually in use; a
    # harmless prefix stands in for bwrap so the test needs no bwrap binary
    monkeypatch.setattr(B, "_sandbox_prefix", lambda *a, **k: ["env"])
    bench = ALEBench(data_dir=str(tmp_path / "public"),
                     private_data_dir=str(tmp_path / "private"),
                     sandbox=True, workers=1)
    score = bench.grade(task, sub)
    assert score.valid and score.value == 66.0       # nothing lost to the race
    assert score.details["n_error"] == 0
    assert score.details["n_bwrap_retries"] == 1


def test_a_genuine_submission_error_is_never_retried_away(tmp_path,
                                                          monkeypatch):
    """The retry must not mask a broken submission. A solver that crashes
    while echoing bwrap's words on stderr still fails: the signature is
    anchored to bwrap's own line PREFIX, so mid-line text never matches."""
    import arbench.benchmarks.ale_bench.benchmark as B
    _stage(tmp_path)
    task = _bench(tmp_path).load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text(
        "import sys\n"
        "n = int(input())\n"
        "if n == 11:\n"
        "    sys.stderr.write("
        "'sandbox said bwrap: Unable to remount recursively' + chr(10))\n"
        "    raise SystemExit(3)\n"
        "print(n)\n")
    monkeypatch.setattr(B, "_sandbox_prefix", lambda *a, **k: ["env"])
    bench = ALEBench(data_dir=str(tmp_path / "public"),
                     private_data_dir=str(tmp_path / "private"),
                     sandbox=True, workers=4)
    score = bench.grade(task, sub)
    assert score.valid and score.value == 44.0       # 20 + 24; case 11 errors
    assert score.details["sandboxed"] is True
    assert score.details["n_error"] == 1
    assert score.details["n_bwrap_retries"] == 0

    # the signature itself: bwrap's own line, and only at a line start
    assert B._BWRAP_SETUP_RACE_RE.search(
        b"bwrap: Can't bind mount /oldroot/ on /newroot/")
    assert not B._BWRAP_SETUP_RACE_RE.search(
        b"my solver logged bwrap: Can't bind mount")
    assert not B._BWRAP_SETUP_RACE_RE.search(b"Traceback: KeyError")


def test_an_unretryable_race_still_records_the_error(tmp_path, monkeypatch):
    """Retries are BOUNDED: a case that races every attempt records the error
    exactly as it does today rather than retrying forever."""
    import arbench.benchmarks.ale_bench.benchmark as B
    _stage(tmp_path)
    task = _bench(tmp_path).load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(input().strip())\n")
    monkeypatch.setattr(B, "_run_case", lambda *a, **k: (
        "error", b"bwrap: Unable to remount recursively\n"))
    monkeypatch.setattr(B, "_sandbox_prefix", lambda *a, **k: ["env"])
    monkeypatch.setattr(B, "_BWRAP_RETRY_BACKOFF_S", 0.0)
    bench = ALEBench(data_dir=str(tmp_path / "public"),
                     private_data_dir=str(tmp_path / "private"),
                     sandbox=True, workers=1)
    score = bench.grade(task, sub)
    # a bwrap failure that survives the retries is now a HARNESS fault: the
    # grade refuses loudly (infra) instead of reporting the submission as
    # scoreless — the misread that produced campaign-wide 0% reactive
    # completion (2026-08-10). The retries still happened (and are still
    # the retry test's business); the AGGREGATE is an infra refusal.
    assert not score.valid
    assert score.details.get("infra") is True
    assert "sandbox setup failed on 3/3" in score.details["reason"]


def test_reactive_grading_is_identical_under_concurrency(tmp_path):
    """Reactive cases carry no shared state — the private-root tmpfs lives in
    each case's own bwrap mount namespace and the score arrives on a private
    pipe — so they grade concurrently too."""
    _stage(tmp_path, problem_type="reactive", n_private=8, time_limit=10.0)
    task = _bench(tmp_path).load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(input().strip(), flush=True)\n")
    seq = _bench(tmp_path, workers=1).grade(task, sub)
    par = _bench(tmp_path, workers=8).grade(task, sub)
    assert seq.valid and par.valid and seq.value == par.value
    assert seq.details["cases"] == par.details["cases"]


def test_minimize_invalid_path_survives_concurrency(tmp_path):
    """The invalid-return paths are unchanged by concurrency: a minimize task
    with any failed case still refuses to report a flattering total, and still
    carries its per-case rows."""
    _stage(tmp_path, score_type="minimize", n_private=8)
    task = _bench(tmp_path).load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("n = int(input())\n"
                   "print('BAD' if n == 13 else n)\n")
    score = _bench(tmp_path, workers=8).grade(task, sub)
    assert not score.valid and "minimize" in score.details["reason"]
    assert score.is_higher_better is False
    assert [c["status"] for c in score.details["cases"]] == [
        "ok", "ok", "ok", "rejected", "ok", "ok", "ok", "ok"]


# ------------------------------------------- fail-loud: unrunnable tester
#
# E2 post-mortem: on a host whose glibc could not satisfy the prebuilt Rust
# testers, the tester never ran. The failure surfaced only inside AGENT code,
# which caught it and reported `mean = 0.0`; the harness recorded 2,499 of
# 3,600 attempts as SUCCESSFUL zeros. The harness cannot stop an agent writing
# `except: pass`, so it refuses to hand out the task at all.


def _break_tester(pub, message, returncode=1):
    """Replace the staged public tester with one that reproduces a loader
    failure: the message on STDERR and a nonzero exit, which is what ld.so
    itself does when it cannot resolve a binary."""
    t = pub / "bin" / "tester"
    t.write_text(f"""#!{sys.executable}
import sys
sys.stderr.write({message!r} + chr(10))
sys.exit({returncode})
""")
    t.chmod(t.stat().st_mode | stat.S_IEXEC)


def test_reactive_task_with_unloadable_tester_is_refused(tmp_path):
    """THE E2 DEFECT. A tester the loader rejects must kill the task at
    load_task, before any attempt runs — not reach the agent as a catchable
    error that becomes a plausible 0.0."""
    pub, _ = _stage(tmp_path, problem_type="reactive")
    _break_tester(pub, "/data/bin/tester: /lib64/libc.so.6: version "
                       "`GLIBC_2.34' not found (required by /data/bin/tester)")
    with pytest.raises(ReactiveTesterUnavailable) as e:
        _bench(tmp_path).load_task(TASK)
    # the operator gets the real cause, not a downstream symptom
    assert "GLIBC_2.34" in str(e.value)


@pytest.mark.parametrize("message", [
    "tester: error while loading shared libraries: libgcc_s.so.1: cannot "
    "open shared object file",
    "tester: symbol lookup error: undefined symbol: __libc_start_main",
])
def test_other_loader_failures_are_refused_too(tmp_path, message):
    """The glibc-version string is one of several ways ld.so reports that a
    prebuilt binary cannot run; all of them mean the same thing here."""
    pub, _ = _stage(tmp_path, problem_type="reactive")
    _break_tester(pub, message)
    with pytest.raises(ReactiveTesterUnavailable):
        _bench(tmp_path).load_task(TASK)


def test_non_executable_tester_is_refused(tmp_path):
    pub, _ = _stage(tmp_path, problem_type="reactive")
    t = pub / "bin" / "tester"
    t.chmod(0o644)
    with pytest.raises(ReactiveTesterUnavailable):
        _bench(tmp_path).load_task(TASK)


def test_working_tester_loads_normally(tmp_path):
    """The unchanged path: a tester that RUNS yields a task exactly as
    before. This is the regression guard on the probe — it must not reject
    working binaries."""
    _stage(tmp_path, problem_type="reactive")
    task = _bench(tmp_path).load_task(TASK)
    assert task.metadata["problem_type"] == "reactive"
    assert "Maximise goodness" in task.goal


def test_probe_accepts_a_tester_that_exits_nonzero_on_bare_exec(tmp_path):
    """A usage message + nonzero exit means the binary LOADED and reached its
    own argument parser. That is a working tester (the real ALE testers print
    usage), and the probe must not confuse it with a loader failure."""
    pub, _ = _stage(tmp_path, problem_type="reactive")
    t = pub / "bin" / "tester"
    t.write_text(f"""#!{sys.executable}
import sys
sys.stderr.write("Usage: tester <command> [<args>...]" + chr(10))
sys.exit(2)
""")
    t.chmod(t.stat().st_mode | stat.S_IEXEC)
    assert_tester_executes(t)                    # does not raise
    assert _bench(tmp_path).load_task(TASK).metadata["problem_type"] == \
        "reactive"


def test_probe_does_not_flag_a_running_testers_own_errno_text(tmp_path):
    """A tester that runs and complains about a missing FILE says 'No such
    file or directory' — an errno string, not an ld.so diagnostic. Flagging it
    would refuse working tasks, so the probe keys on loader phrases only."""
    pub, _ = _stage(tmp_path, problem_type="reactive")
    _break_tester(pub, "tester: cannot open case.txt: No such file or "
                       "directory")
    assert_tester_executes(pub / "bin" / "tester")        # does not raise


def test_batch_tasks_do_not_probe_a_tester(tmp_path):
    """Batch problems never exec a binary for self-scoring (pure Python
    scoring rule), so their public bin/ is empty and the probe must not
    apply."""
    _stage(tmp_path, problem_type="batch")
    task = _bench(tmp_path).load_task(TASK)
    assert task.metadata["problem_type"] == "batch"


def test_reactive_contract_promises_harness_scoring(tmp_path):
    """The contract text tells the agent the HARNESS runs the official
    tester on every public case and that no result.json is needed; the
    tester remains staged for optional self-testing. Asserted here because
    it is prompt text: changing it changes the config hash."""
    _stage(tmp_path, problem_type="reactive")
    goal = _bench(tmp_path).load_task(TASK).goal
    assert "OFFICIAL judge (`tester`)" in goal
    assert "no result.json is needed" in goal
    assert "tester" in goal and "Score = N" in goal   # self-test recipe


def test_requested_sandbox_without_bwrap_raises_at_construction(tmp_path,
                                                                monkeypatch):
    """An explicitly requested sandbox that cannot be built must RAISE, and
    must do so at CONSTRUCTION: grade() is contractually forbidden from
    raising (Benchmark ABC), and an operator who asked for confinement needs
    to know before a grid starts, not per-case mid-run."""
    import arbench.benchmarks.ale_bench.benchmark as B
    monkeypatch.setattr(B.shutil, "which", lambda name: None)
    with pytest.raises(SandboxUnavailable) as e:
        ALEBench(data_dir=str(tmp_path / "public"),
                 private_data_dir=str(tmp_path / "private"), sandbox=True)
    assert "bwrap" in str(e.value)


def test_grade_never_raises_for_an_unavailable_sandbox(tmp_path, monkeypatch):
    """The ABC contract itself: `grade` must return Score.invalid(...), never
    raise. Guarding this directly because a raise here does not merely escape
    a grid — Benchmark.validate_submission catches Exception broadly and
    converts it into a generic FORMAT rejection, silently blaming the agent's
    submission for a host misconfiguration."""
    import arbench.benchmarks.ale_bench.benchmark as B
    _stage(tmp_path)
    # construct with a WORKING sandbox so __init__'s check passes...
    monkeypatch.setattr(B, "_sandbox_prefix", lambda *a, **k: [
        sys.executable, "-c", "pass"])
    bench = ALEBench(data_dir=str(tmp_path / "public"),
                     private_data_dir=str(tmp_path / "private"),
                     sandbox=True, workers=1)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(int(input()))\n")
    # ...then the sandbox becomes unavailable underneath it. This is exactly
    # the state the pre-review code raised on, so a reintroduced raise inside
    # grade() fails this test rather than slipping through.
    monkeypatch.setattr(B, "_sandbox_prefix", lambda *a, **k: [])
    score = bench.grade(task, sub)          # must NOT raise (Benchmark ABC)
    assert score.valid


def test_requested_sandbox_with_broken_userns_raises(tmp_path, monkeypatch):
    """bwrap PRESENT but unable to unshare (login nodes with
    max_user_namespaces=0) must fail loudly up front, not once per case.
    Presence is not capability."""
    import arbench.benchmarks.ale_bench.benchmark as B
    monkeypatch.setattr(B.shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(B, "_sandbox_prefix", lambda *a, **k: [
        sys.executable, "-c",
        "import sys; sys.stderr.write('bwrap: No permissions to creat"
        "e new namespace' + chr(10)); sys.exit(1)"])
    with pytest.raises(SandboxUnavailable) as e:
        ALEBench(data_dir=str(tmp_path / "public"),
                 private_data_dir=str(tmp_path / "private"), sandbox=True)
    assert "namespace" in str(e.value)


def test_sandbox_probe_accepts_a_working_bwrap(tmp_path, monkeypatch):
    """The unchanged path: a bwrap that CAN build a namespace is accepted and
    construction proceeds."""
    import arbench.benchmarks.ale_bench.benchmark as B
    monkeypatch.setattr(B, "_sandbox_prefix", lambda *a, **k: [
        sys.executable, "-c", "pass"])
    assert_sandbox_works()                                # does not raise
    bench = ALEBench(data_dir=str(tmp_path / "public"),
                     private_data_dir=str(tmp_path / "private"), sandbox=True)
    assert bench.sandbox is True


def test_autodetect_never_probes_the_sandbox(tmp_path, monkeypatch):
    """Auto-detect (sandbox=None) must not run the probe at all: it only ever
    turns the sandbox ON where bwrap is already present, and paying a
    subprocess on every construction would be a real cost on a bwrap-less
    host that never asked."""
    import arbench.benchmarks.ale_bench.benchmark as B
    calls = []
    monkeypatch.setattr(B, "assert_sandbox_works",
                        lambda: calls.append(1))
    ALEBench(data_dir=str(tmp_path / "public"),
             private_data_dir=str(tmp_path / "private"))
    ALEBench(data_dir=str(tmp_path / "public"),
             private_data_dir=str(tmp_path / "private"), sandbox=False)
    assert calls == []


def test_sandbox_autodetect_on_a_bwrapless_host_still_grades(tmp_path,
                                                             monkeypatch):
    """UNCHANGED DEFAULT. A caller that never asked for a sandbox (sandbox
    defaults to auto-detect) must keep grading unconfined on a bwrap-less
    host, exactly as before — the raise is scoped to an explicit request."""
    import arbench.benchmarks.ale_bench.benchmark as B
    _stage(tmp_path)
    monkeypatch.setattr(B.shutil, "which", lambda name: None)
    bench = ALEBench(data_dir=str(tmp_path / "public"),
                     private_data_dir=str(tmp_path / "private"), workers=1)
    assert bench.sandbox is False
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(int(input()))\n")
    score = bench.grade(task, sub)
    assert score.valid
    assert score.details["sandboxed"] is False


def test_requested_sandbox_that_is_available_grades_sandboxed(tmp_path,
                                                              monkeypatch):
    """The other unchanged path: when bwrap IS available, an explicit request
    is honoured and grading reports itself sandboxed."""
    import arbench.benchmarks.ale_bench.benchmark as B
    _stage(tmp_path)
    monkeypatch.setattr(B, "_sandbox_prefix", lambda *a, **k: ["env"])
    bench = ALEBench(data_dir=str(tmp_path / "public"),
                     private_data_dir=str(tmp_path / "private"),
                     sandbox=True, workers=1)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(int(input()))\n")
    score = bench.grade(task, sub)
    assert score.valid and score.details["sandboxed"] is True


def test_tester_killed_by_a_signal_is_refused(tmp_path):
    """A tester that dies by SIGNAL never reached its argument parser, so it
    cannot run here — even though it prints no ld.so text. SIGILL is the real
    case: a binary built with target-cpu=native (AVX-512) on a host that
    lacks those instructions. Without this the fail-silently defect returns
    through a different door."""
    pub, _ = _stage(tmp_path, problem_type="reactive")
    t = pub / "bin" / "tester"
    t.write_text(f"""#!{sys.executable}
import os, signal
os.kill(os.getpid(), signal.SIGILL)
""")
    t.chmod(t.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(ReactiveTesterUnavailable) as e:
        _bench(tmp_path).load_task(TASK)
    assert "signal" in str(e.value)


def test_tester_killed_by_sigsegv_is_refused(tmp_path):
    """Any startup abort counts, not just SIGILL."""
    pub, _ = _stage(tmp_path, problem_type="reactive")
    t = pub / "bin" / "tester"
    t.write_text(f"""#!{sys.executable}
import os, signal
os.kill(os.getpid(), signal.SIGSEGV)
""")
    t.chmod(t.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(ReactiveTesterUnavailable):
        _bench(tmp_path).load_task(TASK)


# ------------------------------------------------------------ public eval

def test_public_eval_scores_public_cases_with_diagnostics(tmp_path):
    """Batch: official vis over the PUBLIC cases; mean score; the scorer's
    own diagnostic line rides back in details['feedback']."""
    pub, priv = _stage(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    # echoes the case number for case 3, prints BAD (vis rejects) for case 4
    sub = tmp_path / "submission.py"
    sub.write_text("import sys\n"
                   "n = sys.stdin.readline().strip()\n"
                   "print('BAD' if n == '4' else n)\n")
    s = bench.public_eval(task, sub)
    assert s.valid and s.is_higher_better
    # public cases are 3 and 4: case 3 -> 3*2=6 ok; case 4 -> BAD rejected=0
    assert s.value == pytest.approx(6 / 2)
    assert s.details["public_eval"] is True
    assert s.details["n_rejected"] == 1
    assert [c["status"] for c in s.details["cases"]] == ["ok", "rejected"]
    assert "1 rejected" in s.details["feedback"]
    assert "invalid output" in s.details["feedback"]   # vis's own words
    assert "infra" not in s.details


def test_public_eval_minimize_with_failed_case_is_invalid(tmp_path):
    pub, priv = _stage(tmp_path, score_type="minimize")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("import sys\n"
                   "n = sys.stdin.readline().strip()\n"
                   "print('BAD' if n == '4' else n)\n")
    s = bench.public_eval(task, sub)
    assert not s.valid and not s.is_higher_better
    assert "minimize" in s.details["reason"]
    assert s.details["cases"]                     # per-case rows still there
    assert "infra" not in s.details


def test_public_eval_reactive_uses_the_public_tester(tmp_path):
    pub, priv = _stage(tmp_path, problem_type="reactive")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("import sys\n"
                   "n = sys.stdin.readline().strip()\n"
                   "print(n); sys.stdout.flush()\n")
    s = bench.public_eval(task, sub)
    assert s.valid
    # tester scores reply*10; public cases 3 and 4 -> (30+40)/2
    assert s.value == pytest.approx((30 + 40) / 2)
    assert s.details["n_cases"] == 2


def test_public_eval_missing_staging_is_marked_infra(tmp_path):
    pub, priv = _stage(tmp_path)
    (priv / "bin" / "vis").unlink()               # scorer gone: operator bug
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(1)\n")
    s = bench.public_eval(task, sub)
    assert not s.valid
    assert s.details.get("infra") is True


def test_public_eval_all_failed_is_invalid_not_zero(tmp_path):
    pub, priv = _stage(tmp_path)
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("raise RuntimeError('boom')\n")
    s = bench.public_eval(task, sub)
    assert not s.valid
    assert s.details["n_error"] == 2
    assert "boom" in s.details["feedback"]        # the crash line rides back
    assert "infra" not in s.details


def test_contract_text_promises_official_scoring(tmp_path):
    pub, priv = _stage(tmp_path)
    task = _bench(tmp_path).load_task(TASK)
    assert "OFFICIAL contest scorer" in task.goal
    assert "result.json" in task.goal             # told it is NOT needed
    assert "no result.json is needed" in task.goal


def test_public_eval_partial_public_staging_is_infra(tmp_path):
    """A missing public case shifts the mean's denominator: infra, never a
    score the loop can learn from."""
    pub, priv = _stage(tmp_path)
    (pub / "cases" / "case_0001.txt").unlink()
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(1)\n")
    s = bench.public_eval(task, sub)
    assert not s.valid
    assert s.details.get("infra") is True


def test_public_eval_scorer_outage_is_infra_not_a_zero(tmp_path):
    """The official scorer failing to RUN is a host fault; it must not be
    fed back to the agent as a rejected/zero case."""
    import stat as _stat
    pub, priv = _stage(tmp_path)
    vis = priv / "bin" / "vis"
    vis.chmod(_stat.S_IRUSR | _stat.S_IWUSR)     # not executable -> OSError
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("import sys; print(sys.stdin.readline().strip())\n")
    s = bench.public_eval(task, sub)
    assert not s.valid
    assert s.details.get("infra") is True
    assert "scorer failed" in s.details["reason"]


def test_public_eval_tle_case_scores_zero_in_a_maximize_mean(tmp_path):
    pub, priv = _stage(tmp_path, time_limit=0.4)      # 0.4 * 3.0 = 1.2s cap
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("import sys, time\n"
                   "n = sys.stdin.readline().strip()\n"
                   "if n == '3':\n"
                   "    time.sleep(5)\n"
                   "print(n)\n")
    s = bench.public_eval(task, sub)
    assert s.valid
    assert s.details["n_tle"] == 1
    assert s.value == pytest.approx((4 * 2) / 2)      # tle case counts as 0
    assert [c["status"] for c in s.details["cases"]] == ["tle", "ok"]
    assert "time limit" in s.details["feedback"]


# ---------------------------------------- sandbox path + infra fixes


def test_sandbox_prefix_masks_the_resolved_path(tmp_path):
    """Myriad's $HOME is a symlink into /myriadfs: bwrap cannot mkdir mount
    parents through a symlinked component, and a tmpfs at the symlink path
    would not mask the REAL path anyway (agent code resolving the link
    reads through it). The prefix must mount at the RESOLVED path — found
    live 2026-08-10, when every sandboxed reactive public-eval case died
    at setup and read as "rejected"."""
    from arbench.benchmarks.ale_bench.benchmark import _sandbox_prefix
    real = tmp_path / "real" / "prepared"
    (real / "bin").mkdir(parents=True)
    (real / "bin" / "tester").write_text("#!/bin/sh\n")
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "real")
    prefix = _sandbox_prefix(link / "prepared")
    if not prefix:                    # host without bwrap: prefix is []
        return
    joined = " ".join(prefix)
    assert str(real.resolve()) in joined
    assert str(link / "prepared") not in joined


def test_public_eval_reactive_is_unmasked(tmp_path, monkeypatch):
    """public_eval's tester lives in the PUBLIC staging — masking it is
    what broke every reactive public eval on Myriad. grade() keeps the
    mask (asserted via the signature default its call sites rely on)."""
    import inspect

    import arbench.benchmarks.ale_bench.benchmark as B
    calls = []

    def fake(tester, submission, case_file, timeout_s, sandbox, cwd=None,
             mask_private=True):
        calls.append(mask_private)
        return "ok", 10, b""

    monkeypatch.setattr(B, "_run_and_score_reactive", fake)
    _stage(tmp_path, problem_type="reactive")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(1)\n")
    s = bench.public_eval(task, sub)
    assert s.valid
    assert calls and all(m is False for m in calls)
    sig = inspect.signature(B._run_and_score_reactive)
    assert sig.parameters["mask_private"].default is True


def test_reactive_bwrap_setup_failure_is_infra(tmp_path, monkeypatch):
    """A sandbox that fails to build must surface as infra (loop kills the
    cell loudly), never as the agent's interaction being rejected — the
    misread that produced campaign-wide 0% reactive completion."""
    import arbench.benchmarks.ale_bench.benchmark as B

    def fake(tester, submission, case_file, timeout_s, sandbox, cwd=None,
             mask_private=True):
        return ("infra", None,
                b"bwrap: Can't mkdir parents for /x: No such file or directory")

    monkeypatch.setattr(B, "_run_and_score_reactive", fake)
    _stage(tmp_path, problem_type="reactive")
    bench = _bench(tmp_path)
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(1)\n")
    s = bench.public_eval(task, sub)
    assert not s.valid
    assert s.details.get("infra") is True
    assert "harness fault" in s.details["reason"]


def test_batch_bwrap_setup_failure_is_infra(tmp_path, monkeypatch):
    import arbench.benchmarks.ale_bench.benchmark as B

    def fake_run_case(submission, case_file, out_path, time_limit, sandbox):
        return ("error",
                b"bwrap: Can't mkdir parents for /x: No such file or directory")

    monkeypatch.setattr(B, "_run_case", fake_run_case)
    _stage(tmp_path)
    bench = _bench(tmp_path)
    bench.sandbox = True              # the infra check is sandbox-gated
    task = bench.load_task(TASK)
    sub = tmp_path / "submission.py"
    sub.write_text("print(1)\n")
    s = bench.public_eval(task, sub)
    assert not s.valid
    assert s.details.get("infra") is True
