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

def test_reactive_problems_are_excluded_from_the_draw():
    """The 12 reactive problems can't be served by the batch contract, so
    they are excluded -> assign_roles draws only over the 28 batch problems."""
    from arbench.core.splits import families, assign_roles
    fams = families(("ale_bench",))
    assert len(fams["code"]) == 28              # 40 - 12 reactive
    roles = assign_roles({"bank": 0.55, "val": 0.30, "test": 0.15}, seed=0)
    drawn = {t for (b, t), r in roles.items() if b == "ale_bench"}
    assert len(drawn) == 28


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
