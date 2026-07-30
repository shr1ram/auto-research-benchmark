"""ALE-Bench plugin: AtCoder Heuristic Contest problems as arbench Tasks.

An alternative benchmark, selectable like any other: score-based optimisation
problems with no perfect solution — score-guided improvement IS the task.
Task distance is NOT baked in here: family `code` is a modality fact; near/far
grouping is assigned at ANALYSIS time relative to wherever the banks were
built (benchmark plan §2). Full 40-problem set; `tasks.LITE_PROBLEMS` is the
official cheap subset (splits.yaml marks its members `subset: lite`).

SELF-CONTAINED grading (owner call 2026-07-13): the dataset zips ship the
official Rust tools (case generator `gen`, scorer `vis`, reactive `tester`)
plus every seed list — so prepare AND grade run entirely on the lab boxes,
no Docker, no Mac, no `ale-bench` package. Internal consistency is the goal,
not parity with Sakana's judge environment (their own results inherit host
CPUs too; see design-decisions "ale_bench standalone decisions"). Their
package remains only as an optional one-off cross-check
(`uv sync --extra ale-crosscheck`).

Layout:

    $ALE_DATA_DIR/<problem_id>/prepared/          # PUBLIC (agent-bindable)
        problem.md        # official statement (EN) + our submission contract
        meta.json         # score_type, problem_type, time_limit_s,
                          # n_public/private_cases, public_seeds, seed_regime
        cases/case_<i>.txt   # official public inputs (their public seed list)
        .data_version     # trust-on-first-use stamp
    $ALE_PRIVATE_DATA_DIR/<problem_id>/           # PRIVATE (grader-only)
        cases/case_<i>.txt   # official private inputs (their private seeds)
        bin/vis (+ bin/tester, bin/gen)   # official tools, compiled at prep
        seeds.json           # the private seed list (never in public meta)

Submission contract: `submission.py` — a single-file Python 3 program that
reads ONE input case from stdin and writes the answer to stdout. The agent's
solution.py writes it, runs it over the public cases/, self-scores per the
statement's scoring rule (what human contestants do), and reports the mean
(negated for minimise problems: higher is always better) as its proxy.

Grading, BATCH = run `submission.py` over the PRIVATE cases under the official time
limit × PYTHON_TIME_SCALE (one constant, identical across arms — their judge
also scales time per language), score each (input, output) with the official
`vis` binary, SUM the case scores (their overall_absolute_score semantics).
The submission is agent code: it executes inside a bwrap sandbox when
available (same posture as attempt execution), plain process-group subprocess
otherwise. TLE / nonzero exit / unparsable score = 0 for that case, counted
in Score.details.

FIREWALL: the dataset zips contain the private seed lists (data.json), so
raw zips and everything derived from private seeds live ONLY under the
private root; public meta.json carries public seeds and counts, never the
private list. validate_submission is pure format and touches neither tree.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional

from arbench.core.benchmark import Benchmark
from arbench.core.data_version import verify_data_version
from arbench.core.result import Score
from arbench.core.task import Task

from arbench.benchmarks.ale_bench.tasks import ALL_PROBLEMS

#: AtCoder's submission size limit is 512 KiB; enforce the same
MAX_SUBMISSION_BYTES = 512 * 1024

#: grading time limit = official time_limit × this (Python does less work per
#: second than the C++ the contests assume; their judge scales per-language
#: too). One constant, identical across arms — internal-consistency knob,
#: never a comparability claim.
PYTHON_TIME_SCALE = 3.0

#: the official vis binary prints exactly this (stdout on batch problems)
_SCORE_RE = re.compile(r"Score\s*=\s*(-?\d+)")

#: cap on a solver's stdout per case — AHC answers are KBs; a runaway
#: printer must become a rejected case, not grader OOM / a full disk
MAX_OUTPUT_BYTES = 16 * 1024 * 1024

_CONTRACT = """
---
## Submission contract (this harness, not part of the contest statement)

- Write your final solver to `submission.py` in your working directory: a
  SINGLE-FILE Python 3 program that reads ONE input case from standard input
  and writes the answer to standard output ({time_note}).
- The public input cases are the files under: {cases_dir}
  (named case_<i>.txt — read EVERY file in that directory; do not
  assume any other naming pattern).
  Run your solver on every one of them, compute your score for each case
  using the scoring rule defined in the statement above, and write the MEAN
  into result.json. This problem {direction_note} — report a higher-is-better
  number (negate the mean if the problem minimises).
- The hidden test set uses the same case generator; your reported score is
  your own estimate of your solver's true quality.
- Your script MUST END by writing result.json containing a NUMERIC "score".
  A run that exits 0 without a numeric score in result.json is INVALID and
  scores nothing, however good the solver was: make sure the scoring block is
  always reached (not left unreachable after an early return/exit), and that
  the value is a real number — NaN and inf are NOT numeric, so guard
  degenerate cases before writing.
"""

_CONTRACT_REACTIVE = """
---
## Submission contract (this harness, not part of the contest statement)

- This is an INTERACTIVE (reactive) problem: your solver holds a turn-by-turn
  dialogue with a judge program over standard input/output, exactly as the
  Interaction section of the statement above describes. Follow that protocol
  — it is authoritative: read a line, decide your move, print it, FLUSH
  standard output, and repeat until the interaction ends. A solver that does
  not flush after each message will hang and time out.
- Write your final solver to `submission.py`: a SINGLE-FILE Python 3 program
  implementing that protocol ({time_note}).
- The public cases + the official judge (`tester`) are staged for you. Score
  your solver on each public case by running:
      {tester} {python} submission.py < <case-file>
  (the case files are under {cases_dir}); the tester prints "Score = N" on
  its standard error. Write the MEAN score across the public cases into
  result.json. This problem {direction_note} — report a higher-is-better
  number (negate the mean if the problem minimises).
- The hidden test set uses the same generator; your reported score is your
  own estimate of your solver's true quality.
- Your script MUST END by writing result.json containing a NUMERIC "score".
  A run that exits 0 without a numeric score in result.json is INVALID and
  scores nothing, however good the solver was: make sure the scoring block is
  always reached (not left unreachable after an early return/exit), and that
  the value is a real number — NaN and inf are NOT numeric, so guard
  degenerate cases before writing.
"""


def _sandbox_prefix(private_root: Optional[Path] = None) -> list[str]:
    """bwrap when available: read-only root, no network, private /tmp, and a
    PRIVATE PID namespace with a fresh /proc — the submission is agent code
    and grading must not extend its reach.

    When `private_root` is given (reactive grading), a tmpfs is mounted over
    it so the solver cannot read the hidden cases/seeds (cubic P0), and the
    tester binary under it is re-bound read-only so the tester still runs.
    The private PID namespace (--unshare-pid + fresh /proc) is what stops a
    reactive solver forging a score via /proc/<tester>/fd/2: the tester is
    not visible in the solver's PID namespace (cubic P0, verified)."""
    if shutil.which("bwrap") is None:
        return []
    cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--tmpfs", "/tmp",
           "--unshare-net", "--unshare-pid", "--proc", "/proc",
           "--die-with-parent"]
    if private_root is not None:
        pr = str(private_root)
        cmd += ["--tmpfs", pr]
        tester = private_root / "bin" / "tester"
        if tester.is_file():
            cmd += ["--ro-bind", str(tester), str(tester)]
    return cmd


def _run_case(submission: Path, case_file: Path, out_path: Path,
              timeout_s: float, sandbox: bool) -> str:
    """One case: solver stdout streams to out_path ON DISK (never grader
    RAM — a runaway printer was an OOM, cubic P1) with a byte cap enforced
    after. Returns ok|tle|error|too_long."""
    cmd = (_sandbox_prefix() if sandbox else []) + [sys.executable,
                                                    str(submission)]
    with open(case_file, "rb") as stdin, open(out_path, "wb") as stdout:
        proc = subprocess.Popen(cmd, stdin=stdin, stdout=stdout,
                                stderr=subprocess.DEVNULL,
                                start_new_session=True)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
            return "tle"
    if proc.returncode != 0:
        return "error"
    if out_path.stat().st_size > MAX_OUTPUT_BYTES:
        return "too_long"
    return "ok"


def _score_case(vis: Path, case_file: Path, out_path: Path) -> Optional[int]:
    """Official scorer: `vis <input> <output>` prints 'Score = N'. None if
    the scorer rejects the output (invalid answer) or emits no score. cwd is
    the output's private tempdir: vis writes vis.html into its cwd, and
    concurrent grid cells grading at once must not share scratch space."""
    try:
        proc = subprocess.run([str(vis), str(case_file), str(out_path)],
                              capture_output=True, text=True, timeout=60,
                              cwd=out_path.parent)
        matches = _SCORE_RE.findall(proc.stdout + proc.stderr)
        return int(matches[-1]) if matches else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def _run_and_score_reactive(tester: Path, submission: Path, case_file: Path,
                            timeout_s: float, sandbox: bool
                            ) -> tuple[str, Optional[int]]:
    """Reactive case: the official `tester` reads the case on ITS stdin,
    spawns the solver, mediates the dialogue over pipes, and prints
    'Score = N' to ITS stderr — no separate scorer, no output file. Returns
    the score, or None if the interaction failed (invalid move, TLE, tester
    rejected, or no score). The whole tester+solver tree runs in the sandbox
    as one process group.

    SECURITY (cubic P0, both verified exploitable then fixed): a reactive
    solver shares the tester's stderr fd and its filesystem/proc view, so it
    could (a) read the hidden private cases and (b) forge a score by writing
    'Score = 999...' to the tester's stderr via /proc/<tester>/fd/2. The
    sandbox closes both: a tmpfs masks the private root (cases hidden), and a
    PRIVATE PID NAMESPACE (--unshare-pid + fresh /proc) means the tester is
    not visible in the solver's proc, so its fds are unreachable. The solver's
    own stderr is additionally /dev/null'd (noise, not trusted). Only a
    SUCCESSFUL tester exit whose stderr carries 'Score = N' counts."""
    # the tester's private root is masked in the sandbox (hide hidden cases)
    # and re-binds the tester binary; the PID namespace stops proc-fd forgery.
    private_root = tester.parent.parent
    solver = (f"exec {shlex.quote(sys.executable)} "
              f"{shlex.quote(str(submission))} 2>/dev/null")
    inner = [str(tester), "/bin/sh", "-c", solver]
    cmd = (_sandbox_prefix(private_root) if sandbox else []) + inner
    with open(case_file, "rb") as stdin:
        proc = subprocess.Popen(cmd, stdin=stdin, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, start_new_session=True)
        try:
            # communicate() is the timeout-safe reader (a raw read() would
            # ignore the timeout and hang on a deadlocked solver). After the
            # sandbox fix the ONLY writer to this stderr is the trusted
            # official tester (the solver's stderr is /dev/null'd and it can no
            # longer reach the tester's fd), which emits a single score line —
            # so it is bounded in practice; sliced to the cap defensively.
            _, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.communicate()
            return "tle", None
    if proc.returncode != 0:
        # the tester exits nonzero on a rejected interaction (invalid move,
        # solver crash) — don't trust a score line from a failed run
        return "rejected", None
    matches = _SCORE_RE.findall(err[:MAX_OUTPUT_BYTES].decode(errors="replace"))
    if matches:
        return "ok", int(matches[-1])
    return "rejected", None    # successful exit but no score line


class ALEBench(Benchmark):
    name = "ale_bench"

    def __init__(self, data_dir: str | None = None,
                 private_data_dir: str | None = None,
                 sandbox: bool | None = None):
        root = data_dir or os.environ.get("ALE_DATA_DIR", "")
        self.data_dir = Path(root) if root else None
        priv = private_data_dir or os.environ.get("ALE_PRIVATE_DATA_DIR", "")
        if priv:
            self.private_dir = Path(priv)
        else:
            # same convention as openml: sibling `private/` of the public root
            self.private_dir = (self.data_dir.parent / "private"
                                if self.data_dir else None)
        # None = auto (bwrap if present); tests pass False for portability
        self.sandbox = shutil.which("bwrap") is not None if sandbox is None \
            else sandbox

    # ------------------------------------------------------------- tasks

    def list_tasks(self) -> Iterator[str]:
        yield from ALL_PROBLEMS

    def _prepared(self, task_id: str) -> Path:
        if not self.data_dir:
            raise RuntimeError("ALE_DATA_DIR not set (point it at the prepared "
                               "ale_bench public root).")
        return self.data_dir / task_id / "prepared"

    def _private(self, task_id: str) -> Path:
        if not self.private_dir:
            raise RuntimeError("no private root: set ALE_PRIVATE_DATA_DIR or "
                               "ALE_DATA_DIR (private/ sibling convention)")
        return self.private_dir / task_id

    def load_task(self, task_id: str) -> Task:
        if task_id not in ALL_PROBLEMS:
            raise RuntimeError(f"unknown ale_bench task {task_id!r}; "
                               f"have {sorted(ALL_PROBLEMS)}")
        prep = self._prepared(task_id)
        if not (prep / "meta.json").exists():
            raise RuntimeError(
                f"task {task_id!r} is not prepared at {prep}.\n"
                f"Prepare it (needs cargo — `rustup` is a user-level "
                f"install):  python scripts/prepare_ale_bench.py {task_id}")
        meta = json.loads((prep / "meta.json").read_text())
        data_version = verify_data_version(prep)   # loud on drift (plan §2)
        problem_type = meta.get("problem_type", "batch")
        reactive = problem_type == "reactive"
        if reactive and not (prep / "bin" / "tester").is_file():
            # a reactive contract needs the public tester for the agent's
            # self-eval; an old batch-only staging lacks it (re-prepare)
            raise RuntimeError(
                f"task {task_id!r} is reactive but its public tester is not "
                f"staged at {prep / 'bin' / 'tester'} — re-run "
                f"scripts/prepare_ale_bench.py {task_id}")
        score_type = meta["score_type"]                 # minimize | maximize
        time_note = (f"time limit {meta['time_limit_s']}s per case"
                     if meta.get("time_limit_s") else "per-case time limit in "
                     "the statement")
        goal = (prep / "problem.md").read_text()
        if reactive:
            goal += _CONTRACT_REACTIVE.format(
                cases_dir=prep / "cases", tester=prep / "bin" / "tester",
                python=sys.executable, time_note=time_note,
                direction_note=f"{score_type.upper()}S its score")
        else:
            goal += _CONTRACT.format(
                cases_dir=prep / "cases", time_note=time_note,
                direction_note=f"{score_type.upper()}S its score")
        return Task(
            task_id=task_id, benchmark=self.name, goal=goal,
            eval=(f"official AtCoder Heuristic Contest scorer over hidden "
                  f"cases; absolute score, {score_type} "
                  f"({'higher' if score_type == 'maximize' else 'lower'} is "
                  f"better on the judge; your result.json must always report "
                  f"higher-is-better)"),
            data_dir=prep,
            submission_filename="submission.py",
            metadata={"score_type": score_type,
                      "problem_type": meta.get("problem_type", "batch"),
                      "time_limit_s": meta.get("time_limit_s"),
                      "n_public_cases": meta.get("n_public_cases"),
                      "n_private_cases": meta.get("n_private_cases"),
                      "seed_regime": meta.get("seed_regime"),
                      "data_version": data_version,
                      "split": meta.get("split")},
        )

    # ---------------------------------------------------------- verdicts

    def validate_submission(self, task: Task, submission_path: Path):
        """FORMAT-only (the sole grader output allowed into a running loop):
        a regular, non-empty, size-capped file that parses as Python. Touches
        neither data tree."""
        p = Path(submission_path)
        if not p.is_file():
            return False, "submission.py is missing or not a regular file"
        size = p.stat().st_size
        if size == 0:
            return False, "submission.py is empty"
        if size > MAX_SUBMISSION_BYTES:
            return False, (f"submission.py is {size} bytes; the judge caps "
                           f"submissions at {MAX_SUBMISSION_BYTES}")
        try:
            code = p.read_text(errors="replace")
        except OSError as e:
            return False, f"submission.py is unreadable: {e.__class__.__name__}"
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"submission.py is not valid Python: {e.msg} " \
                          f"(line {e.lineno})"
        return True, None

    # ------------------------------------------------------------ grading

    def grade(self, task: Task, submission_path: Path) -> Score:
        meta = task.metadata
        hb = meta.get("score_type") == "maximize"
        sub = Path(submission_path)
        if not sub.is_file():
            return Score.invalid("submission.py missing or not a regular file",
                                 is_higher_better=hb)
        reactive = meta.get("problem_type") == "reactive"
        priv = self._private(task.task_id)
        scorer = priv / "bin" / ("tester" if reactive else "vis")
        case_files = sorted((priv / "cases").glob("case_*.txt")) \
            if (priv / "cases").is_dir() else []
        if not case_files or not scorer.is_file():
            return Score.invalid(
                f"private cases/scorer not staged for {task.task_id} — run "
                f"scripts/prepare_ale_bench.py on the grading host",
                is_higher_better=hb)
        expected = meta.get("n_private_cases")
        if expected is not None and len(case_files) != expected:
            # a partially-staged private tree would score on FEWER cases
            # silently (a truncated grade reads as a genuine low score)
            return Score.invalid(
                f"private tree has {len(case_files)} cases, meta expects "
                f"{expected} — re-stage {task.task_id}", is_higher_better=hb)
        try:
            # the private cases + scorer are the grade inputs; drift here
            # shifts scores as surely as public drift shifts the task
            verify_data_version(priv)
        except RuntimeError as e:
            return Score.invalid(f"private-tree drift: {e}",
                                 is_higher_better=hb)
        time_limit = float(meta.get("time_limit_s") or 2.0) * PYTHON_TIME_SCALE
        # resolved ONCE: requesting a sandbox on a bwrap-less host must be
        # loud in details, never a silent degrade
        sandboxed = self.sandbox and bool(_sandbox_prefix())

        total, n_tle, n_error, n_rejected = 0, 0, 0, 0
        for case_file in case_files:
            if reactive:
                # the tester runs+scores in one call; TLE vs rejected
                # (invalid move / no score) are distinguished for the audit
                outcome, case_score = _run_and_score_reactive(
                    scorer, sub, case_file, time_limit, sandboxed)
                if outcome == "tle":
                    n_tle += 1
                    continue
                if case_score is None:
                    n_rejected += 1
                    continue
                total += case_score
                continue
            with tempfile.TemporaryDirectory(prefix="ale-grade-") as td:
                out_path = Path(td) / "case.out"
                outcome = _run_case(sub, case_file, out_path, time_limit,
                                    sandboxed)
                if outcome == "tle":
                    n_tle += 1
                    continue
                if outcome == "error":
                    n_error += 1
                    continue
                if outcome == "too_long":
                    n_rejected += 1          # runaway printer: rejected case
                    continue
                case_score = _score_case(scorer, case_file, out_path)
            if case_score is None:
                n_rejected += 1              # invalid answer: scores 0
                continue
            total += case_score
        details = {"n_cases": len(case_files), "n_tle": n_tle,
                   "n_error": n_error, "n_rejected": n_rejected,
                   "seed_regime": meta.get("seed_regime"),
                   "time_limit_s": time_limit,
                   "python_time_scale": PYTHON_TIME_SCALE,
                   "sandboxed": sandboxed,
                   "sandbox_requested": self.sandbox,
                   "problem_type": meta.get("problem_type", "batch"),
                   "score_type": meta.get("score_type")}
        n_failed = n_tle + n_error + n_rejected
        if n_failed == len(case_files):
            return Score.invalid("no case produced a scoreable output "
                                 f"({details})", is_higher_better=hb)
        if n_failed and not hb:
            # on a MINIMIZE problem a failed case contributing 0 would
            # IMPROVE the total — never report a misleading number
            return Score.invalid(
                f"{n_failed} failed case(s) on a minimize problem ({details})",
                is_higher_better=hb)
        return Score(value=float(total), valid=True, is_higher_better=hb,
                     details=details)
