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
in Score.details. Cases are graded CONCURRENTLY (`grade_workers()`): each is
an independent (solver, scorer) pair over its own private tempdir, so the
only coupling is the box's cores — which is what bounds the worker count.

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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Optional

from arbench.core.benchmark import Benchmark
from arbench.core.data_version import verify_data_version
from arbench.core.result import Score
from arbench.core.task import Task

from arbench.benchmarks.ale_bench.tasks import ALL_PROBLEMS

#: AtCoder's submission size limit is 512 KiB; enforce the same
MAX_SUBMISSION_BYTES = 512 * 1024


class SandboxUnavailable(RuntimeError):
    """An explicitly requested grading sandbox cannot be provided on this
    host. Never downgrade silently: an operator who asked for confinement
    must learn it is absent BEFORE a grid starts, not from metadata they
    would have had to diff afterwards."""


class ReactiveTesterUnavailable(RuntimeError):
    """A reactive task's public `tester` cannot execute on this host.

    Reactive is the one contract that REQUIRES the agent to exec a shipped
    binary to score itself, so a tester that will not load turns an
    environment failure into agent-visible output the agent may catch and
    report as a number. Probing once, up front, is what keeps that from
    being mistaken for a measurement."""

#: grading time limit = official time_limit × this (Python does less work per
#: second than the C++ the contests assume; their judge scales per-language
#: too). One constant, identical across arms — internal-consistency knob,
#: never a comparability claim.
PYTHON_TIME_SCALE = 3.0

#: the official vis binary prints exactly this (stdout on batch problems)
_SCORE_RE = re.compile(r"Score\s*=\s*(-?\d+)")

#: ld.so's own diagnostics when a prebuilt binary cannot be loaded: a missing
#: symbol VERSION (the measured E2 case, GLIBC_2.34 on a 2.17 host), an absent
#: shared object, or an unresolved symbol. The loader emits these BEFORE the
#: program's first instruction and then exits nonzero, so a match proves the
#: tester never ran. Each alternative is a fixed ld.so phrase, not a generic
#: errno string — a tester that RUNS and prints "No such file or directory"
#: about its own arguments is a working tester and must not be flagged.
_LOADER_FAILURE_RE = re.compile(
    r"(version `[A-Z_]+_[\d.]+' not found"
    r"|error while loading shared libraries"
    r"|undefined symbol:"
    r"|cannot execute binary file)")

#: cap on a solver's stdout per case — AHC answers are KBs; a runaway
#: printer must become a rejected case, not grader OOM / a full disk
MAX_OUTPUT_BYTES = 16 * 1024 * 1024

#: hard ceiling on grading concurrency. Measured (rq1-ale-pilot-v3 A/B): at
#: P<=8 scores are bit-identical to sequential on the deterministic-solver
#: population (5.81x median speedup); at P>=16 deadline-ANNEALING solvers
#: degrade (-0.73% at P=16, -1.06% at P=24, against a 0.000% sequential noise
#: floor) because each solver process itself occupies ~1.3-1.5 cores. Raising
#: this re-enters measured degradation, so it is a ceiling, not a tuning knob.
MAX_GRADE_WORKERS = 8

#: cores one ALE solver process occupies (measured 1.3-1.5; 2 is the
#: conservative divisor). Worker count is a CORE SHARE divided by this.
CORES_PER_SOLVER = 2

#: how many cores this process may use. Grading runs inside a cell that is
#: itself one of `ops.driver.max_workers` concurrent cells, and the consuming
#: loop already publishes that cell's share here (arloop `derive_thread_cap`
#: exports nproc // max_workers over THREAD_CAP_KEYS before grading). Reading
#: the same variable makes grading concurrency and the agent thread cap share
#: ONE number, so P workers x max_workers cells cannot oversubscribe the box.
_CORE_SHARE_ENV = "OMP_NUM_THREADS"

#: explicit override, for a host that wants to state P directly (0/unset =
#: derive). Bounded by MAX_GRADE_WORKERS like every other path.
_WORKERS_ENV = "ARBENCH_ALE_GRADE_WORKERS"

#: bwrap's own setup failing is NOT a submission failure, but looks like one:
#: bwrap exits nonzero having produced no output, which `_run_case` records as
#: `error`, indistinguishable from a broken submission (worth a measured
#: -174,065 points on one run). Under concurrency (~1258 mounts per case,
#: bubblewrap 0.6.3) that setup races intermittently — 0.1-0.5% of cases,
#: 0/900 sequential occurrences — bursty and non-monotonic in P, so no safe-P
#: setting avoids it and the case is retried instead. Anchored to bwrap's own
#: `bwrap: ` line prefix so a submission cannot fake the signature: its
#: stderr is a separate stream from the sandbox's own diagnostics only insofar
#: as bwrap writes these lines before exec, which is where the race occurs.
_BWRAP_SETUP_RACE_RE = re.compile(
    rb"^bwrap: (Can't bind mount|Can't parse mountinfo|Unable to remount)",
    re.MULTILINE)

#: attempts per case when the mount race is detected (1 would disable retry).
#: The race is transient; after these the case records the error as it always
#: has, so a genuinely broken submission still fails — only slower.
_BWRAP_RETRY_ATTEMPTS = 3

#: pause before re-attempting a mount-raced case: the race is mount-table
#: contention in the kernel, not resource exhaustion, so it is short.
_BWRAP_RETRY_BACKOFF_S = 0.25

#: stderr kept per case, for the retry signature ONLY — never scored, never
#: reported. Bounded so a chatty solver cannot grow grader RAM per case.
_STDERR_KEEP_BYTES = 4096


def grade_workers(core_share: Optional[int] = None) -> int:
    """Cases to grade concurrently — DERIVED from this process's core share.

    A bare constant would be wrong: grading happens inside one of
    `ops.driver.max_workers` concurrent cells, so P processes x max_workers
    cells is the real box load. `_CORE_SHARE_ENV` is the share the consuming
    loop already computed for that cell, so the two caps stay consistent by
    construction.

    ASSUMES `_CORE_SHARE_ENV` states this process's core share. When it is
    unset nothing has claimed a share, so the whole box is assumed available
    — the regime the A/B measured the ceiling in, and the right reading for a
    manual one-off grade. A co-scheduled second grader with neither variable
    set would double-count; that is the same one-arm-per-box deployment
    assumption the thread cap makes.
    """
    override = os.environ.get(_WORKERS_ENV, "").strip()
    if override:
        try:
            n = int(override)
        except ValueError:
            n = 0
        if n > 0:
            return min(MAX_GRADE_WORKERS, n)
    if core_share is None:
        raw = os.environ.get(_CORE_SHARE_ENV, "").strip()
        try:
            core_share = int(raw) if raw else (os.cpu_count() or 1)
        except ValueError:
            core_share = os.cpu_count() or 1
    return max(1, min(MAX_GRADE_WORKERS, core_share // CORES_PER_SOLVER))

_CONTRACT = """
---
## Submission contract (this harness, not part of the contest statement)

- Write your final solver to `submission.py` in your working directory: a
  SINGLE-FILE Python 3 program that reads ONE input case from standard input
  and writes the answer to standard output ({time_note}).
- When your script finishes, the harness runs submission.py on EVERY public
  case (the files under {cases_dir}, named case_<i>.txt) and scores each
  output with the OFFICIAL contest scorer under the official time limit.
  Your attempt's score is the mean official case score. This problem
  {direction_note}; the harness reports your score so that higher is always
  better. Per-case failures (time limit exceeded, output rejected, crash)
  come back to you with the scorer's own diagnostics.
- You do NOT score yourself: no result.json is needed, and your own estimate
  of quality is never used. The official scorer enforces every validity rule
  in the statement exactly — an output that violates a constraint scores 0
  for that case no matter how good it looks otherwise.
- You may read the public cases (path above) to test your solver inside
  solution.py before it finishes; keep within the script time ceiling.
- The hidden test set uses the same case generator, so the official public
  score is your best available estimate of true quality.
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
- When your script finishes, the harness runs the OFFICIAL judge (`tester`)
  against submission.py on EVERY public case (the files under {cases_dir},
  named case_<i>.txt) under the official time limit. Your attempt's score is
  the mean of the tester's official scores. This problem {direction_note};
  the harness reports your score so that higher is always better. Per-case
  failures (time limit exceeded, rejected interaction, crash) come back to
  you with the tester's own diagnostics.
- You do NOT score yourself: no result.json is needed, and your own estimate
  of quality is never used.
- To test your solver yourself inside solution.py before it finishes, the
  same official tester is staged for you:
      {tester} {python} submission.py < <case-file>
  (it prints "Score = N" on its standard error).
- The hidden test set uses the same case generator, so the official public
  score is your best available estimate of true quality.
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


def assert_sandbox_works() -> None:
    """Fail-closed probe that a grading sandbox can actually be BUILT here.

    Presence is not capability: managed clusters ship the bwrap binary with
    unprivileged user namespaces disabled (measured: Myriad login nodes have
    max_user_namespaces=0, compute nodes 10000), where every confined case
    would fail one at a time instead of the request failing once, loudly.
    Same shape as arloop's `assert_bwrap_works()` — present AND able to
    unshare."""
    prefix = _sandbox_prefix()
    if not prefix:
        raise SandboxUnavailable(
            "grading sandbox requested (sandbox=True) but bwrap is not "
            "available on this host — install bubblewrap, or pass "
            "sandbox=False to grade agent code unconfined")
    try:
        probe = subprocess.run(prefix + ["/bin/true"], capture_output=True,
                               timeout=60)
    except OSError as e:
        raise SandboxUnavailable(
            f"grading sandbox requested but bwrap cannot be executed: "
            f"{e.__class__.__name__}: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise SandboxUnavailable(
            "grading sandbox requested but the bwrap probe hung for 60s") \
            from e
    if probe.returncode != 0:
        raise SandboxUnavailable(
            f"grading sandbox requested but bwrap cannot create namespaces "
            f"on this host (rc={probe.returncode}: "
            f"{probe.stderr.decode(errors='replace').strip()}) — unprivileged "
            f"user namespaces are often disabled on login nodes; pass "
            f"sandbox=False to grade agent code unconfined")


def assert_tester_executes(tester: Path) -> None:
    """Fail-closed probe that a reactive task's public tester actually RUNS
    on this host, run once per task before any attempt.

    The binary is prebuilt at prepare time, so a grid can be dispatched to a
    host whose loader cannot satisfy it (measured: E2 on Myriad, glibc 2.17
    against the testers' GLIBC_2.34). That failure surfaces only inside agent
    code, which is free to catch it — and did, converting a dead environment
    into `mean = 0.0` on 2,499 of 3,600 attempts that the harness then
    recorded as SUCCESSFUL. Biased data is worse than missing data, so the
    task refuses to load instead.

    Bare `tester` with no arguments prints a usage line and exits (measured
    rc=0 on the shipped binaries, but either way its own argument parser was
    reached, which proves the loader resolved every symbol). Only a binary
    that cannot load at all is fatal — this probe deliberately does NOT
    require a particular exit code or message from a tester that runs.
    Death by a SIGNAL is the exception: no usage-exit is ever a signal, so
    that is treated as fatal whatever the stderr says."""
    if not tester.is_file():
        raise ReactiveTesterUnavailable(f"reactive tester missing at {tester}")
    if not os.access(tester, os.X_OK):
        raise ReactiveTesterUnavailable(
            f"reactive tester at {tester} is not executable")
    try:
        probe = subprocess.run([str(tester)], capture_output=True, timeout=60)
    except OSError as e:
        raise ReactiveTesterUnavailable(
            f"reactive tester at {tester} cannot be executed on this host: "
            f"{e.__class__.__name__}: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise ReactiveTesterUnavailable(
            f"reactive tester at {tester} hung for 60s on a bare exec") from e
    # Death by SIGNAL is unconditionally fatal: a binary that reached its own
    # argument parser exits normally, so a signal means it never got there.
    # This catches the modes that carry no ld.so text at all — SIGILL from a
    # binary built for another microarchitecture (target-cpu=native, AVX-512
    # on a host without it), SIGSEGV/SIGABRT in startup — which would
    # otherwise reproduce this PR's own defect through a different door.
    if probe.returncode < 0:
        raise ReactiveTesterUnavailable(
            f"reactive tester at {tester} was killed by signal "
            f"{-probe.returncode} on a bare exec — it cannot start on this "
            f"host, so it would never run for the agent either")
    # A loader failure never reaches the program either: ld.so writes to
    # stderr and exits NONZERO without running main. Bare `tester` with no
    # arguments is itself allowed to exit nonzero (usage), so for a plain
    # nonzero exit the ld.so text is what distinguishes the two.
    if probe.returncode == 0:
        return
    err = (probe.stderr or b"").decode(errors="replace")
    if _LOADER_FAILURE_RE.search(err):
        raise ReactiveTesterUnavailable(
            f"reactive tester at {tester} cannot load on this host — the "
            f"dynamic loader rejected it, so it would never run for the "
            f"agent either:\n{err.strip()[:_STDERR_KEEP_BYTES]}")


def _run_case(submission: Path, case_file: Path, out_path: Path,
              timeout_s: float, sandbox: bool) -> tuple[str, bytes]:
    """One case: solver stdout streams to out_path ON DISK (never grader
    RAM — a runaway printer was an OOM, cubic P1) with a byte cap enforced
    after. Returns (ok|tle|error|too_long, stderr tail).

    cwd is the case's OWN tempdir: solvers grade concurrently, and a solver
    writing a fixed-name scratch file beside itself would otherwise collide
    with every other case (measured: 21/24 cases corrupted at P=8 under
    `sandbox: none`; the sandbox's read-only root masks it, it does not fix
    it).

    stderr is captured for ONE purpose — telling a bwrap setup failure apart
    from a solver crash (`_BWRAP_SETUP_RACE_RE`). It is never scored and never
    leaves the grader, and the tail is byte-capped."""
    cmd = (_sandbox_prefix() if sandbox else []) + [sys.executable,
                                                    str(submission)]
    with open(case_file, "rb") as stdin, open(out_path, "wb") as stdout:
        proc = subprocess.Popen(cmd, stdin=stdin, stdout=stdout,
                                stderr=subprocess.PIPE,
                                start_new_session=True,
                                cwd=str(out_path.parent))
        try:
            _, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.communicate()
            return "tle", b""
    err = (err or b"")[-_STDERR_KEEP_BYTES:]
    if proc.returncode != 0:
        return "error", err
    if out_path.stat().st_size > MAX_OUTPUT_BYTES:
        return "too_long", err
    return "ok", err


#: per-case outcomes recorded in Score.details["cases"]. "rejected" covers
#: every way a run produced no usable answer (invalid output, runaway stdout,
#: refused interaction) — on a RELATIVE contest that is a rejection, NOT a
#: zero score, and conflating the two inverts a minimize task's result.
_CASE_STATUSES = ("ok", "tle", "error", "rejected")


def _case_row(case_file: Path, status: str, score: Optional[int]) -> dict:
    """One row of Score.details["cases"]: the case's own name (the private
    seed order is meaningful), its outcome, and its official score when it
    earned one."""
    assert status in _CASE_STATUSES, f"unknown case status {status!r}"
    return {"case": case_file.name, "status": status,
            "score": score if status == "ok" else None}


def _score_case_msg(vis: Path, case_file: Path, out_path: Path
                    ) -> tuple[Optional[int], str, bool]:
    """Official scorer: `vis <input> <output>` prints 'Score = N'. Returns
    (score, diagnostic, scorer_failed): score is None if the scorer rejects
    the output or emits no score; diagnostic is the scorer's own first
    non-score line ("rectangles 2 and 7 overlap"), captured for public_eval
    feedback — the scorer only ever sees the case and the output, both
    public on the eval path, so the line is safe to forward; scorer_failed
    marks the scorer itself failing to RUN (timeout/launch error) — an
    infrastructure fact about the host, never a verdict on the output, and
    public_eval must not let it masquerade as a rejected answer. cwd is the
    output's private tempdir: vis writes vis.html into its cwd, and
    concurrent grid cells grading at once must not share scratch space."""
    try:
        proc = subprocess.run([str(vis), str(case_file), str(out_path)],
                              capture_output=True, text=True, timeout=60,
                              cwd=out_path.parent)
        text = proc.stdout + proc.stderr
        matches = _SCORE_RE.findall(text)
        msg = next((ln.strip() for ln in text.splitlines()
                    if ln.strip() and not _SCORE_RE.search(ln)), "")
        return (int(matches[-1]) if matches else None), msg[:200], False
    except (subprocess.TimeoutExpired, OSError) as e:
        return None, f"scorer failed to run: {e.__class__.__name__}", True


def _score_case(vis: Path, case_file: Path, out_path: Path) -> Optional[int]:
    return _score_case_msg(vis, case_file, out_path)[0]


def _run_and_score_reactive(tester: Path, submission: Path, case_file: Path,
                            timeout_s: float, sandbox: bool,
                            cwd: Optional[str] = None
                            ) -> tuple[str, Optional[int], bytes]:
    """Reactive case: the official `tester` reads the case on ITS stdin,
    spawns the solver, mediates the dialogue over pipes, and prints
    'Score = N' to ITS stderr — no separate scorer, no output file. Returns
    (status, score, stderr tail); score is None if the interaction failed
    (invalid move, TLE, tester rejected, or no score). The whole tester+solver
    tree runs in the sandbox as one process group.

    CONCURRENCY-SAFE: each case's tmpfs mask of the private root lives in that
    bwrap process's OWN mount namespace, so concurrent cases share no
    writable state; the score arrives on a private pipe, not a file. `cwd` is
    the case's own tempdir so a tester writing scratch beside itself cannot
    collide either (measured bit-identical at P=1..16).

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
                                stderr=subprocess.PIPE, start_new_session=True,
                                cwd=cwd)
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
            return "tle", None, b""
    tail = (err or b"")[-_STDERR_KEEP_BYTES:]
    if proc.returncode != 0:
        # the tester exits nonzero on a rejected interaction (invalid move,
        # solver crash) — don't trust a score line from a failed run
        return "rejected", None, tail
    matches = _SCORE_RE.findall(err[:MAX_OUTPUT_BYTES].decode(errors="replace"))
    if matches:
        return "ok", int(matches[-1]), tail
    return "rejected", None, tail   # successful exit but no score line


def _grade_one_case(index: int, case_file: Path, submission: Path,
                    scorer: Path, time_limit: float, sandboxed: bool,
                    reactive: bool) -> tuple[int, str, Optional[int], int]:
    """Grade ONE case end to end, independently of every other case.

    Returns (index, outcome, score, bwrap_retries). `index` is carried so the
    caller can restore case_files order regardless of completion order — the
    per-case rows are consumed positionally by official relative scoring.

    Outcome normalisation (too_long -> rejected, scoreless -> rejected) is
    done HERE so the aggregation loop only counts, and sequential and
    concurrent grading cannot drift apart on the classification.

    A case whose FAILURE is bwrap's own mount setup racing (not the
    submission) is retried: recording it as `error` would mark a good
    submission broken. Only that narrow signature retries, and a retry is
    counted so it is visible rather than silent."""
    retries = 0
    for attempt in range(_BWRAP_RETRY_ATTEMPTS):
        score: Optional[int] = None
        with tempfile.TemporaryDirectory(prefix="ale-grade-") as td:
            if reactive:
                outcome, score, err = _run_and_score_reactive(
                    scorer, submission, case_file, time_limit, sandboxed,
                    cwd=td)
            else:
                out_path = Path(td) / "case.out"
                outcome, err = _run_case(submission, case_file, out_path,
                                         time_limit, sandboxed)
                if outcome == "ok":
                    score = _score_case(scorer, case_file, out_path)
        raced = (outcome == "error" and sandboxed
                 and _BWRAP_SETUP_RACE_RE.search(err) is not None)
        if not raced or attempt == _BWRAP_RETRY_ATTEMPTS - 1:
            break
        retries += 1
        time.sleep(_BWRAP_RETRY_BACKOFF_S)
    if outcome == "too_long":
        outcome = "rejected"             # runaway printer: rejected case
    elif outcome not in ("tle", "error") and score is None:
        outcome = "rejected"             # invalid answer / no score line
    return index, outcome, score, retries


class ALEBench(Benchmark):
    name = "ale_bench"

    def __init__(self, data_dir: str | None = None,
                 private_data_dir: str | None = None,
                 sandbox: bool | None = None,
                 workers: int | None = None):
        # None = derive from this process's core share (grade_workers); an
        # explicit value is still bounded by MAX_GRADE_WORKERS, above which
        # deadline-annealing solvers measurably degrade.
        self.workers = (None if workers is None
                        else max(1, min(MAX_GRADE_WORKERS, workers)))
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
        # An EXPLICIT request is a promise this object cannot keep silently.
        # Verified HERE, at construction, because grade() must never raise
        # (Benchmark ABC) and because an operator who asked for confinement
        # must find out before a grid starts, not per-case mid-run. Auto-
        # detect never reaches this: it only turns the sandbox ON when bwrap
        # is already present, and a bwrap-less host resolves to False.
        if sandbox is True:
            assert_sandbox_works()

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

    # ------------------------------------------------------- public eval

    def public_eval(self, task: Task, submission_path: Path) -> Score:
        """Official scoring on the PUBLIC cases — the loop's score signal.

        Exists because the previous contract made the agent implement the
        statement's scoring rule itself, and the E2 held-out regrade priced
        that: 25% of proxy "improvements" graded to zero (validity rules the
        self-scorers never implement) and 23% of runs ended captured by a
        phantom self-score. ALE-Bench's own session API does exactly this
        (session.public_eval between attempts, private_eval once at the
        end); real AHC contestants likewise get official provisional
        results per submission, never their own estimate.

        Same machinery and posture as grade() — official scorer, official
        time limit x PYTHON_TIME_SCALE, sandboxed when available,
        concurrent per case — but over the PUBLIC cases, so per the ABC
        contract everything returned is built from agent-readable inputs
        plus the official tools and may cross into the loop: value (the
        MEAN official case score, raw contest sign; is_higher_better
        carries direction), details["cases"], and details["feedback"] (a
        counts line plus the scorer's own first diagnostics). A failed case
        is 0 in the mean on a maximise problem; on a minimise problem ANY
        failed case makes the eval invalid (a zero would read as an
        improvement) with the failure detail in the feedback.
        details["infra"] = True only for staging problems — never for
        anything the agent's code did."""
        meta = task.metadata
        hb = meta.get("score_type") == "maximize"
        sub = Path(submission_path)
        if not sub.is_file():
            return Score.invalid("submission.py missing or not a regular "
                                 "file", is_higher_better=hb)
        reactive = meta.get("problem_type") == "reactive"
        pub = Path(task.data_dir)
        case_files = sorted((pub / "cases").glob("case_*.txt")) \
            if (pub / "cases").is_dir() else []
        scorer = (pub / "bin" / "tester") if reactive \
            else self._private(task.task_id) / "bin" / "vis"
        expected = meta.get("n_public_cases")
        if not case_files or not scorer.is_file() or \
                (expected is not None and len(case_files) != expected):
            # a PARTIAL public staging would silently shift both the mean's
            # denominator and the score — same guard grade() applies to the
            # private tree
            s = Score.invalid(
                f"public staging incomplete for {task.task_id}: "
                f"{len(case_files)} cases (meta expects {expected}), "
                f"scorer {'present' if scorer.is_file() else 'MISSING'} — "
                f"run scripts/prepare_ale_bench.py on this host",
                is_higher_better=hb)
            s.details["infra"] = True
            return s
        time_limit = float(meta.get("time_limit_s") or 2.0) * PYTHON_TIME_SCALE
        sandboxed = self.sandbox
        workers = self.workers if self.workers is not None else grade_workers()

        def run(item):
            index, cf = item
            # same bwrap-setup-race retry grade()'s _grade_one_case applies:
            # a transient mount race is the sandbox misfiring, and without
            # the retry it would be fed back to the agent as a failed case
            for attempt in range(_BWRAP_RETRY_ATTEMPTS):
                msg, scorer_failed = "", False
                with tempfile.TemporaryDirectory(prefix="ale-pubeval-") as td:
                    if reactive:
                        outcome, score, err = _run_and_score_reactive(
                            scorer, sub, cf, time_limit, sandboxed, cwd=td)
                        if outcome != "ok" or not score:
                            msg = err.decode(errors="replace").strip()
                            msg = msg.splitlines()[-1][:200] if msg else ""
                    else:
                        out_path = Path(td) / "case.out"
                        outcome, err = _run_case(sub, cf, out_path,
                                                 time_limit, sandboxed)
                        score = None
                        if outcome == "ok":
                            score, msg, scorer_failed = _score_case_msg(
                                scorer, cf, out_path)
                        elif outcome == "error":
                            tail = err.decode(errors="replace").strip()
                            msg = tail.splitlines()[-1][:200] if tail else ""
                raced = (outcome == "error" and sandboxed
                         and _BWRAP_SETUP_RACE_RE.search(err) is not None)
                if not raced or attempt == _BWRAP_RETRY_ATTEMPTS - 1:
                    break
                time.sleep(_BWRAP_RETRY_BACKOFF_S)
            if outcome == "too_long":
                outcome = "rejected"
            elif outcome not in ("tle", "error") and score is None:
                outcome = "rejected"
            return index, outcome, score, msg, scorer_failed

        if workers == 1:
            results = [run(w) for w in enumerate(case_files)]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(run, enumerate(case_files)))
        results.sort(key=lambda r: r[0])

        total, n_tle, n_error, n_rejected = 0, 0, 0, 0
        cases, diags, scorer_failures = [], [], []
        for index, outcome, case_score, msg, scorer_failed in results:
            if scorer_failed:
                scorer_failures.append(f"{case_files[index].name}: {msg}")
            if outcome == "tle":
                n_tle += 1
            elif outcome == "error":
                n_error += 1
            elif outcome == "rejected":
                n_rejected += 1
            else:
                total += case_score
            cases.append(_case_row(case_files[index], outcome, case_score))
            if msg and len(diags) < 3:
                diags.append(f"{case_files[index].name}: {msg}")
        if scorer_failures:
            # the OFFICIAL SCORER failing to run is a host fault: feeding it
            # back as a rejected/zero case would hand the agent an outage
            # dressed up as a score
            s = Score.invalid(
                f"official scorer failed on {len(scorer_failures)} case(s): "
                + "; ".join(scorer_failures[:3]), is_higher_better=hb)
            s.details["infra"] = True
            return s
        n = len(case_files)
        n_failed = n_tle + n_error + n_rejected
        mean = total / n
        summary = (f"official public eval: {n - n_failed}/{n} cases scored"
                   + (f"; {n_tle} over the "
                      f"{time_limit:.0f}s time limit" if n_tle else "")
                   + (f"; {n_rejected} rejected" if n_rejected else "")
                   + (f"; {n_error} crashed" if n_error else "")
                   + (". " + " | ".join(diags) if diags else ""))
        details = {"n_cases": n, "n_tle": n_tle, "n_error": n_error,
                   "n_rejected": n_rejected, "time_limit_s": time_limit,
                   "python_time_scale": PYTHON_TIME_SCALE,
                   "sandboxed": sandboxed, "public_eval": True,
                   "feedback": summary, "cases": cases}
        if n_failed == n:
            s = Score.invalid(f"no case produced a scoreable output — "
                              f"{summary}", is_higher_better=hb)
            s.details.update(details)
            return s
        if n_failed and not hb:
            # a failed case contributing 0 would IMPROVE a minimize mean
            s = Score.invalid(f"{n_failed} failed case(s) on a minimize "
                              f"problem — {summary}", is_higher_better=hb)
            s.details.update(details)
            return s
        return Score(value=float(mean), valid=True, is_higher_better=hb,
                     details=details)

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
        if reactive:
            # staged is not the same as runnable: the binary is prebuilt and
            # the grid may run on a different host. Probe once, here, so an
            # unloadable tester kills the cell with its real cause instead of
            # reaching agent code that can catch it and report a number.
            assert_tester_executes(prep / "bin" / "tester")
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
            eval=(f"official AtCoder Heuristic Contest scorer; the harness "
                  f"scores every attempt on the public cases and the hidden "
                  f"cases use the same generator; absolute score, "
                  f"{score_type} "
                  f"({'higher' if score_type == 'maximize' else 'lower'} is "
                  f"better on the judge; scores reported to you are always "
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
        # Resolved ONCE, and already proven usable: an explicitly requested
        # sandbox is verified in __init__, so by here `self.sandbox` is the
        # truth rather than a hope. grade() must never raise (Benchmark ABC),
        # which is exactly why the check cannot live here.
        sandboxed = self.sandbox

        # Cases are independent (own tempdir, own cwd, own sandbox mount
        # namespace), so they grade CONCURRENTLY — 5.81x median on the pilot
        # grid. The worker count is the one thing that is not free: solvers
        # that anneal against a wall-clock deadline lose score when the box is
        # oversubscribed, which is what bounds `grade_workers()`.
        workers = self.workers if self.workers is not None else grade_workers()
        work = [(i, cf) for i, cf in enumerate(case_files)]

        def run(item: tuple[int, Path]):
            return _grade_one_case(item[0], item[1], sub, scorer, time_limit,
                                   sandboxed, reactive)

        if workers == 1:
            results = [run(w) for w in work]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(run, work))
        # completion order is arbitrary; the rows are consumed POSITIONALLY by
        # official relative scoring, so restore case_files order
        results.sort(key=lambda r: r[0])

        total, n_tle, n_error, n_rejected, n_bwrap_retries = 0, 0, 0, 0, 0
        # Per-case rows, in case_files order. The SUM alone is not enough to
        # reconstruct the official ALE-Bench metric: the relative contests
        # re-normalise per case, and there a REJECTED case is not a zero-cost
        # answer (see the `status` note on _CASE_STATUSES). Recovering this
        # after the fact means re-running agent code over the private cases,
        # so it is recorded once, here, where it is already known.
        cases: list[dict[str, Any]] = []
        for index, outcome, case_score, retries in results:
            n_bwrap_retries += retries
            if outcome == "tle":
                n_tle += 1
            elif outcome == "error":
                n_error += 1
            elif outcome == "rejected":
                n_rejected += 1
            else:
                total += case_score
            cases.append(_case_row(case_files[index], outcome, case_score))
        details = {"n_cases": len(case_files), "n_tle": n_tle,
                   "n_error": n_error, "n_rejected": n_rejected,
                   "seed_regime": meta.get("seed_regime"),
                   "time_limit_s": time_limit,
                   "python_time_scale": PYTHON_TIME_SCALE,
                   "sandboxed": sandboxed,
                   "sandbox_requested": self.sandbox,
                   "grade_workers": workers,
                   # bwrap's mount setup racing under concurrency, retried:
                   # nonzero means the sandbox misfired, never that the
                   # submission did
                   "n_bwrap_retries": n_bwrap_retries,
                   "problem_type": meta.get("problem_type", "batch"),
                   "score_type": meta.get("score_type")}
        n_failed = n_tle + n_error + n_rejected
        # the reason strings interpolate the AGGREGATE details only: `cases`
        # is per-case and would bloat a human-readable reason into KBs
        if n_failed == len(case_files):
            return Score.invalid("no case produced a scoreable output "
                                 f"({details})", is_higher_better=hb,
                                 cases=cases)
        if n_failed and not hb:
            # on a MINIMIZE problem a failed case contributing 0 would
            # IMPROVE the total — never report a misleading number
            return Score.invalid(
                f"{n_failed} failed case(s) on a minimize problem ({details})",
                is_higher_better=hb, cases=cases)
        return Score(value=float(total), valid=True, is_higher_better=hb,
                     details={**details, "cases": cases})
