"""ALE-Bench plugin: AtCoder Heuristic Contest problems as arbench Tasks.

An alternative benchmark, selectable like any other: score-based
optimisation problems with no perfect solution — score-guided improvement IS
the task, which is exactly the shape our draft→debug→improve loop optimises.
Task distance is NOT baked in here: family `code` is a modality fact, and
near/far grouping is assigned at ANALYSIS time, relative to wherever the
banks were built (benchmark plan §2 — distance is benchmark-relational).
Full 40-problem set; `tasks.LITE_PROBLEMS` is the official cheap subset for
task selection (splits.yaml marks its members `subset: lite`).

Layout (public tree only — see FIREWALL below):

    $ALE_DATA_DIR/<problem_id>/prepared/
        problem.md        # official statement (EN) + our submission contract
        meta.json         # {problem_id, score_type, judge_version,
                          #  n_public_cases, public_seeds, seed_regime, ...}
        cases/<seed>.txt  # the staged PUBLIC input cases
        .data_version     # trust-on-first-use stamp (core/data_version.py)

Submission contract: `submission.py` — a single-file Python 3 program that
reads ONE input case from stdin and writes the answer to stdout. The agent's
solution.py writes it, runs it over cases/, self-scores per the statement's
scoring rule (that is what human contestants do — the score formula is part
of the statement), and reports the mean as its proxy score (negated for
minimise problems: higher is always better).

Grading: `session.private_eval` from the `ale-bench` package (Sakana AI) —
the OFFICIAL judge over hidden cases derived from private seeds. Seed regime
is a constructor knob: `full_seeds=False` (default) grades on the package's
lite seed set (10% of private seeds — tractable), `full_seeds=True` on the
full contest seed set (plus rank/performance). The regime is recorded in
Score.details; regimes are NOT comparable across runs. The dependency is
deliberately optional (`uv sync --extra ale`, pinned commit) and
Docker-backed: grading runs where Docker exists (dev machine), not on the
Singularity-only boxes; a missing judge grades Score.invalid, loudly.

FIREWALL: nothing private is ever staged — hidden cases exist only inside
ale-bench's own data cache (default ~/.cache/ale-bench, or $ALE_BENCH_DATA),
materialised grader-side at private_eval time. That cache directory must
NEVER be bind-allowlisted into an agent container. validate_submission is
pure format (exists / size / parses) and touches no ale-bench state.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Iterator

from arbench.core.benchmark import Benchmark
from arbench.core.data_version import verify_data_version
from arbench.core.result import Score
from arbench.core.task import Task

from arbench.benchmarks.ale_bench.tasks import ALL_PROBLEMS

#: AtCoder's submission size limit is 512 KiB; enforce the same
MAX_SUBMISSION_BYTES = 512 * 1024

_CONTRACT = """
---
## Submission contract (this harness, not part of the contest statement)

- Write your final solver to `submission.py` in your working directory: a
  SINGLE-FILE Python 3 program that reads ONE input case from standard input
  and writes the answer to standard output ({time_note}).
- The public input cases are the files under: {cases_dir}
  Run your solver on every one of them, compute your score for each case
  using the scoring rule defined in the statement above, and write the MEAN
  into result.json. This problem {direction_note} — report a higher-is-better
  number (negate the mean if the problem minimises).
- The official judge runs `submission.py` on HIDDEN cases drawn from the
  same generator; your reported score is your own estimate of it.
"""


class ALEBench(Benchmark):
    name = "ale_bench"

    def __init__(self, data_dir: str | None = None,
                 private_data_dir: str | None = None,
                 session_factory=None, full_seeds: bool = False):
        root = data_dir or os.environ.get("ALE_DATA_DIR", "")
        self.data_dir = Path(root) if root else None
        # accepted for get_benchmark(**kwargs) parity, but there IS no private
        # tree: hidden cases live only in ale-bench's grader-side cache
        if private_data_dir:
            raise ValueError(
                "ale_bench has no private data tree — hidden cases live in "
                "ale-bench's own cache; do not pass private_data_dir")
        self.full_seeds = full_seeds
        # tests inject a fake; the default lazily imports ale-bench at grade
        self._session_factory = session_factory

    # ------------------------------------------------------------- tasks

    def list_tasks(self) -> Iterator[str]:
        yield from ALL_PROBLEMS

    def _prepared(self, task_id: str) -> Path:
        if not self.data_dir:
            raise RuntimeError("ALE_DATA_DIR not set (point it at the prepared "
                               "ale_bench public root).")
        return self.data_dir / task_id / "prepared"

    def load_task(self, task_id: str) -> Task:
        if task_id not in ALL_PROBLEMS:
            raise RuntimeError(f"unknown ale_bench task {task_id!r}; "
                               f"have {sorted(ALL_PROBLEMS)}")
        prep = self._prepared(task_id)
        if not (prep / "meta.json").exists():
            raise RuntimeError(
                f"task {task_id!r} is not prepared at {prep}.\n"
                f"Prepare it (needs the `ale` extra + Docker):  "
                f"python scripts/prepare_ale_bench.py {task_id}")
        meta = json.loads((prep / "meta.json").read_text())
        data_version = verify_data_version(prep)   # loud on drift (plan §2)
        score_type = meta["score_type"]                 # minimize | maximize
        time_note = (f"time limit {meta['time_limit_s']}s per case"
                     if meta.get("time_limit_s") else "per-case time limit in "
                     "the statement")
        goal = (prep / "problem.md").read_text()
        goal += _CONTRACT.format(
            cases_dir=prep / "cases",
            time_note=time_note,
            direction_note=f"{score_type.upper()}S its score")
        return Task(
            task_id=task_id, benchmark=self.name, goal=goal,
            eval=(f"official AtCoder Heuristic Contest judge on hidden cases; "
                  f"absolute score, {score_type} "
                  f"({'higher' if score_type == 'maximize' else 'lower'} is "
                  f"better on the judge; your result.json must always report "
                  f"higher-is-better)"),
            data_dir=prep,
            submission_filename="submission.py",
            metadata={"score_type": score_type,
                      "judge_version": meta.get("judge_version"),
                      "n_public_cases": meta.get("n_public_cases"),
                      "staged_seed_regime": meta.get("seed_regime"),
                      "data_version": data_version,
                      "split": meta.get("split")},
        )

    # ---------------------------------------------------------- verdicts

    def validate_submission(self, task: Task, submission_path: Path):
        """FORMAT-only (the sole grader output allowed into a running loop):
        a regular, non-empty, size-capped file that parses as Python. Never
        touches ale-bench or anything private."""
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

    def _session(self, task_id: str):
        if self._session_factory is not None:
            return self._session_factory(task_id)
        import ale_bench   # the `ale` extra; needs a Docker host
        return ale_bench.start(problem_id=task_id,
                               lite_version=not self.full_seeds,
                               run_visualization_server=False)

    def grade(self, task: Task, submission_path: Path) -> Score:
        meta = task.metadata
        hb = meta.get("score_type") == "maximize"
        try:
            code = Path(submission_path).read_text(errors="replace")
        except OSError as e:   # missing, unreadable, or a directory
            return Score.invalid(f"submission.py unreadable: "
                                 f"{e.__class__.__name__}", is_higher_better=hb)
        try:
            session = self._session(task.task_id)
        except Exception as e:  # noqa: BLE001 — missing extra / no Docker
            return Score.invalid(
                f"ale-bench judge unavailable ({type(e).__name__}: {e}); "
                f"grade on a Docker host with the `ale` extra installed",
                is_higher_better=hb)
        try:
            result, rank, performance = session.private_eval(code, "python")
            value = float(result.overall_absolute_score)
            details = {"judge_result": str(result.overall_judge_result),
                       "n_cases": len(result.case_results),
                       "rank": rank, "performance": performance,
                       "seed_regime": "full" if self.full_seeds else "lite",
                       "score_type": meta.get("score_type")}
        except Exception as e:  # noqa: BLE001 — judge is external
            return Score.invalid(f"private_eval failed: {type(e).__name__}: {e}",
                                 is_higher_better=hb)
        finally:
            try:
                session.close()
            except Exception:  # noqa: BLE001,S110 — best-effort cleanup
                pass
        if not (value == value and abs(value) != float("inf")):  # non-finite
            return Score.invalid(f"judge returned non-finite score {value!r}",
                                 is_higher_better=hb)
        return Score(value=value, valid=True, is_higher_better=hb,
                     details=details)
