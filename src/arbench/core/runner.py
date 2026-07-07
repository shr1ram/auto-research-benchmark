"""The orchestration loop that joins the two contracts:

    benchmark.load_task -> adapter.prepare -> adapter.run -> benchmark.grade

This is the only place that knows about both sides. Everything it touches is the
ABC surface, so any (adapter, benchmark) pair composes.

It also produces a full, gitignored run bundle for reproducibility: run.json
(metadata + token/cost/timing aggregates), llm_calls.jsonl (every LLM call),
the submission, and copied adapter artifacts.
"""
from __future__ import annotations

import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Optional

from arbench.core.adapter import AutoResearchAdapter
from arbench.core.benchmark import Benchmark
from arbench.core.result import RunResult, Score
from arbench.core.trace import RunMeta, LLM_TRACE_ENV, build_trace

# repo root of arbench (for stamping its git SHA into the trace)
_ARBENCH_REPO = Path(__file__).resolve().parents[3]

# env keys worth snapshotting for reconstruction (no secrets)
_ENV_KEYS = (
    "ARBENCH_LLM_BACKEND", "ARBENCH_LLM_MODEL", "DEFAULT_API_BASE_URL",
    "OPENAI_BASE_URL", "MLEBENCH_DATA_DIR", "MLEBENCH_PRIVATE_DATA_DIR",
    "OPENML_DATA_DIR",
    "OPENAI_REQUEST_TIMEOUT", "OPENAI_MAX_RETRIES",
)

# Adapter output trees that must not survive into a re-run of the same workspace:
# adapters glob the newest submission.csv under
# these, so a stale tree lets a FAILED re-run be graded on the previous attempt's
# file.
_STALE_ADAPTER_DIRS = ("car_iters", "car_best")


def _clear_stale_attempt(workspace: Path, submission_path: Path) -> None:
    """Reset a reused workspace before a fresh attempt: drop the previous
    attempt's adapter output trees, submission, artifacts and LLM trace, so
    nothing from a prior run can leak into this run's grading or aggregates."""
    try:
        for name in _STALE_ADAPTER_DIRS:
            d = workspace / name
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        if submission_path.exists():
            submission_path.unlink()
        trace_path = workspace / "llm_calls.jsonl"
        if trace_path.exists():
            trace_path.unlink()
        art = workspace / "artifacts"
        if art.exists():
            for p in art.glob("*"):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink()
    except Exception:
        pass  # best-effort; a clear failure must not sink the run


def _collect_artifacts(workspace: Path) -> None:
    """Copy the adapter's key reconstruction artifacts into <workspace>/artifacts.
    Patterns are per-adapter; none registered currently (the retired AIDE globs
    lived here) — the new system's adapter adds its own."""
    art = workspace / "artifacts"
    art.mkdir(exist_ok=True)
    for pat in ():
        for src in workspace.glob(pat):
            # flatten name with the node dir to avoid collisions
            dest = art / f"{src.parent.name}__{src.name}"
            try:
                shutil.copyfile(src, dest)
            except Exception:
                pass


def run_one(
    adapter: AutoResearchAdapter,
    benchmark: Benchmark,
    task_id: str,
    workspace: Path,
    *,
    trace: bool = True,
) -> RunResult:
    """Run a single autoresearch system on a single task and grade it.

    Never raises for adapter/grader failure — failures become an invalid Score
    plus an adapter_error string, so a batch run can continue. When trace=True,
    writes a full reproducibility bundle into the workspace.
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    task = benchmark.load_task(task_id)
    submission_path = task.submission_path(workspace)

    # Reset any leftovers from a prior attempt into the SAME workspace (stale
    # adapter trees / submission / artifacts / llm trace) — else record_llm_call's
    # append mode double-counts and a failed re-run gets graded on the previous
    # attempt's submission.
    _clear_stale_attempt(workspace, submission_path)

    # Point the LLM backend's call-trace sink at this run's JSONL (the LLM client
    # appends one record per call). NOTE: $ARBENCH_LLM_TRACE is process-global —
    # run_one is one-run-per-process, NOT thread/async safe. The batch scheduler
    # runs each job in its own ssh'd process, so that's fine.
    prev_trace = os.environ.get(LLM_TRACE_ENV)
    if trace:
        os.environ[LLM_TRACE_ENV] = str(workspace / "llm_calls.jsonl")

    # Resolve the real model name via the backend resolver when the adapter
    # didn't pin one (adapter.model is often None — the model is the backend's
    # default, e.g. Kimi). Falls back to env, then "unknown". Used both for
    # meta.model AND as the per-call cost fallback, so it must be the real model.
    resolved_model = getattr(adapter, "model", None) or os.environ.get("ARBENCH_LLM_MODEL")
    if not resolved_model:
        try:
            from arbench.llm.backends import resolve_backend
            resolved_model = resolve_backend(getattr(adapter, "backend", "litellm")).model
        except Exception:
            resolved_model = "unknown"

    meta = RunMeta(
        adapter=adapter.name,
        benchmark=benchmark.name,
        task_id=task_id,
        model=resolved_model,
        backend=getattr(adapter, "backend", None) or os.environ.get("ARBENCH_LLM_BACKEND", "unknown"),
        steps=getattr(adapter, "steps", None),
        started_at=time.time(),
        env_snapshot={k: os.environ[k] for k in _ENV_KEYS if k in os.environ},
    )

    started = time.monotonic()
    adapter_error = None
    try:
        adapter.prepare(task, workspace)
        produced = adapter.run(task, workspace)
        if produced is not None:
            submission_path = Path(produced)
    except Exception:
        adapter_error = traceback.format_exc()
    duration_s = time.monotonic() - started

    meta.finished_at = time.time()
    meta.wall_clock_s = round(duration_s, 3)

    # Grade whatever submission exists, even when the adapter raised: a crash
    # AFTER producing a (best-so-far) submission — e.g. a transport failure in
    # a late iteration — must not throw away the gradeable work. The run is
    # invalid only if no submission was ever produced; adapter_error stays
    # recorded either way.
    if not submission_path.exists():
        score = Score.invalid(
            "adapter raised before producing a submission" if adapter_error is not None
            else f"no submission at {submission_path}")
    else:
        score = benchmark.grade(task, submission_path)
        if adapter_error is not None:
            score.details.setdefault(
                "note", "adapter raised mid-run; graded the best submission it "
                        "produced before failing (see adapter_error)")

    score_dict = {
        "value": score.value, "valid": score.valid,
        "is_higher_better": score.is_higher_better, "details": score.details,
    }

    if trace:
        try:
            _collect_artifacts(workspace)
            adapter_repo = getattr(adapter, "repo_dir", None)
            build_trace(
                meta=meta, score=score_dict, workspace=workspace,
                submission_path=submission_path if submission_path.exists() else None,
                adapter_error=adapter_error,
                arbench_repo=_ARBENCH_REPO, adapter_repo=adapter_repo,
            )
        except Exception:
            pass  # tracing failure must not sink the run
        finally:
            # restore env
            if prev_trace is None:
                os.environ.pop(LLM_TRACE_ENV, None)
            else:
                os.environ[LLM_TRACE_ENV] = prev_trace

    return RunResult(
        task_id=task_id,
        benchmark=benchmark.name,
        adapter=adapter.name,
        score=score,
        submission_path=submission_path if submission_path.exists() else None,
        workspace=workspace,
        duration_s=duration_s,
        adapter_error=adapter_error,
    )
