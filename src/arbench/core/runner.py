"""The orchestration loop that joins the two contracts:

    benchmark.load_task -> adapter.prepare -> adapter.run -> benchmark.grade

This is the only place that knows about both sides. Everything it touches is the
ABC surface, so any (adapter, benchmark) pair composes.

It also produces a full, gitignored run bundle for reproducibility: run.json
(metadata + token/cost/timing aggregates), llm_calls.jsonl (every LLM call),
the submission, and copied adapter artifacts (AIDE journal/best-solution/tree).
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
    "OPENAI_BASE_URL", "MLEBENCH_DATA_DIR", "AIDE_STEPS",
    "OPENAI_REQUEST_TIMEOUT", "OPENAI_MAX_RETRIES",
)


def _collect_artifacts(workspace: Path) -> None:
    """Copy the adapter's key reconstruction artifacts into <workspace>/artifacts.
    For AIDE: journal.json, best_solution.py, tree_plot.html, config.yaml."""
    art = workspace / "artifacts"
    art.mkdir(exist_ok=True)
    # AIDE writes logs under aide_run/logs/<node>/...
    for pat in ("aide_run/logs/**/journal.json",
                "aide_run/logs/**/best_solution.py",
                "aide_run/logs/**/tree_plot.html",
                "aide_run/logs/**/config.yaml"):
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

    # Point the LLM backend's call-trace sink at this run's JSONL (the AIDE fork
    # appends one record per call). Saved+restored so concurrent runs are isolated.
    prev_trace = os.environ.get(LLM_TRACE_ENV)
    if trace:
        os.environ[LLM_TRACE_ENV] = str(workspace / "llm_calls.jsonl")

    meta = RunMeta(
        adapter=adapter.name,
        benchmark=benchmark.name,
        task_id=task_id,
        model=getattr(adapter, "model", None) or os.environ.get("ARBENCH_LLM_MODEL", "unknown"),
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

    if adapter_error is not None:
        score = Score.invalid("adapter raised before producing a submission")
    elif not submission_path.exists():
        score = Score.invalid(f"no submission at {submission_path}")
    else:
        score = benchmark.grade(task, submission_path)

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
