"""The orchestration loop that joins the two contracts:

    benchmark.load_task -> adapter.prepare -> adapter.run -> benchmark.grade

This is the only place that knows about both sides. Everything it touches is the
ABC surface, so any (adapter, benchmark) pair composes.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

from arbench.core.adapter import AutoResearchAdapter
from arbench.core.benchmark import Benchmark
from arbench.core.result import RunResult, Score


def run_one(
    adapter: AutoResearchAdapter,
    benchmark: Benchmark,
    task_id: str,
    workspace: Path,
) -> RunResult:
    """Run a single autoresearch system on a single task and grade it.

    Never raises for adapter/grader failure — failures become an invalid Score
    plus an adapter_error string, so a batch run can continue.
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    task = benchmark.load_task(task_id)
    submission_path = task.submission_path(workspace)

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

    if adapter_error is not None:
        score = Score.invalid("adapter raised before producing a submission")
    elif not submission_path.exists():
        score = Score.invalid(f"no submission at {submission_path}")
    else:
        score = benchmark.grade(task, submission_path)

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
