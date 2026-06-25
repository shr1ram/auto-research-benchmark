"""MLE-Bench Lite benchmark, wrapping OpenAI's `mlebench` package.

We reuse mlebench's registry (task metadata + prepared data layout) and its
grader (`grade_csv`) rather than reimplementing either. The only thing we add is
the adapter-facing Task view and a clean Score.

`mlebench` and a prepared competition must be present on the run box; if they are
not, load_task/grade raise a clear, actionable error instead of failing deep in
an import.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from arbench.core.benchmark import Benchmark
from arbench.core.task import Task
from arbench.core.result import Score
from arbench.core.registry import register_benchmark
from arbench.benchmarks.mlebench_lite.tasks import LITE_COMPETITIONS, SMALL_FIRST


def _require_mlebench():
    try:
        from mlebench.registry import registry  # noqa
        from mlebench.grade import grade_csv  # noqa
        return registry, grade_csv
    except Exception as e:  # pragma: no cover - depends on box install
        raise RuntimeError(
            "mlebench is not importable. Install it on the run box:\n"
            "  uv pip install -e /path/to/mle-bench\n"
            f"(underlying import error: {e})"
        ) from e


class MLEBenchLite(Benchmark):
    name = "mlebench_lite"

    def __init__(self, data_dir: str | None = None):
        # Where prepared competition data lives. Default to MLEBENCH_DATA_DIR env
        # (we point this at the project FS on the box, never NFS home).
        self.data_dir = Path(
            data_dir or os.environ.get("MLEBENCH_DATA_DIR", "")
        ) if (data_dir or os.environ.get("MLEBENCH_DATA_DIR")) else None

    def _registry(self):
        registry, _ = _require_mlebench()
        return registry.set_data_dir(self.data_dir) if self.data_dir else registry

    def list_tasks(self) -> Iterable[str]:
        # The Lite split, smallest-first so the cheap smoke tasks lead.
        seen, ordered = set(), []
        for cid in SMALL_FIRST + LITE_COMPETITIONS:
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)
        return ordered

    def load_task(self, task_id: str) -> Task:
        comp = self._registry().get_competition(task_id)
        public_dir = Path(comp.public_dir)
        if not public_dir.exists():
            raise RuntimeError(
                f"competition {task_id!r} is not prepared at {public_dir}.\n"
                f"Prepare it first:  mlebench prepare -c {task_id}\n"
                "(requires Kaggle API creds in ~/.kaggle/kaggle.json and "
                "accepting the competition rules on kaggle.com)."
            )
        goal = (
            f"{comp.description}\n\n"
            f"Read the training data under: {public_dir}\n"
            f"Produce a submission CSV matching the format of: "
            f"{comp.sample_submission}"
        )
        return Task(
            task_id=task_id,
            benchmark=self.name,
            goal=goal,
            eval=f"Graded by the Kaggle '{task_id}' metric. "
                 f"{'Lower is better.' if getattr(comp.grader, 'is_lower_better', False) else 'Higher is better.'}",
            data_dir=public_dir,
            submission_filename="submission.csv",
            metadata={
                "competition_id": task_id,
                "sample_submission": str(comp.sample_submission),
                "is_lower_better": bool(getattr(comp.grader, "is_lower_better", False)),
            },
        )

    def grade(self, task: Task, submission_path: Path) -> Score:
        registry, grade_csv = _require_mlebench()
        reg = self._registry()
        try:
            comp = reg.get_competition(task.task_id)
            report = grade_csv(Path(submission_path), comp)
        except Exception as e:  # grading must never raise out of the harness
            return Score.invalid(f"grader error: {e}",
                                 is_higher_better=not task.metadata.get("is_lower_better", False))

        is_higher_better = not bool(getattr(report, "is_lower_better", False))
        if not getattr(report, "valid_submission", False):
            return Score.invalid("mlebench marked submission invalid",
                                 is_higher_better=is_higher_better)
        return Score(
            value=float(report.score) if report.score is not None else None,
            valid=True,
            is_higher_better=is_higher_better,
            details={
                "any_medal": bool(getattr(report, "any_medal", False)),
                "gold_medal": bool(getattr(report, "gold_medal", False)),
                "silver_medal": bool(getattr(report, "silver_medal", False)),
                "bronze_medal": bool(getattr(report, "bronze_medal", False)),
                "above_median": bool(getattr(report, "above_median", False)),
                "gold_threshold": getattr(report, "gold_threshold", None),
                "median_threshold": getattr(report, "median_threshold", None),
            },
        )


register_benchmark("mlebench_lite", MLEBenchLite)
