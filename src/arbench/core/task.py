"""The Task: the unit of work an autoresearch system is asked to solve.

A Task is deliberately benchmark-agnostic. MLE-Bench produces Tasks; so could
SWE-bench, a custom Kaggle-style problem, or a toy regression. The adapter only
ever sees this — never the benchmark internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Task:
    # Stable identifier, unique within a benchmark (e.g. the competition id).
    task_id: str

    # Which benchmark minted this task (e.g. "mlebench_lite"). Used by the runner
    # to route grading back to the right Benchmark.
    benchmark: str

    # Natural-language statement handed to the autoresearch system. For MLE-bench
    # this is the competition description + the data dictionary.
    goal: str

    # How success is measured, in prose (e.g. "AUC, higher is better"). The
    # machine-readable grading lives in the Benchmark; this is for the model.
    eval: str

    # Absolute path to the prepared input data the system may read (read-only).
    data_dir: Optional[Path] = None

    # The exact artifact the system must produce, relative to its workspace
    # (e.g. "submission.csv"). The benchmark grader looks here.
    submission_filename: str = "submission.csv"

    # Free-form, benchmark-specific extras (kaggle competition id, sample
    # submission path, grade direction, etc.). Adapters should ignore unknown keys.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data_dir is not None and not isinstance(self.data_dir, Path):
            self.data_dir = Path(self.data_dir)

    def submission_path(self, workspace: Path) -> Path:
        """Where the grader expects this task's submission inside a workspace."""
        return Path(workspace) / self.submission_filename

    def render_goal(self, visible_data_dir: str | Path) -> str:
        """The goal text as the AGENT should see it.

        `goal` references the data by its HOST path (self.data_dir). When the
        run executes somewhere that path does not exist — e.g. inside a
        Singularity container with the data bind-mounted at /data — the adapter
        must hand the agent a goal whose paths match its actual view. This
        rewrites every occurrence of the data_dir prefix, which also covers
        paths UNDER it (e.g. the sample-submission line). Adapters that pass
        metadata paths (sample_submission etc.) to the agent must translate
        those the same way.
        """
        if self.data_dir is None:
            return self.goal
        return self.goal.replace(str(self.data_dir), str(visible_data_dir))
