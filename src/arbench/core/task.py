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
