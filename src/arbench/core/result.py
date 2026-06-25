"""Exit-point datatypes: the Score a benchmark assigns, and the RunResult that
bundles everything one (adapter, task) run produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Score:
    """A graded outcome. `value` is the raw metric; `valid` is False when the
    submission was missing/malformed (so a crash is distinguishable from a real
    bad score). `is_higher_better` lets callers rank across tasks."""
    value: Optional[float]
    valid: bool
    is_higher_better: bool = True
    # Benchmark-specific extras: any_medal, gold_threshold, percentile, raw grader
    # report, etc.
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def invalid(cls, reason: str, is_higher_better: bool = True) -> "Score":
        return cls(value=None, valid=False, is_higher_better=is_higher_better,
                   details={"reason": reason})


@dataclass
class RunResult:
    """Everything a single (adapter, task) run produced — the object the runner
    returns and the CLI serialises."""
    task_id: str
    benchmark: str
    adapter: str
    score: Score
    # Path to the submission the adapter produced (may not exist if it crashed).
    submission_path: Optional[Path] = None
    # Where the adapter did its work — logs/artifacts live here for inspection.
    workspace: Optional[Path] = None
    # Wall-clock seconds the adapter ran.
    duration_s: Optional[float] = None
    # True if the adapter itself raised before producing anything.
    adapter_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "adapter": self.adapter,
            "score": {
                "value": self.score.value,
                "valid": self.score.valid,
                "is_higher_better": self.score.is_higher_better,
                "details": self.score.details,
            },
            "submission_path": str(self.submission_path) if self.submission_path else None,
            "workspace": str(self.workspace) if self.workspace else None,
            "duration_s": self.duration_s,
            "adapter_error": self.adapter_error,
        }
