"""Exit-point datatype: the Score a benchmark assigns to a submission."""
from __future__ import annotations

from dataclasses import dataclass, field
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
    def invalid(cls, reason: str, is_higher_better: bool = True,
                cases: Optional[list] = None) -> "Score":
        """`cases` carries per-case grading rows when the grader got far
        enough to produce them: an invalid aggregate can still be a fully
        informative per-case record (e.g. a minimize task refused because
        some cases were rejected)."""
        details: dict[str, Any] = {"reason": reason}
        if cases is not None:
            details["cases"] = cases
        return cls(value=None, valid=False, is_higher_better=is_higher_better,
                   details=details)
