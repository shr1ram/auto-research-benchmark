"""The benchmark contract: hands out Tasks and grades submissions.

A Benchmark is the source of truth for *what* the problem is and *how well* a
submission did. It never knows which autoresearch system produced the
submission — that asymmetry is what lets us compare systems fairly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from arbench.core.task import Task
from arbench.core.result import Score


class Benchmark(ABC):
    #: Short stable name, e.g. "mlebench_lite". Matches Task.benchmark.
    name: str = "unnamed"

    @abstractmethod
    def list_tasks(self) -> Iterable[str]:
        """Task ids this benchmark can serve (e.g. the competition ids)."""
        raise NotImplementedError

    @abstractmethod
    def load_task(self, task_id: str) -> Task:
        """Build the Task for `task_id`, preparing/locating its data. May be
        expensive (downloads, staging) the first time."""
        raise NotImplementedError

    @abstractmethod
    def grade(self, task: Task, submission_path: Path) -> Score:
        """Grade a produced submission. Must return Score.invalid(...) — never
        raise — when the submission is missing or malformed, so the harness can
        tell a crash apart from a genuinely bad score."""
        raise NotImplementedError

    def validate_submission(self, task: Task,
                            submission_path: Path) -> tuple[bool, str | None]:
        """Format-only verdict on a submission: (ok, reason-if-not).

        This is the ONLY grader output that may cross into a running agent
        loop: everything else grade() produces (value, medals, thresholds,
        percentiles) is heldout-score-derived and firewalled. The default
        grades and strips; benchmarks whose validity is checkable from PUBLIC
        data alone should override so the answers file isn't touched per
        attempt. Never raises; never returns score information."""
        try:
            score = self.grade(task, submission_path)
        except Exception as e:  # noqa: BLE001 — a validator must never raise
            return False, f"validation error: {e}"
        if score.valid:
            return True, None
        return False, score.details.get("reason", "submission invalid")
