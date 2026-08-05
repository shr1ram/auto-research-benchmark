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
    #: Short stable name, e.g. "ale_bench". Matches Task.benchmark.
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

    #: the default validator's only failure reason — grade()'s own reason
    #: strings and exceptions may embed private detail (answers paths,
    #: heldout-derived text) and are NEVER forwarded to the agent loop.
    GENERIC_INVALID = ("submission failed the benchmark's validity check; "
                       "match the format the task describes (see "
                       "sample_submission if provided)")

    def validate_submission(self, task: Task,
                            submission_path: Path) -> tuple[bool, str | None]:
        """Format-only verdict on a submission: (ok, reason-if-not).

        This is the ONLY grader output that may cross into a running agent
        loop: everything else grade() produces (value, medals, thresholds,
        percentiles) is heldout-score-derived and firewalled. The default
        grades and DISCARDS EVERYTHING but the boolean — including grade()'s
        reason strings and exception text, which it cannot prove free of
        private detail (an answers-file path in an IOError would hand the
        agent the private tree). Benchmarks wanting actionable reasons must
        override with a check built ONLY from public data (see openml).
        Never raises; never returns score information."""
        try:
            score = self.grade(task, submission_path)
        except Exception:  # noqa: BLE001 — a validator must never raise
            return False, self.GENERIC_INVALID
        if score.valid:
            return True, None
        return False, self.GENERIC_INVALID
