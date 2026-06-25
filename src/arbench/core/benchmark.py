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
