"""The ENTRY contract.

An AutoResearchAdapter wraps one autoresearch system (AIDE, continual-auto-
research, ...) so the runner can drive it uniformly. The contract is tiny on
purpose: the harness owns the workspace and the task; the system owns *how* it
gets from goal -> submission.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from arbench.core.task import Task


class AutoResearchAdapter(ABC):
    #: Short stable name, e.g. "aide". Used in results and the registry.
    name: str = "unnamed"

    def prepare(self, task: Task, workspace: Path) -> None:
        """Optional one-time setup before a run (stage data into the workspace,
        write a config, materialise a problem statement). Default: no-op."""

    @abstractmethod
    def run(self, task: Task, workspace: Path) -> Path:
        """Drive the underlying system to solve `task`, working inside
        `workspace`. MUST write the submission to
        ``task.submission_path(workspace)`` and return that path.

        May raise — the runner catches it and records an adapter_error so a
        crashed system is graded as an invalid submission, not a silent zero.
        """
        raise NotImplementedError
