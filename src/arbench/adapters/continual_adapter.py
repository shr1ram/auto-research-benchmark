"""Adapter stub for the continual-auto-research system ("car").

This is the second autoresearch system the benchmark targets. The contract is
identical to the AIDE adapter: given a Task + workspace, drive the system and
leave a submission.csv where the harness expects it.

The integration point depends on how `car` is invoked (Python API vs CLI vs the
running service on the box). Left as a documented stub with the wiring sketch so
it can be filled in once the car entry point is settled — the benchmark side is
already complete and identical for both systems.
"""
from __future__ import annotations

from pathlib import Path

from arbench.core.adapter import AutoResearchAdapter
from arbench.core.task import Task
from arbench.core.registry import register_adapter


class ContinualAdapter(AutoResearchAdapter):
    name = "continual"

    def __init__(self, steps: int = 10, backend: str | None = None, model: str | None = None):
        self.steps = steps
        self.backend = backend
        self.model = model

    def run(self, task: Task, workspace: Path) -> Path:
        raise NotImplementedError(
            "ContinualAdapter is a stub. Wire it to the car system's entry point "
            "(Python API / CLI / running service) and copy its produced "
            "submission.csv to task.submission_path(workspace). The benchmark + "
            "runner contracts are already satisfied; only this run() body is TODO."
        )


register_adapter("continual", ContinualAdapter)
