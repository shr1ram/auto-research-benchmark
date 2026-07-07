"""Expand a sweep spec (tasks × seeds × arms) into a flat list of Jobs.

A Job is one `arbench run` invocation: a (task, seed, arm) triple with its own
output dir. Seeds are independent repeats (LLM stochasticity gives the
variance); arms are e.g. "baseline" vs "continual" once that adapter exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    task_id: str
    seed: int
    arm: str           # "baseline" | "continual" | ... (maps to an adapter/config)
    adapter: str
    benchmark: str
    out_dir: Path      # runs/<sweep>/<arm>/<task>-seed<seed>/
    steps: int
    backend: str
    model: str | None

    @property
    def name(self) -> str:
        return f"{self.arm}/{self.task_id}-seed{self.seed}"

    @property
    def done_marker(self) -> Path:
        # a job is complete iff its run.json exists (written at end of run_one)
        return self.out_dir / "run.json"

    def is_done(self) -> bool:
        return self.done_marker.exists()


def expand_worklist(
    *,
    tasks: list[str],
    seeds: int,
    arms: list[str],
    adapter: str,
    benchmark: str,
    sweep_dir: Path,
    steps: int,
    backend: str,
    model: str | None,
) -> list[Job]:
    sweep_dir = Path(sweep_dir)
    jobs: list[Job] = []
    for arm in arms:
        for task in tasks:
            for seed in range(seeds):
                out = sweep_dir / arm / f"{task}-seed{seed}"
                jobs.append(Job(
                    task_id=task, seed=seed, arm=arm,
                    adapter=adapter, benchmark=benchmark,
                    out_dir=out, steps=steps, backend=backend, model=model,
                ))
    return jobs
