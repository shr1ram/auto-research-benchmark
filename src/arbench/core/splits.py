"""Task splits: distance by DESIGN, never post hoc (benchmark plan §2).

Each benchmark plugin ships a splits.yaml assigning every task a modality and
a bin. Distance is benchmark-relational (2026-07-08 decision — no semantic
task families): in_domain_* bins live on the spine benchmark with a seeded
random source/eval split; near/far_* bins on the transfer benchmark are
defined by modality, a checkable fact.

The INDEX: `split_of(benchmark, task_id)` answers "what is this task?";
`split_index()` answers "which tasks are in bin X?" across all benchmarks.
Loaded tasks carry their entry in Task.metadata["split"], and the main repo
records it into every run manifest — so analysis can group by distance
without re-deriving anything.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any, Optional

import yaml

BINS = ("in_domain_source", "in_domain_eval", "control",
        "near", "far_vision", "far_text", "excluded")


@lru_cache(maxsize=None)
def load_splits(benchmark: str) -> dict[str, dict[str, Any]]:
    """The splits table for one benchmark plugin: task_id -> {modality, bin, ...}."""
    ref = resources.files(f"arbench.benchmarks.{benchmark}") / "splits.yaml"
    data = yaml.safe_load(ref.read_text())
    if data.get("benchmark") != benchmark:
        raise ValueError(f"splits.yaml under {benchmark} names "
                         f"benchmark {data.get('benchmark')!r}")
    table = data["splits"]
    for task_id, entry in table.items():
        if entry.get("bin") not in BINS:
            raise ValueError(f"{benchmark}/{task_id}: unknown bin {entry.get('bin')!r}")
    return table


def split_of(benchmark: str, task_id: str) -> Optional[dict[str, Any]]:
    """This task's split entry ({modality, bin, ...}), or None if unlisted."""
    return load_splits(benchmark).get(task_id)


def split_index(benchmarks: tuple[str, ...] = ("openml_tabular", "mlebench_lite"),
                ) -> dict[str, list[tuple[str, str]]]:
    """bin -> [(benchmark, task_id), ...] across benchmarks — the reverse index."""
    index: dict[str, list[tuple[str, str]]] = {b: [] for b in BINS}
    for benchmark in benchmarks:
        for task_id, entry in load_splits(benchmark).items():
            index[entry["bin"]].append((benchmark, task_id))
    return index


def tasks_in(bin_name: str, benchmark: Optional[str] = None) -> list[str]:
    """Task ids in one bin, optionally restricted to one benchmark."""
    if bin_name not in BINS:
        raise ValueError(f"unknown bin {bin_name!r}; have {list(BINS)}")
    return [tid for b, tid in split_index()[bin_name]
            if benchmark is None or b == benchmark]
