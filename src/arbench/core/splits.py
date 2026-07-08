"""Task splits: automatic role assignment from human-defined fractions
(benchmark plan §2, 2026-07-08 design).

Each plugin's splits.yaml records ONLY checkable facts — every task's FAMILY
(benchmark × modality, the stratification unit) and exclusions with reasons.
Roles are computed, never stored:

    assign_roles({"bank": .4, "train": .2, "val": .2, "test": .2}, seed=0)
        -> {(benchmark, task_id): "bank" | "train" | "val" | "test"}

The draw is deterministic (seeded shuffle per family) and stratified: every
family is split at the same percentages, rounded by largest remainder. The
FRACTIONS are the single human knob and live in the experiment config (a
hashed confound) — with no fractions defined, no role exists and nothing can
run on a split.

Roles:
    bank   memory-free source runs; their traces build the banks
    train  development runs (iterate pipeline/prompts with memory)
    val    model selection (choose g*, retriever, k)
    test   final readout — touched once
"""
from __future__ import annotations

import random
from functools import lru_cache
from importlib import resources
from typing import Any, Optional

import yaml

ROLES = ("bank", "train", "val", "test")
BENCHMARKS = ("openml_tabular", "mlebench_lite")

TaskKey = tuple[str, str]   # (benchmark, task_id)


@lru_cache(maxsize=None)
def load_split_meta(benchmark: str) -> dict[str, dict[str, Any]]:
    """The factual table for one plugin: task_id -> {family, kind?, note?,
    excluded?}."""
    ref = resources.files(f"arbench.benchmarks.{benchmark}") / "splits.yaml"
    data = yaml.safe_load(ref.read_text())
    if data.get("benchmark") != benchmark:
        raise ValueError(f"splits.yaml under {benchmark} names "
                         f"benchmark {data.get('benchmark')!r}")
    for task_id, entry in data["tasks"].items():
        if not entry.get("family"):
            raise ValueError(f"{benchmark}/{task_id}: missing family")
    return data["tasks"]


def families(benchmarks: tuple[str, ...] = BENCHMARKS) -> dict[str, list[TaskKey]]:
    """family -> [(benchmark, task_id), ...], exclusions left out (with their
    reasons available via load_split_meta)."""
    out: dict[str, list[TaskKey]] = {}
    for benchmark in benchmarks:
        for task_id, entry in load_split_meta(benchmark).items():
            if entry.get("excluded"):
                continue
            out.setdefault(entry["family"], []).append((benchmark, task_id))
    return {f: sorted(members) for f, members in sorted(out.items())}


def _validate_fractions(fractions: dict[str, float]) -> None:
    if set(fractions) != set(ROLES):
        raise ValueError(f"fractions must define exactly {ROLES}, got "
                         f"{sorted(fractions)}")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {total}")
    if any(f < 0 for f in fractions.values()):
        raise ValueError("fractions must be non-negative")


def _allocate(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of n tasks across the roles."""
    quotas = {role: n * fractions[role] for role in ROLES}
    counts = {role: int(quotas[role]) for role in ROLES}
    for role in sorted(ROLES, key=lambda r: quotas[r] - counts[r], reverse=True):
        if sum(counts.values()) == n:
            break
        counts[role] += 1
    return counts


def assign_roles(fractions: dict[str, float], seed: int,
                 benchmarks: tuple[str, ...] = BENCHMARKS) -> dict[TaskKey, str]:
    """The automatic split: seeded shuffle within each family, sliced at the
    given percentages. Pure — same (fractions, seed) always gives the same
    assignment; record both in the experiment config/manifest."""
    _validate_fractions(fractions)
    assignment: dict[TaskKey, str] = {}
    for family, members in families(benchmarks).items():
        order = list(members)
        random.Random(f"{seed}:{family}").shuffle(order)
        counts = _allocate(len(order), fractions)
        i = 0
        for role in ROLES:
            for key in order[i:i + counts[role]]:
                assignment[key] = role
            i += counts[role]
    return assignment


def tasks_with_role(role: str, fractions: dict[str, float], seed: int,
                    benchmark: Optional[str] = None,
                    family: Optional[str] = None) -> list[str]:
    """Task ids holding `role` under (fractions, seed), optionally filtered."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; have {list(ROLES)}")
    meta = {b: load_split_meta(b) for b in BENCHMARKS}
    return sorted(
        task_id for (b, task_id), r in assign_roles(fractions, seed).items()
        if r == role
        and (benchmark is None or b == benchmark)
        and (family is None or meta[b][task_id]["family"] == family))


def split_of(benchmark: str, task_id: str) -> Optional[dict[str, Any]]:
    """The task's factual split metadata ({family, ...}), or None if unlisted."""
    return load_split_meta(benchmark).get(task_id)
