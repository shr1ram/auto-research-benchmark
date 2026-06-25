"""Name -> factory lookup for adapters and benchmarks, so the CLI can wire
`--adapter aide --benchmark mlebench_lite` without importing everything eagerly
(MLE-bench / AIDE pull heavy deps only present on the run box).
"""
from __future__ import annotations

from typing import Callable

from arbench.core.adapter import AutoResearchAdapter
from arbench.core.benchmark import Benchmark

_ADAPTERS: dict[str, Callable[..., AutoResearchAdapter]] = {}
_BENCHMARKS: dict[str, Callable[..., Benchmark]] = {}


def register_adapter(name: str, factory: Callable[..., AutoResearchAdapter]) -> None:
    _ADAPTERS[name] = factory


def register_benchmark(name: str, factory: Callable[..., Benchmark]) -> None:
    _BENCHMARKS[name] = factory


def _load_builtins() -> None:
    """Import built-in adapters/benchmarks lazily; ignore ones whose optional
    deps are absent so the package still imports on a bare machine."""
    try:
        from arbench.adapters import aide_adapter  # noqa: F401  (self-registers)
    except Exception:
        pass
    try:
        from arbench.benchmarks.mlebench_lite import benchmark as _mb  # noqa: F401
    except Exception:
        pass


def get_adapter(name: str, **kwargs) -> AutoResearchAdapter:
    if name not in _ADAPTERS:
        _load_builtins()
    if name not in _ADAPTERS:
        raise KeyError(f"unknown adapter {name!r}; have {sorted(_ADAPTERS)}")
    return _ADAPTERS[name](**kwargs)


def get_benchmark(name: str, **kwargs) -> Benchmark:
    if name not in _BENCHMARKS:
        _load_builtins()
    if name not in _BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; have {sorted(_BENCHMARKS)}")
    return _BENCHMARKS[name](**kwargs)


def available() -> dict[str, list[str]]:
    _load_builtins()
    return {"adapters": sorted(_ADAPTERS), "benchmarks": sorted(_BENCHMARKS)}
