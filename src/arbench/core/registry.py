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


# import failures captured here (surfaced via available()), so a real bug in an
# adapter/benchmark module isn't silently swallowed as "optional dep missing".
_LOAD_ERRORS: dict[str, str] = {}


def _load_builtins() -> None:
    """Import built-in adapters/benchmarks lazily. Tolerate a MISSING optional
    dependency (ImportError) so the package imports on a bare machine, but let
    any OTHER error (SyntaxError, a bug in the module) propagate — and record the
    ImportError messages so they're visible rather than silently lost."""
    for label, importer in (
        ("aide", lambda: __import__("arbench.adapters.aide_adapter", fromlist=["*"])),
        ("continual", lambda: __import__("arbench.adapters.continual_adapter", fromlist=["*"])),
        ("mlebench_lite", lambda: __import__("arbench.benchmarks.mlebench_lite.benchmark", fromlist=["*"])),
        ("openml_tabular", lambda: __import__("arbench.benchmarks.openml_tabular.benchmark", fromlist=["*"])),
    ):
        try:
            importer()
        except ImportError as e:
            _LOAD_ERRORS[label] = str(e)  # optional dep absent — recorded, not hidden
        # any non-ImportError (real bug) intentionally propagates


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
    out = {"adapters": sorted(_ADAPTERS), "benchmarks": sorted(_BENCHMARKS)}
    if _LOAD_ERRORS:
        out["unavailable"] = dict(_LOAD_ERRORS)  # optional deps that failed to import
    return out
