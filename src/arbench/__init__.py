"""auto-research-benchmark — benchmark tasks + held-out grading, as a LIBRARY.

Two contracts:

    ENTRY  — Benchmark.load_task(id) -> Task: what the problem is (the agent-
             visible view only; answers live in a separate private tree).
    EXIT   — Benchmark.grade(task, submission) -> Score: how well a submission
             did, ignorant of which system produced it.

Nothing here drives an agent. The consuming system (the thesis repo's loop)
lists tasks, loads them, runs itself, and calls grade at finalise — that
asymmetry is what keeps grading fair and the firewall mount-level.
"""
from arbench.core.task import Task
from arbench.core.result import Score
from arbench.core.benchmark import Benchmark

BENCHMARK_NAMES = ("mlebench_lite", "openml_tabular")


def get_benchmark(name: str, **kwargs) -> Benchmark:
    """Construct a benchmark plugin by name.

    Imports are lazy so the package works on a bare machine: mlebench_lite
    needs the `mlebench` extra (uv sync --extra mlebench) only when actually
    used. kwargs are the plugin's (data_dir, private_data_dir); both default
    to their env vars ($MLEBENCH_DATA_DIR / $OPENML_DATA_DIR etc.).
    """
    if name == "mlebench_lite":
        from arbench.benchmarks.mlebench_lite.benchmark import MLEBenchLite
        return MLEBenchLite(**kwargs)
    if name == "openml_tabular":
        from arbench.benchmarks.openml_tabular.benchmark import OpenMLTabular
        return OpenMLTabular(**kwargs)
    raise KeyError(f"unknown benchmark {name!r}; have {list(BENCHMARK_NAMES)}")


__all__ = ["Task", "Score", "Benchmark", "get_benchmark", "BENCHMARK_NAMES"]
__version__ = "0.2.0"
