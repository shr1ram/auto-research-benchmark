"""auto-research-benchmark — a thin orchestration layer that runs *arbitrary*
autoresearch systems against *arbitrary* benchmarks behind two stable contracts:

    ENTRY  — AutoResearchAdapter: given a Task + workspace, produce a submission.
    EXIT   — Benchmark.grade(): given a submission, produce a Score.

Any system that can read a task description and emit a submission artifact (
the continual-auto-research system, a human, a shell script) plugs in by
implementing AutoResearchAdapter. Any benchmark that can hand out tasks and grade
submissions (MLE-Bench Lite first) plugs in by implementing Benchmark.
"""
from arbench.core.task import Task
from arbench.core.result import RunResult, Score
from arbench.core.adapter import AutoResearchAdapter
from arbench.core.benchmark import Benchmark

__all__ = ["Task", "RunResult", "Score", "AutoResearchAdapter", "Benchmark"]
__version__ = "0.1.0"
