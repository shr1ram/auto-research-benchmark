"""arbench CLI — run an autoresearch system against a benchmark task.

    arbench list
    arbench run --adapter aide --benchmark mlebench_lite \
        --task random-acts-of-pizza --steps 10 --backend litellm \
        --workspace /cs/.../runs/raop --data-dir /cs/.../mlebench-data
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from arbench.core import registry
from arbench.core.runner import run_one


@click.group()
def main() -> None:
    """auto-research-benchmark CLI."""


@main.command("list")
def list_cmd() -> None:
    """List registered adapters and benchmarks."""
    avail = registry.available()
    click.echo("adapters:   " + ", ".join(avail["adapters"]) or "(none)")
    click.echo("benchmarks: " + ", ".join(avail["benchmarks"]) or "(none)")


@main.command("run")
@click.option("--adapter", required=True, help="autoresearch system, e.g. aide")
@click.option("--benchmark", required=True, help="benchmark, e.g. mlebench_lite")
@click.option("--task", "task_id", required=True, help="task/competition id")
@click.option("--workspace", required=True, type=click.Path(),
              help="dir for this run's artifacts (use the project FS on the box)")
@click.option("--data-dir", default=None,
              help="benchmark data dir (else MLEBENCH_DATA_DIR env)")
@click.option("--backend", default="litellm",
              help="LLM backend: litellm | claude_p | local")
@click.option("--model", default=None, help="override the backend's default model")
@click.option("--steps", default=10, type=int, help="autoresearch iterations")
@click.option("--out", default=None, type=click.Path(),
              help="write the RunResult JSON here (else stdout)")
def run_cmd(adapter, benchmark, task_id, workspace, data_dir, backend, model, steps, out):
    """Run one (adapter, task) and grade it."""
    bench_kwargs = {"data_dir": data_dir} if data_dir else {}
    bench = registry.get_benchmark(benchmark, **bench_kwargs)
    adapt = registry.get_adapter(adapter, steps=steps, backend=backend, model=model)

    result = run_one(adapt, bench, task_id, Path(workspace))
    payload = json.dumps(result.to_dict(), indent=2)
    if out:
        Path(out).write_text(payload)
        click.echo(f"wrote {out}")
    else:
        click.echo(payload)

    # Non-zero exit if the run failed, so batch scripts can detect it.
    if not result.score.valid:
        sys.exit(2)


if __name__ == "__main__":
    main()
