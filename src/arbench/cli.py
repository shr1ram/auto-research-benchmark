"""arbench CLI — run an autoresearch system against a benchmark task.

    arbench list
    arbench run --adapter aide --benchmark mlebench_lite \
        --task random-acts-of-pizza --steps 10 --backend litellm \
        --workspace /cs/.../runs/raop --data-dir /cs/.../mlebench-data
"""
from __future__ import annotations

import json
import os
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


@main.command("batch")
@click.option("--adapter", required=True)
@click.option("--benchmark", default="mlebench_lite")
@click.option("--tasks", default="all-lite",
              help="'all-lite', a comma list, or @file with one task id per line")
@click.option("--seeds", default=1, type=int, help="independent repeats per task")
@click.option("--arms", default="baseline", help="comma list, e.g. baseline,continual")
@click.option("--out", "sweep_dir", required=True, type=click.Path(),
              help="sweep output dir on the SHARED FS (e.g. runs/sweep-001)")
@click.option("--max-boxes", default=4, type=int, help="max concurrent GPU boxes")
@click.option("--data-dir", default=None, help="benchmark data dir (else MLEBENCH_DATA_DIR)")
@click.option("--steps", default=8, type=int)
@click.option("--backend", default="litellm")
@click.option("--model", default=None)
@click.option("--venv", "venv_activate", required=True,
              help="path to the venv activate on the shared FS each box sources")
@click.option("--repo-dir", required=True, help="arbench repo dir on the shared FS")
@click.option("--boxes", default=None, help="explicit comma list of ssh hosts (else auto-discover)")
@click.option("--env-export", default="", help="extra 'export A=b; export C=d;' prefixed on each remote run")
def batch_cmd(adapter, benchmark, tasks, seeds, arms, sweep_dir, max_boxes,
              data_dir, steps, backend, model, venv_activate, repo_dir, boxes, env_export):
    """Run many (task × seed × arm) jobs in parallel across GPU boxes."""
    from arbench.batch import expand_worklist, run_batch

    # resolve task list
    if tasks == "all-lite":
        bench = registry.get_benchmark(benchmark, **({"data_dir": data_dir} if data_dir else {}))
        task_list = list(bench.list_tasks())
    elif tasks.startswith("@"):
        task_list = [l.strip() for l in Path(tasks[1:]).read_text().splitlines() if l.strip()]
    else:
        task_list = [t.strip() for t in tasks.split(",") if t.strip()]

    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    explicit = [b.strip() for b in boxes.split(",")] if boxes else None
    data_dir = data_dir or os.environ.get("MLEBENCH_DATA_DIR")

    jobs = expand_worklist(
        tasks=task_list, seeds=seeds, arms=arm_list, adapter=adapter,
        benchmark=benchmark, sweep_dir=Path(sweep_dir), steps=steps,
        backend=backend, model=model,
    )
    click.echo(f"[batch] {len(task_list)} tasks × {seeds} seeds × {len(arm_list)} arms "
               f"= {len(jobs)} jobs")
    summary = run_batch(
        jobs, sweep_dir=Path(sweep_dir), max_boxes=max_boxes,
        venv_activate=venv_activate, repo_dir=repo_dir, data_dir=data_dir,
        env_exports=env_export, explicit_boxes=explicit,
    )
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
