"""arbench CLI — the library's two hand-tools; nothing here drives an agent
(the loop lives in the thesis repo).

    arbench tasks --benchmark openml_tabular
    arbench grade --benchmark openml_tabular --task wine_quality submission.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

import arbench


def _bench(name: str, data_dir: str | None, private_data_dir: str | None):
    kwargs = {}
    if data_dir:
        kwargs["data_dir"] = data_dir
    if private_data_dir:
        kwargs["private_data_dir"] = private_data_dir
    return arbench.get_benchmark(name, **kwargs)


@click.group()
def main() -> None:
    """auto-research-benchmark — task listing + held-out grading."""


@main.command("tasks")
@click.option("--benchmark", "benchmark_name", required=True,
              type=click.Choice(arbench.BENCHMARK_NAMES),
              help="benchmark plugin, e.g. openml_tabular")
@click.option("--data-dir", default=None,
              help="benchmark public data root (else the plugin's env var)")
def tasks_cmd(benchmark_name, data_dir):
    """List the task ids a benchmark can serve."""
    for task_id in _bench(benchmark_name, data_dir, None).list_tasks():
        click.echo(task_id)


@main.command("splits")
@click.option("--fractions", default=None,
              help="bank=0.4,train=0.2,val=0.2,test=0.2 -> show the assignment")
@click.option("--seed", default=0, type=int, help="draw seed (default 0)")
def splits_cmd(fractions, seed):
    """The split index: families (facts) and, given fractions, the automatic
    seeded role assignment (bank/train/val/test, stratified per family)."""
    from arbench.core.splits import assign_roles, families, load_split_meta
    fams = families()
    if not fractions:
        for fam, members in fams.items():
            click.echo(f"{fam} ({len(members)}):")
            for benchmark, task_id in members:
                click.echo(f"  {task_id}")
        for b in ("openml_tabular", "mlebench_lite"):
            for task_id, e in load_split_meta(b).items():
                if e.get("excluded"):
                    click.echo(f"excluded: {task_id} — {e['excluded']}")
        click.echo("(pass --fractions to see the role assignment)")
        return
    fr = {k: float(v) for k, v in (kv.split("=") for kv in fractions.split(","))}
    assignment = assign_roles(fr, seed)
    for fam, members in fams.items():
        click.echo(f"{fam}:")
        for key in members:
            click.echo(f"  {assignment[key]:5s} {key[1]}")


@main.command("grade")
@click.option("--benchmark", "benchmark_name", required=True,
              type=click.Choice(arbench.BENCHMARK_NAMES))
@click.option("--task", "task_id", required=True, help="task/competition id")
@click.option("--data-dir", default=None,
              help="benchmark public data root (else the plugin's env var)")
@click.option("--private-data-dir", default=None,
              help="grader-only answers root (else env / sibling private/)")
@click.argument("submission", type=click.Path(exists=True, dir_okay=False))
def grade_cmd(benchmark_name, task_id, data_dir, private_data_dir, submission):
    """Grade a submission file against the held-out answers."""
    bench = _bench(benchmark_name, data_dir, private_data_dir)
    task = bench.load_task(task_id)
    score = bench.grade(task, Path(submission))
    click.echo(json.dumps({
        "task_id": task_id,
        "benchmark": benchmark_name,
        "value": score.value,
        "valid": score.valid,
        "is_higher_better": score.is_higher_better,
        "details": score.details,
    }, indent=2))
    # Non-zero exit on an invalid submission, so scripts can detect it.
    if not score.valid:
        sys.exit(2)


if __name__ == "__main__":
    main()
