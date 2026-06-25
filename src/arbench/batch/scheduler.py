"""The batch scheduler: assign jobs to free GPU boxes, run them concurrently,
track progress, release boxes, and resume cleanly.

Model: one job per box at a time (an AIDE run wants the whole GPU). The shared
project FS means every box sees the same code/venv/data and writes its run
bundle to the same output tree — so the scheduler just needs to pick a box, ssh
`arbench run` to it, and collect the result.

Resumability: jobs whose run.json already exists are skipped. The manifest
(<sweep>/manifest.json) records per-job status/box/timing and is rewritten
atomically after every state change.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

from arbench.batch.worklist import Job
from arbench.batch import boxes as boxmod

_SSH_OPTS = [
    "-o", "RemoteCommand=none", "-o", "RequestTTY=no",
    "-o", "ControlMaster=no", "-o", "ControlPath=none",
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=30",
]


def _remote_run_cmd(job: Job, *, venv_activate: str, repo_dir: str,
                    data_dir: str, env_exports: str) -> str:
    """The bash run on the chosen box: activate venv, export env, `arbench run`."""
    job.out_dir.mkdir(parents=True, exist_ok=True)
    inner = (
        f"set -e; {env_exports} "
        f". {shlex.quote(venv_activate)}; "
        f"cd {shlex.quote(repo_dir)}; "
        f"arbench run --adapter {shlex.quote(job.adapter)} "
        f"--benchmark {shlex.quote(job.benchmark)} "
        f"--task {shlex.quote(job.task_id)} "
        f"--backend {shlex.quote(job.backend)} "
        f"--steps {job.steps} "
        f"--data-dir {shlex.quote(data_dir)} "
        f"--workspace {shlex.quote(str(job.out_dir))} "
        f"--out {shlex.quote(str(job.out_dir / 'result.json'))}"
    )
    if job.model:
        inner += f" --model {shlex.quote(job.model)}"
    return inner


class _Manifest:
    def __init__(self, sweep_dir: Path):
        self.path = Path(sweep_dir) / "manifest.json"
        self.lock = threading.Lock()
        self.data: dict = {"jobs": {}, "started_at": time.time()}

    def update(self, job: Job, **fields) -> None:
        with self.lock:
            j = self.data["jobs"].setdefault(job.name, {"name": job.name,
                                                         "task": job.task_id,
                                                         "seed": job.seed,
                                                         "arm": job.arm})
            j.update(fields)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2))
            os.replace(tmp, self.path)


def run_batch(
    jobs: list[Job],
    *,
    sweep_dir: Path,
    max_boxes: int,
    venv_activate: str,
    repo_dir: str,
    data_dir: str,
    env_exports: str = "",
    box_pattern: str | None = None,
    explicit_boxes: list[str] | None = None,
    poll_s: float = 15.0,
    log=print,
) -> dict:
    """Run all jobs across up to max_boxes GPU boxes. Returns a summary dict.

    Skips already-done jobs (resume). Releases each box as its job finishes and
    reassigns it to the next pending job.
    """
    sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest = _Manifest(sweep_dir)

    pending = [j for j in jobs if not j.is_done()]
    done_already = [j for j in jobs if j.is_done()]
    for j in done_already:
        manifest.update(j, status="already_done")
    log(f"[batch] {len(jobs)} jobs total, {len(done_already)} already done, "
        f"{len(pending)} to run, up to {max_boxes} boxes")

    candidates = explicit_boxes or boxmod.discover_boxes(box_pattern)
    log(f"[batch] {len(candidates)} candidate boxes: {', '.join(candidates[:8])}"
        + (" ..." if len(candidates) > 8 else ""))

    running: dict[str, tuple[Job, subprocess.Popen]] = {}   # box -> (job, proc)
    results: list[dict] = []
    self_held: set[str] = set()

    def _dispatch(job: Job, box: str) -> None:
        cmd = _remote_run_cmd(job, venv_activate=venv_activate, repo_dir=repo_dir,
                              data_dir=data_dir, env_exports=env_exports)
        logf = open(job.out_dir / "dispatch.log", "w")
        proc = subprocess.Popen(["ssh", *_SSH_OPTS, box, cmd],
                                stdout=logf, stderr=subprocess.STDOUT)
        running[box] = (job, proc)
        manifest.update(job, status="running", box=box, started_at=time.time())
        log(f"[batch] -> {box}: {job.name}")

    try:
        while pending or running:
            # 1) reap finished
            for box in list(running):
                job, proc = running[box]
                if proc.poll() is not None:
                    ok = job.is_done()  # run.json present => success
                    score = None
                    if ok:
                        try:
                            score = json.loads((job.out_dir / "run.json").read_text())["score"]["value"]
                        except Exception:
                            pass
                    manifest.update(job, status="done" if ok else "failed",
                                    rc=proc.returncode, score=score,
                                    finished_at=time.time())
                    results.append({"job": job.name, "ok": ok, "rc": proc.returncode, "score": score})
                    log(f"[batch] <- {box}: {job.name} {'OK' if ok else 'FAILED'}"
                        + (f" score={score}" if score is not None else ""))
                    boxmod.release(sweep_dir, box); self_held.discard(box)
                    del running[box]

            # 2) fill free slots with free boxes
            if pending and len(running) < max_boxes:
                free = boxmod.free_boxes([b for b in candidates if b not in running])
                for box in free:
                    if not pending or len(running) >= max_boxes:
                        break
                    if box in running:
                        continue
                    if not boxmod.try_lease(sweep_dir, box):
                        continue  # claimed by another scheduler/sweep
                    self_held.add(box)
                    _dispatch(pending.pop(0), box)

            if pending or running:
                time.sleep(poll_s)
    finally:
        # release any boxes we still hold (e.g. on Ctrl-C)
        for box in list(self_held):
            boxmod.release(sweep_dir, box)

    ok = sum(1 for r in results if r["ok"])
    summary = {"total": len(jobs), "ran": len(results), "ok": ok,
               "failed": len(results) - ok, "already_done": len(done_already),
               "sweep_dir": str(sweep_dir)}
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"[batch] done: {ok}/{len(results)} ok, {summary['already_done']} pre-done")
    return summary
