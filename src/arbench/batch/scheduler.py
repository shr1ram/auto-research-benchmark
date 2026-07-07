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


def _ssh_payload(bash_cmd: str) -> str:
    """Wrap a bash command so it runs correctly on the lab boxes.

    The login shell on the UCL lab boxes is csh, which can't parse the bash
    `export ...; . activate; ...` payload. Base64-encode it and decode+run under
    `bash -l` — immune to csh and to all quoting issues across the ssh boundary.
    """
    import base64
    b64 = base64.b64encode(bash_cmd.encode()).decode()
    return f"echo {b64} | base64 -d | bash -l"


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
    discovery_interval_s: float = 90.0,   # re-probe free boxes at most this often
    job_timeout_s: float = 7200.0,        # kill+free a box if a job hangs (2h default)
    max_attempts: int = 2,                # re-queue a failed job (e.g. box died) this many times
    log=print,
) -> dict:
    """Run all jobs across up to max_boxes GPU boxes. Returns a summary dict.

    Skips already-done jobs (resume). Releases each box as its job finishes and
    reassigns it to the next pending job. Box discovery is cached for
    discovery_interval_s so we don't hammer the ssh jump host every poll. A job
    exceeding job_timeout_s is killed and its box freed (a hung run can't wedge
    the sweep).
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

    running: dict[str, tuple[Job, subprocess.Popen, float]] = {}  # box -> (job, proc, started)
    results: list[dict] = []
    self_held: set[str] = set()
    _free_cache: list[str] = []
    _free_cache_ts: list[float] = [0.0]   # mutable holder for closure
    attempts: dict[str, int] = {}         # job.name -> times dispatched
    dispatch_fails: dict[str, int] = {}   # job.name -> failed dispatch tries

    def _dispatch(job: Job, box: str) -> bool:
        """Dispatch job to box. Returns True on success. On failure (mkdir/open/
        Popen error) it does NOT raise — releases the lease, leaves the job for
        the caller to re-queue, so one bad box can't abort the whole sweep."""
        try:
            # Dispatch owns its log file's parent dir — do not depend on the
            # cmd builder having created it.
            job.out_dir.mkdir(parents=True, exist_ok=True)
            cmd = _remote_run_cmd(job, venv_activate=venv_activate, repo_dir=repo_dir,
                                  data_dir=data_dir, env_exports=env_exports)
            # Close the parent's copy of the fd after Popen dups it into the child,
            # else every dispatch leaks an fd (sweeps exhaust the fd table).
            with open(job.out_dir / "dispatch.log", "w") as logf:
                proc = subprocess.Popen(["ssh", *_SSH_OPTS, box, _ssh_payload(cmd)],
                                        stdout=logf, stderr=subprocess.STDOUT)
        except Exception as e:
            boxmod.release(sweep_dir, box); self_held.discard(box)
            dispatch_fails[job.name] = dispatch_fails.get(job.name, 0) + 1
            if dispatch_fails[job.name] >= max_attempts:
                # Persistent dispatch failure is a JOB failure, not a box blip —
                # dispatch errors never bump `attempts`, so without this cap an
                # undispatchable job (e.g. unwritable out_dir) spins the sweep
                # forever.
                manifest.update(job, status="failed", box=box,
                                error=f"dispatch failed x{dispatch_fails[job.name]}: {str(e)[:160]}")
                results.append({"job": job.name, "ok": False, "rc": None,
                                "score": None, "status": "failed"})
                if job in pending:
                    pending.remove(job)
                log(f"[batch] xx {box}: {job.name} dispatch failed ({e}) — "
                    f"giving up after {dispatch_fails[job.name]} tries")
            else:
                manifest.update(job, status="dispatch_error", box=box, error=str(e)[:200])
                log(f"[batch] xx {box}: {job.name} dispatch failed ({e}) — leaving pending")
            return False
        attempts[job.name] = attempts.get(job.name, 0) + 1
        running[box] = (job, proc, time.monotonic())
        manifest.update(job, status="running", box=box, attempt=attempts[job.name],
                        started_at=time.time())
        log(f"[batch] -> {box}: {job.name} (attempt {attempts[job.name]})")
        return True

    def _reap(box, job, proc, killed=False):
        ok = (not killed) and job.is_done()
        score = None
        if ok:
            try:
                score = json.loads((job.out_dir / "run.json").read_text())["score"]["value"]
            except Exception:
                pass
        boxmod.release(sweep_dir, box); self_held.discard(box)
        del running[box]

        # Re-queue a non-success job (box died / ssh failed / transient) until
        # attempts are exhausted. A genuine graded failure still writes run.json,
        # so it's `ok` and never re-queued.
        if not ok and attempts.get(job.name, 0) < max_attempts:
            manifest.update(job, status="requeued", rc=proc.returncode,
                            attempt=attempts.get(job.name, 0))
            log(f"[batch] ~~ {box}: {job.name} {'TIMEOUT' if killed else 'FAILED'} "
                f"-> requeue (attempt {attempts.get(job.name,0)}/{max_attempts})")
            pending.append(job)
            return

        status = "timeout" if killed else ("done" if ok else "failed")
        manifest.update(job, status=status, rc=proc.returncode, score=score,
                        finished_at=time.time())
        results.append({"job": job.name, "ok": ok, "rc": proc.returncode,
                        "score": score, "status": status})
        log(f"[batch] <- {box}: {job.name} {status.upper()}"
            + (f" score={score}" if score is not None else ""))

    try:
        while pending or running:
            now = time.monotonic()
            # 1) reap finished / kill timed-out
            for box in list(running):
                job, proc, started = running[box]
                if proc.poll() is not None:
                    _reap(box, job, proc)
                elif now - started > job_timeout_s:
                    log(f"[batch] !! {box}: {job.name} exceeded {job_timeout_s:.0f}s — killing")
                    try:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except Exception:
                            proc.kill()
                            proc.wait(timeout=10)
                    except Exception:
                        pass
                    _reap(box, job, proc, killed=True)

            # 2) fill free slots (cached discovery to spare the jump host)
            if pending and len(running) < max_boxes:
                if now - _free_cache_ts[0] > discovery_interval_s or not _free_cache:
                    busy = set(running) | set(boxmod.held_leases(sweep_dir))
                    probe = [b for b in candidates if b not in busy]
                    _free_cache[:] = boxmod.free_boxes(probe)
                    _free_cache_ts[0] = now
                    log(f"[batch] discovery: {len(_free_cache)} free of {len(probe)} probed")
                while pending and len(running) < max_boxes and _free_cache:
                    box = _free_cache.pop(0)
                    if box in running or not boxmod.try_lease(sweep_dir, box):
                        continue
                    self_held.add(box)
                    job = pending[0]
                    if _dispatch(job, box):
                        pending.pop(0)        # dispatched — remove from queue
                    # else: _dispatch released the lease and left job in pending;
                    # drop this box from the cache so we try a different one next.

            if pending or running:
                time.sleep(poll_s)
    finally:
        # On Ctrl-C/exception, kill the ssh children FIRST (else remote arbench
        # runs keep using GPUs after we've released the lease), THEN release.
        for box in list(running):
            job, proc, _ = running[box]
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
            except Exception:
                pass
        for box in list(self_held):
            boxmod.release(sweep_dir, box)

    ok = sum(1 for r in results if r["ok"])
    summary = {"total": len(jobs), "ran": len(results), "ok": ok,
               "failed": len(results) - ok, "already_done": len(done_already),
               "sweep_dir": str(sweep_dir)}
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"[batch] done: {ok}/{len(results)} ok, {summary['already_done']} pre-done")
    return summary
