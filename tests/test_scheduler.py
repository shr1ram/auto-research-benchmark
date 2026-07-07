"""Integration test for the scheduler loop using a FAKE box pool and FAKE ssh.

No cluster needed: we monkeypatch box discovery to a fixed pool and replace the
ssh dispatch with a local subprocess that writes a run.json (simulating a remote
`arbench run`). This exercises the real claim->dispatch->reap->release->reassign
loop, resume-skip, timeout-kill, and lease lifecycle.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from arbench.batch.worklist import expand_worklist
from arbench.batch import scheduler as sch
from arbench.batch import boxes as bx


def _fake_run_cmd(success=True, sleep=0.0, write_json=True):
    """Return a python -c command that simulates a remote arbench run: optionally
    sleeps, optionally writes run.json into the workspace given by --workspace."""
    body = (
        "import sys,os,json,time;"
        "ws=sys.argv[sys.argv.index('--workspace')+1];"
        "os.makedirs(ws,exist_ok=True);"
        f"time.sleep({sleep});"
    )
    if write_json:
        body += (
            "open(os.path.join(ws,'run.json'),'w').write("
            "json.dumps({'score':{'value':0.5}}));"
        )
    if not success:
        body += "sys.exit(2);"
    return body


@pytest.fixture
def patched(monkeypatch):
    pool = ["lab-gpu-a-l", "lab-gpu-b-l", "lab-gpu-c-l"]
    monkeypatch.setattr(bx, "discover_boxes", lambda *a, **k: list(pool))
    monkeypatch.setattr(sch.boxmod, "discover_boxes", lambda *a, **k: list(pool))
    # all pool boxes always "free"
    monkeypatch.setattr(sch.boxmod, "free_boxes", lambda cands, **k: list(cands))
    return pool


def _intercept_dispatch(monkeypatch, cmd_body):
    """Replace the remote-run command builder so dispatch runs cmd_body locally."""
    def fake_cmd(job, **kw):
        job.out_dir.mkdir(parents=True, exist_ok=True)
        return cmd_body + f" --workspace {job.out_dir}"
    monkeypatch.setattr(sch, "_remote_run_cmd", fake_cmd)
    # make the ssh invocation actually run python locally instead of ssh <box> <cmd>
    real_popen = sch.subprocess.Popen

    def fake_popen(args, **kw):
        # args = ["ssh", *opts, box, "echo <b64> | base64 -d | bash -l"].
        # Decode the payload (mirrors _ssh_payload) to recover the bash command,
        # then run our python body locally instead of ssh-ing.
        import base64
        payload = args[-1]
        b64 = payload.split(" ", 2)[1]  # "echo <b64> | ..."
        cmd = base64.b64decode(b64).decode()
        body, _, tail = cmd.partition(" --workspace ")
        new = [sys.executable, "-c", body, "--workspace", tail.strip()]
        return real_popen(new, **kw)
    monkeypatch.setattr(sch.subprocess, "Popen", fake_popen)


def test_all_jobs_run_and_succeed(tmp_path, patched, monkeypatch):
    _intercept_dispatch(monkeypatch, _fake_run_cmd(success=True))
    jobs = expand_worklist(tasks=["t1", "t2", "t3", "t4"], seeds=1, arms=["baseline"],
                           adapter="continual", benchmark="b", sweep_dir=tmp_path,
                           steps=1, backend="litellm", model=None)
    summary = sch.run_batch(jobs, sweep_dir=tmp_path, max_boxes=2,
                            venv_activate="x", repo_dir="x", data_dir="x",
                            poll_s=0.05, discovery_interval_s=0.0, log=lambda *a: None)
    assert summary["ok"] == 4 and summary["failed"] == 0
    # every job has a run.json and all leases released
    for j in jobs:
        assert (j.out_dir / "run.json").exists()
    assert bx.held_leases(tmp_path) == []


def test_resume_skips_done_jobs(tmp_path, patched, monkeypatch):
    jobs = expand_worklist(tasks=["t1", "t2"], seeds=1, arms=["baseline"],
                           adapter="continual", benchmark="b", sweep_dir=tmp_path,
                           steps=1, backend="litellm", model=None)
    # pre-complete t1
    jobs[0].out_dir.mkdir(parents=True)
    (jobs[0].out_dir / "run.json").write_text(json.dumps({"score": {"value": 0.9}}))
    _intercept_dispatch(monkeypatch, _fake_run_cmd(success=True))
    summary = sch.run_batch(jobs, sweep_dir=tmp_path, max_boxes=2,
                            venv_activate="x", repo_dir="x", data_dir="x",
                            poll_s=0.05, discovery_interval_s=0.0, log=lambda *a: None)
    assert summary["already_done"] == 1
    assert summary["ran"] == 1  # only t2 actually ran


def test_failed_job_is_recorded_not_crash(tmp_path, patched, monkeypatch):
    # remote exits 2 AND writes no run.json -> failed
    _intercept_dispatch(monkeypatch, _fake_run_cmd(success=False, write_json=False))
    jobs = expand_worklist(tasks=["t1", "t2"], seeds=1, arms=["baseline"],
                           adapter="continual", benchmark="b", sweep_dir=tmp_path,
                           steps=1, backend="litellm", model=None)
    summary = sch.run_batch(jobs, sweep_dir=tmp_path, max_boxes=2,
                            venv_activate="x", repo_dir="x", data_dir="x",
                            poll_s=0.05, discovery_interval_s=0.0, log=lambda *a: None)
    assert summary["failed"] == 2 and summary["ok"] == 0
    assert bx.held_leases(tmp_path) == []   # boxes freed even on failure


def test_failed_job_is_requeued_and_can_succeed(tmp_path, patched, monkeypatch):
    """A transient failure (no run.json, exit!=0) is re-queued; on the retry it
    succeeds. Simulated by a body that fails iff a marker file is absent, then
    creates the marker — so attempt 1 fails, attempt 2 writes run.json."""
    marker = tmp_path / "seen"

    def flaky_cmd(job, **kw):
        job.out_dir.mkdir(parents=True, exist_ok=True)
        body = (
            "import sys,os,json;"
            f"m={str(marker)!r};"
            "ws=sys.argv[sys.argv.index('--workspace')+1];"
            "os.makedirs(ws,exist_ok=True);"
            "first=not os.path.exists(m);"
            "(open(m,'w').write('x') if first else None);"
            # first attempt: fail with no run.json; retry: write run.json
            "(sys.exit(2) if first else open(os.path.join(ws,'run.json'),'w')"
            ".write(json.dumps({'score':{'value':0.7}})))"
        )
        return body + f" --workspace {job.out_dir}"
    monkeypatch.setattr(sch, "_remote_run_cmd", flaky_cmd)
    real_popen = sch.subprocess.Popen

    def fake_popen(args, **kw):
        import base64
        cmd = base64.b64decode(args[-1].split(" ", 2)[1]).decode()
        body, _, tail = cmd.partition(" --workspace ")
        return real_popen([sys.executable, "-c", body, "--workspace", tail.strip()], **kw)
    monkeypatch.setattr(sch.subprocess, "Popen", fake_popen)

    jobs = expand_worklist(tasks=["t1"], seeds=1, arms=["baseline"],
                           adapter="continual", benchmark="b", sweep_dir=tmp_path,
                           steps=1, backend="litellm", model=None)
    summary = sch.run_batch(jobs, sweep_dir=tmp_path, max_boxes=1,
                            venv_activate="x", repo_dir="x", data_dir="x",
                            poll_s=0.05, discovery_interval_s=0.0, max_attempts=2,
                            log=lambda *a: None)
    assert summary["ok"] == 1  # succeeded on the retry
    assert (jobs[0].out_dir / "run.json").exists()
    assert bx.held_leases(tmp_path) == []


def test_hung_job_is_timed_out(tmp_path, patched, monkeypatch):
    # remote sleeps 30s but never writes run.json; timeout=1s -> killed
    _intercept_dispatch(monkeypatch, _fake_run_cmd(success=True, sleep=30, write_json=False))
    jobs = expand_worklist(tasks=["t1"], seeds=1, arms=["baseline"],
                           adapter="continual", benchmark="b", sweep_dir=tmp_path,
                           steps=1, backend="litellm", model=None)
    summary = sch.run_batch(jobs, sweep_dir=tmp_path, max_boxes=1,
                            venv_activate="x", repo_dir="x", data_dir="x",
                            poll_s=0.1, discovery_interval_s=0.0, job_timeout_s=1.0,
                            max_attempts=1, log=lambda *a: None)
    # job killed -> not ok, but loop completed (didn't wedge) and box freed
    assert summary["ok"] == 0
    assert bx.held_leases(tmp_path) == []
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["jobs"]["baseline/t1-seed0"]["status"] == "timeout"


def test_persistent_dispatch_failure_fails_job_not_hangs(tmp_path, patched, monkeypatch):
    """A job whose dispatch ALWAYS fails must be marked failed after
    max_attempts dispatch tries — dispatch errors never bump `attempts`, so
    without the give-up cap this loops forever (the old suite hang)."""
    def always_fail(args, **kw):
        raise OSError("simulated permanent ssh failure")
    monkeypatch.setattr(sch, "_remote_run_cmd",
                        lambda job, **kw: "true --workspace " + str(job.out_dir))
    monkeypatch.setattr(sch.subprocess, "Popen", always_fail)

    jobs = expand_worklist(tasks=["t1"], seeds=1, arms=["baseline"], adapter="continual",
                           benchmark="b", sweep_dir=tmp_path, steps=1, backend="litellm", model=None)
    summary = sch.run_batch(jobs, sweep_dir=tmp_path, max_boxes=1,
                            venv_activate="x", repo_dir="x", data_dir="x",
                            poll_s=0.01, discovery_interval_s=0.0, max_attempts=2,
                            log=lambda *a: None)
    assert summary["ok"] == 0
    assert summary["failed"] == 1          # recorded as a job failure, not lost
    assert bx.held_leases(tmp_path) == []  # no leaked lease


def test_dispatch_failure_requeues_not_loses_job(tmp_path, patched, monkeypatch):
    """If Popen/mkdir fails for a box, the job must stay pending (not be lost)
    and the lease released — one bad box can't abort the sweep."""
    calls = {"n": 0}
    real_popen = sch.subprocess.Popen

    def flaky_popen(args, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("simulated ssh/Popen failure on first box")
        # subsequent dispatches: run a body that writes run.json
        import base64
        cmd = base64.b64decode(args[-1].split(" ", 2)[1]).decode()
        body, _, tail = cmd.partition(" --workspace ")
        return real_popen([sys.executable, "-c", body, "--workspace", tail.strip()], **kw)
    monkeypatch.setattr(sch, "_remote_run_cmd",
                        lambda job, **kw: _fake_run_cmd(True) + f" --workspace {job.out_dir}")
    monkeypatch.setattr(sch.subprocess, "Popen", flaky_popen)

    jobs = expand_worklist(tasks=["t1"], seeds=1, arms=["baseline"], adapter="continual",
                           benchmark="b", sweep_dir=tmp_path, steps=1, backend="litellm", model=None)
    summary = sch.run_batch(jobs, sweep_dir=tmp_path, max_boxes=1,
                            venv_activate="x", repo_dir="x", data_dir="x",
                            poll_s=0.05, discovery_interval_s=0.0, log=lambda *a: None)
    # the first dispatch failed but the job was re-dispatched and succeeded
    assert summary["ok"] == 1
    assert bx.held_leases(tmp_path) == []   # no leaked lease from the failed dispatch
