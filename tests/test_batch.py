"""Tests for the batch layer: worklist expansion, resume-skip, atomic leasing.

The scheduler's ssh dispatch isn't unit-tested here (needs a cluster); these
cover the deterministic logic that decides *what* runs and prevents double-claims.
"""
from __future__ import annotations

from pathlib import Path

from arbench.batch.worklist import expand_worklist, Job
from arbench.batch import boxes as bx


def test_worklist_cartesian(tmp_path):
    jobs = expand_worklist(
        tasks=["a", "b"], seeds=3, arms=["baseline", "continual"],
        adapter="continual", benchmark="mlebench_lite", sweep_dir=tmp_path,
        steps=8, backend="litellm", model=None,
    )
    assert len(jobs) == 2 * 3 * 2  # tasks × seeds × arms
    names = {j.name for j in jobs}
    assert "baseline/a-seed0" in names and "continual/b-seed2" in names
    # distinct out dirs
    assert len({j.out_dir for j in jobs}) == len(jobs)


def test_job_is_done_detects_run_json(tmp_path):
    j = Job(task_id="a", seed=0, arm="baseline", adapter="continual",
            benchmark="mlebench_lite", out_dir=tmp_path / "a-seed0",
            steps=8, backend="litellm", model=None)
    assert not j.is_done()
    j.out_dir.mkdir(parents=True)
    (j.out_dir / "run.json").write_text("{}")
    assert j.is_done()


def test_lease_is_exclusive(tmp_path):
    assert bx.try_lease(tmp_path, "lab-gpu-eider-l") is True
    # second claim of the same box fails (atomic O_EXCL)
    assert bx.try_lease(tmp_path, "lab-gpu-eider-l") is False
    assert "lab-gpu-eider-l" in bx.held_leases(tmp_path)
    bx.release(tmp_path, "lab-gpu-eider-l")
    assert "lab-gpu-eider-l" not in bx.held_leases(tmp_path)
    # released box can be re-claimed
    assert bx.try_lease(tmp_path, "lab-gpu-eider-l") is True


def test_discover_boxes_matches_pattern(tmp_path):
    cfg = tmp_path / "config"
    cfg.write_text(
        "Host lab-gpu-eider-l\n  HostName eider\n"
        "Host lab-gpu-scaup-l\n  HostName scaup\n"
        "Host knuckles\n  HostName knuckles\n"
        "Host gcp-tpu-v4\n  HostName x\n"
    )
    found = bx.discover_boxes(ssh_config=str(cfg))
    assert set(found) == {"lab-gpu-eider-l", "lab-gpu-scaup-l"}
    assert "knuckles" not in found and "gcp-tpu-v4" not in found
