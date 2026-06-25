"""GPU-box discovery and atomic file-leasing on the shared filesystem.

Self-contained (no dependency on private cluster infra): a free box is one we can
ssh into whose GPU is under a utilisation/memory threshold. A lease is an
O_EXCL-created file under <sweep>/.leases/<box> — so two concurrent schedulers
(or two sweeps) can't dispatch to the same box. Leases are released by deleting
the file; stale leases (holder PID/host gone) can be reaped.

Box list comes from ssh config Host entries matching a pattern (default the UCL
lab-gpu-*-l naming), overridable via ARBENCH_BOX_PATTERN / an explicit --boxes list.
"""
from __future__ import annotations

import os
import re
import socket
import subprocess
import time
from pathlib import Path

# default: UCL lab GPU boxes in ~/.ssh/config
DEFAULT_BOX_PATTERN = r"lab-gpu-[A-Za-z0-9]+-l"

_SSH_OPTS = [
    "-o", "RemoteCommand=none", "-o", "RequestTTY=no",
    "-o", "ControlMaster=no", "-o", "ControlPath=none",
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
]


def discover_boxes(pattern: str | None = None, ssh_config: str | None = None) -> list[str]:
    """All ssh Host aliases matching the pattern."""
    pattern = pattern or os.environ.get("ARBENCH_BOX_PATTERN", DEFAULT_BOX_PATTERN)
    cfg = Path(ssh_config or os.path.expanduser("~/.ssh/config"))
    if not cfg.exists():
        return []
    hosts = []
    for line in cfg.read_text().splitlines():
        m = re.match(r"\s*Host\s+(\S+)", line)
        if m and re.fullmatch(pattern, m.group(1)):
            hosts.append(m.group(1))
    return hosts


def _gpu_free(host: str, util_max: int = 15, mem_frac_max: float = 0.15) -> bool:
    """True if `host` is reachable and its GPU is idle enough to claim."""
    try:
        out = subprocess.run(
            ["ssh", *_SSH_OPTS, host,
             "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        line = out.stdout.strip().splitlines()[0]
        util, used, total = [int(x.strip()) for x in line.split(",")]
        return util < util_max and (used / total) < mem_frac_max
    except Exception:
        return False


def free_boxes(candidates: list[str], max_workers: int = 16) -> list[str]:
    """Concurrently probe candidates, return those currently free."""
    import concurrent.futures as cf
    free: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for host, ok in zip(candidates, ex.map(_gpu_free, candidates)):
            if ok:
                free.append(host)
    return free


# ── leasing ──────────────────────────────────────────────────────────────────
class LeaseError(Exception):
    pass


def _lease_dir(sweep_dir: Path) -> Path:
    d = Path(sweep_dir) / ".leases"
    d.mkdir(parents=True, exist_ok=True)
    return d


def try_lease(sweep_dir: Path, box: str) -> bool:
    """Atomically claim `box` for this sweep. True if acquired."""
    path = _lease_dir(sweep_dir) / box
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(f"{socket.gethostname()}:{os.getpid()}:{int(time.time())}\n")
    return True


def release(sweep_dir: Path, box: str) -> None:
    try:
        (_lease_dir(sweep_dir) / box).unlink()
    except FileNotFoundError:
        pass


def held_leases(sweep_dir: Path) -> list[str]:
    return [p.name for p in _lease_dir(sweep_dir).glob("*")]
