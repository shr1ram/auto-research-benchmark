"""data_version: a hash manifest of a prepared task dir, checked at load
(benchmark plan §2 — "never silently shifting numbers").

The version hashes the FILE LISTING (relative path + size) rather than full
contents: exact enough to catch re-prepares, renames, truncations, and added
files, cheap enough to run on every load_task against multi-GB competition
dirs. Trust-on-first-use: the first load stamps `.data_version` beside the
data; every later load recomputes and must match — a mismatch fails LOUDLY
(the run must die, not grade against different data). Graders recompute at
grade time and stamp the value into Score.details, so a manifest-scan audit
can catch anything graded before a change landed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

STAMP_NAME = ".data_version"


def compute_data_version(data_dir: Path) -> str:
    data_dir = Path(data_dir)
    h = hashlib.sha256()
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name == STAMP_NAME:
            continue
        rel = path.relative_to(data_dir)
        h.update(f"{rel}:{path.stat().st_size}\n".encode())
    return h.hexdigest()[:16]


def verify_data_version(data_dir: Path) -> str:
    """Compute, then check against (or create) the stamp. Returns the version;
    raises RuntimeError on drift."""
    data_dir = Path(data_dir)
    version = compute_data_version(data_dir)
    stamp = data_dir / STAMP_NAME
    if stamp.exists():
        pinned = stamp.read_text().strip()
        if pinned != version:
            raise RuntimeError(
                f"data_version mismatch under {data_dir}: pinned {pinned}, "
                f"found {version} — the prepared data changed since it was "
                f"stamped; re-prepare deliberately and delete {STAMP_NAME} "
                f"to re-pin")
    else:
        stamp.write_text(version + "\n")
    return version
