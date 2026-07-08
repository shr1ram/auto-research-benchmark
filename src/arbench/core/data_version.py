"""data_version: a pinned fingerprint of a prepared task dir, checked at load.

Files up to CONTENT_HASH_MAX_BYTES are hashed by CONTENT; larger ones by
relative path + size only (multi-GB competition data would make every
load_task unaffordable). Prepared OpenML tasks and metadata files are all
small, so for them this IS a content hash; equal-size rewrites can hide only
inside huge files.

Trust-on-first-use: the first load stamps `.data_version` beside the data;
every later load recomputes and fails LOUDLY on drift. Graders recompute at
grade time into Score.details, so a manifest-scan audit catches anything
graded before a change landed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

STAMP_NAME = ".data_version"
CONTENT_HASH_MAX_BYTES = 8 * 1024 * 1024


def compute_data_version(data_dir: Path) -> str:
    data_dir = Path(data_dir)
    h = hashlib.sha256()
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name == STAMP_NAME:
            continue
        rel = path.relative_to(data_dir)
        size = path.stat().st_size
        if size <= CONTENT_HASH_MAX_BYTES:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            h.update(f"{rel}:{size}:{digest}\n".encode())
        else:
            h.update(f"{rel}:{size}\n".encode())
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
