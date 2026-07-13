"""Re-emit the TEXT layer of already-prepared openml_tabular tasks in place —
description.md (upstream OpenML prose + fixed contract), sample_submission.csv
(per-class probability columns), meta.json (gains `classes`) — WITHOUT touching
train/test/answers. For the uniform-contract migration: data identical, so no
re-download/re-split; the .data_version stamp is dropped so the next load
re-pins the new fingerprint.

Usage:  OPENML_DATA_DIR=... python scripts/refresh_openml_task_text.py [task_id ...]
        (default: every prepared task under the public root; answers are read
        from OPENML_PRIVATE_DATA_DIR or the sibling private/ tree so `classes`
        covers labels that only occur in the held-out split.)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

from arbench.benchmarks.openml_tabular.prepare import (
    build_description, fetch_openml_description,
)
from arbench.benchmarks.openml_tabular.tasks import BY_ID


def refresh_one(prep: Path, answers_csv: Path) -> dict:
    meta = json.loads((prep / "meta.json").read_text())
    spec = BY_ID[meta["task_id"]]
    # SYNC the spec-owned fields from tasks.py: the stale-meta guard names
    # this script as the migration, so it must actually migrate (cubic P1,
    # PR #11 — refresh used to re-emit the OLD metric and stay refused).
    # kind/target are DATA facts a text refresh cannot change: mismatches
    # there need a full re-prepare, loudly.
    if (meta["kind"], meta["target"]) != (spec.kind, spec.target):
        raise ValueError(
            f"{meta['task_id']}: kind/target changed in tasks.py "
            f"({meta['kind']},{meta['target']}) -> ({spec.kind},{spec.target})"
            f" — a text refresh cannot migrate data; re-run prepare")
    meta["metric"], meta["higher_better"] = spec.metric, spec.higher_better
    target, kind = meta["target"], meta["kind"]

    classes = None
    if kind != "regression":
        train_labels = pd.read_csv(prep / "train.csv", usecols=[target])[target]
        labels = set(train_labels.dropna().astype(str))
        if not answers_csv.exists():
            # classes must cover heldout-only labels; a train-only list would
            # silently corrupt the grader's expected columns
            raise FileNotFoundError(
                f"answers not found at {answers_csv} — set OPENML_PRIVATE_DATA_DIR")
        ans = pd.read_csv(answers_csv, usecols=[target])[target]
        labels |= set(ans.dropna().astype(str))
        classes = sorted(labels)
        if any(c == meta["id_col"] for c in classes):
            raise ValueError(f"class value collides with id column: {classes}")

    test_ids = pd.read_csv(prep / "test.csv", usecols=[meta["id_col"]])
    sample = test_ids.copy()
    if kind == "regression":
        # same neutral value as prepare_one: the TRAIN-split mean
        sample["prediction"] = float(
            pd.read_csv(prep / "train.csv", usecols=[target])[target].mean())
    else:
        for c in classes:
            sample[c] = 1.0 / len(classes)
    sample.to_csv(prep / "sample_submission.csv", index=False)

    upstream = fetch_openml_description(meta["dataset_id"])
    (prep / "description.md").write_text(build_description(
        meta["task_id"], upstream, kind, meta["dataset_id"],
        target, meta["metric"], meta["higher_better"]))

    meta["classes"] = classes
    meta.pop("note", None)   # authored per-task text: deleted entirely
    (prep / "meta.json").write_text(json.dumps(meta, indent=2))
    stamp = prep / ".data_version"
    if stamp.exists():
        stamp.unlink()   # deliberate change: next load re-pins
    return {"classes": len(classes) if classes else None,
            "upstream_chars": len(upstream)}


def main(argv):
    root = os.environ.get("OPENML_DATA_DIR")
    if not root:
        raise SystemExit("set OPENML_DATA_DIR (project FS)")
    data_root = Path(root)
    proot = os.environ.get("OPENML_PRIVATE_DATA_DIR")
    private_root = Path(proot) if proot else data_root.parent / "private"
    ids = argv[1:] or sorted(p.parent.parent.name for p in data_root.glob("*/prepared/meta.json"))
    ok, fail = [], []
    for tid in ids:
        try:
            info = refresh_one(data_root / tid / "prepared",
                               private_root / tid / "answers.csv")
            print(f"OK  {tid:24s} classes={info['classes']} "
                  f"upstream={info['upstream_chars']}ch")
            ok.append(tid)
        except Exception as e:
            print(f"FAIL {tid:24s} {type(e).__name__}: {e}")
            fail.append(tid)
    print(f"\nrefreshed {len(ok)}/{len(ids)}  ({len(fail)} failed: {fail})")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
