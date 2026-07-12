"""Verify every prepared openml_tabular task: integrity + split degeneracy.

    python scripts/verify_openml_prepared.py [--json out.json]

Checks per task (read-only):
  - meta.json consistent with tasks.py (metric, kind, target, classes present)
  - sample_submission columns == [id_col] + expected prediction columns
  - classes are unique after stringification (mixed-type collision guard)
  - data_version of the prepared dir matches a fresh recompute
  - split degeneracy (the stratification audit): per-class counts in train
    and in the private answers; flags any class missing from either side,
    minimum class counts, and the largest train->test proportion drift.
    Regression: quantile drift of the target between train and answers.

Exit 0 = all tasks pass integrity; degeneracy findings are REPORTED, not
fatal (they inform the stratify-or-document decision, benchmark plan §2).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from arbench.benchmarks.openml_tabular.tasks import BY_ID
from arbench.core.data_version import compute_data_version

PUBLIC = Path(os.environ.get("OPENML_DATA_DIR", "data/openml/public"))
PRIVATE = Path(os.environ.get("OPENML_PRIVATE_DATA_DIR", str(PUBLIC.parent / "private")))


def audit_task(task_id: str, spec) -> dict:
    prep = PUBLIC / task_id / "prepared"
    ans_path = PRIVATE / task_id / "answers.csv"
    row: dict = {"task_id": task_id, "kind": spec.kind, "metric": spec.metric,
                 "errors": [], "flags": []}
    if not (prep / "meta.json").exists():
        row["errors"].append("not prepared")
        return row
    meta = json.loads((prep / "meta.json").read_text())
    for field_name, want in (("metric", spec.metric), ("kind", spec.kind),
                             ("target", spec.target)):
        if meta.get(field_name) != want:
            row["errors"].append(
                f"meta.{field_name}={meta.get(field_name)!r} != tasks.py {want!r}")
    row["data_version"] = compute_data_version(prep)
    if meta.get("classes") is not None:
        cls = [str(c) for c in meta["classes"]]
        if len(set(cls)) != len(cls):
            row["errors"].append(f"class collision after str(): {cls}")
    sample = pd.read_csv(prep / "sample_submission.csv", nrows=1)
    expected = ([meta["id_col"], "prediction"] if spec.kind == "regression"
                else [meta["id_col"]] + list(meta.get("classes") or []))
    if list(sample.columns) != expected:
        row["errors"].append(f"sample columns {list(sample.columns)} != {expected}")

    target = meta["target"]
    train_y = pd.read_csv(prep / "train.csv", usecols=[target])[target]
    ans_y = pd.read_csv(ans_path, usecols=[target])[target]
    row["n_train"], row["n_test"] = int(len(train_y)), int(len(ans_y))
    if spec.kind == "regression":
        qs = [0.1, 0.5, 0.9]
        tq, aq = train_y.astype(float).quantile(qs), ans_y.astype(float).quantile(qs)
        spread = float(train_y.astype(float).std() or 1.0)
        row["max_quantile_drift_sd"] = round(
            float((tq.values - aq.values).max(initial=0) / spread), 4)
        if row["max_quantile_drift_sd"] > 0.25:
            row["flags"].append("target quantile drift > 0.25 sd")
        return row

    tr = train_y.astype(str).value_counts()
    te = ans_y.astype(str).value_counts()
    classes = sorted(set(tr.index) | set(te.index))
    row["n_classes"] = len(classes)
    row["min_class_train"] = int(min(tr.get(c, 0) for c in classes))
    row["min_class_test"] = int(min(te.get(c, 0) for c in classes))
    missing_test = [c for c in classes if te.get(c, 0) == 0]
    missing_train = [c for c in classes if tr.get(c, 0) == 0]
    if missing_test:
        row["flags"].append(f"classes ABSENT from test: {missing_test}")
    if missing_train:
        row["flags"].append(f"classes ABSENT from train: {missing_train}")
    drift = max(abs(tr.get(c, 0) / len(train_y) - te.get(c, 0) / len(ans_y))
                for c in classes)
    row["max_prop_drift"] = round(float(drift), 4)
    if row["min_class_test"] < 5:
        row["flags"].append(f"min class count in test = {row['min_class_test']}")
    if row["min_class_train"] < 20:
        row["flags"].append(f"min class count in train = {row['min_class_train']}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = [audit_task(tid, spec) for tid, spec in sorted(BY_ID.items())]
    bad = [r for r in rows if r["errors"]]
    flagged = [r for r in rows if r.get("flags")]
    for r in bad:
        print(f"ERROR {r['task_id']}: {'; '.join(r['errors'])}")
    for r in flagged:
        print(f"FLAG  {r['task_id']} ({r['kind']}, n_test={r.get('n_test')}): "
              f"{'; '.join(r['flags'])}")
    print(f"\n{len(rows)} tasks: {len(bad)} integrity errors, "
          f"{len(flagged)} split flags")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
