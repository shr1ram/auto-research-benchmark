"""Stage OpenML tabular tasks to the project FS, mirroring mlebench_lite's layout.

For each task spec, download the OpenML dataset (by data_id, via the public REST/
minio parquet endpoint — no `openml` pip dep), then write a fixed, seeded split:

    $OPENML_DATA_DIR/<task_id>/prepared/         # <- handed to the agent as data_dir
        train.csv        # features + target  (the agent trains on this; may self-split for CV)
        test.csv         # features only       (the agent predicts these -> submission.csv)
        sample_submission.csv
        description.md    # OpenML's own dataset prose + the fixed submission contract
        meta.json         # {metric, higher_better, kind, target, id_col, classes, dataset_id, ...}
    $OPENML_PRIVATE_DATA_DIR/<task_id>/          # <- grader-only, SEPARATE private tree
        answers.csv      # id,target           (held-out ground truth)
    (OPENML_PRIVATE_DATA_DIR defaults to the sibling `private/` of the public root,
    giving data/openml/{public,private} — the public root is container-bindable.)

FIREWALL: answers.csv must never live under prepared/ — that directory is handed
to the agent verbatim as its data dir, so anything in it is agent-readable.

The split is 80/20 seeded (proxy=train, held-out=test/answers) — the proxy-vs-held-out
structure [E2] needs. Deterministic given the seed so re-prep reproduces the same split.

Usage:  python openml_prepare.py [task_id ...]     (default: all)
        OPENML_DATA_DIR must be set (project FS, never NFS home).
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 1234
TEST_FRAC = 0.20

# import the specs (same dir on the box: benchmarks/openml_tabular/tasks.py)
try:
    from tasks import ALL_TASKS, BY_ID  # when run inside the package dir
except Exception:  # when imported as a module
    from arbench.benchmarks.openml_tabular.tasks import ALL_TASKS, BY_ID  # type: ignore


def fetch_openml_description(dataset_id: int) -> str:
    """The uploader-written dataset description from OpenML, IN FULL — the ONLY
    per-task prose we hand the agent. Everything else in description.md is a
    fixed contract; the science prose is all upstream, never authored or
    edited here."""
    meta_url = f"https://www.openml.org/api/v1/json/data/{dataset_id}"
    with urllib.request.urlopen(meta_url, timeout=60) as r:
        meta = json.load(r)["data_set_description"]
    return (meta.get("description") or "").strip()


def _download_openml_csv(dataset_id: int) -> tuple[pd.DataFrame, str, str]:
    """Fetch an OpenML dataset as a DataFrame + its authoritative target column
    + the uploader's dataset description.

    OpenML serves dataset metadata as JSON (which gives the data file URL — ARFF
    or parquet — AND `default_target_attribute`, the canonical target). We use the
    metadata target rather than trusting a hand-written guess. Parquet preferred.
    Returns (df, default_target_attribute, description).
    """
    meta_url = f"https://www.openml.org/api/v1/json/data/{dataset_id}"
    with urllib.request.urlopen(meta_url, timeout=60) as r:
        meta = json.load(r)["data_set_description"]
    default_target = (meta.get("default_target_attribute") or "").strip()
    description = (meta.get("description") or "").strip()
    parquet_url = meta.get("parquet_url")
    if parquet_url:
        with urllib.request.urlopen(parquet_url, timeout=300) as r:
            return pd.read_parquet(io.BytesIO(r.read())), default_target, description
    # fallback: ARFF via scipy
    arff_url = meta["url"]
    from scipy.io import arff
    with urllib.request.urlopen(arff_url, timeout=300) as r:
        data, _ = arff.loadarff(io.StringIO(r.read().decode("utf-8", "replace")))
    df = pd.DataFrame(data)
    # arff byte-strings -> str
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.decode("utf-8", "replace") if hasattr(df[c], "str") else df[c]
    return df, default_target, description


def build_description(task_id: str, upstream: str, kind: str, dataset_id: int,
                      target: str, metric: str, higher_better: bool) -> str:
    """description.md = [upstream OpenML prose] + [fixed contract]. The contract
    is one template per task KIND with zero per-task content beyond metadata
    facts (target/metric names). sample_submission.csv is the mechanical anchor
    for the expected columns, so no convention is left unstated (the smoke-e2e
    P(positive) orientation bug)."""
    if kind == "regression":
        contract = (
            "- Write `submission.csv` with EXACTLY the columns of "
            "`sample_submission.csv`: `row_id,prediction`, one row per test "
            "row (same `row_id`s), `prediction` = the predicted value.\n"
        )
    else:
        contract = (
            "- Write `submission.csv` with EXACTLY the columns of "
            "`sample_submission.csv`: `row_id` plus one probability column per "
            "class, each named by its class value, one row per test row (same "
            "`row_id`s); each row's probabilities should sum to 1.\n"
        )
    return (
        f"# {task_id}\n\n"
        + (f"{upstream}\n\n---\n\n" if upstream else "")
        + f"Tabular {kind} task (OpenML dataset {dataset_id}).\n\n"
        f"- Train on `train.csv` (columns include the target `{target}` and `row_id`).\n"
        f"- Predict for every row in `test.csv` (features only; has `row_id`, no target).\n"
        + contract
        + f"- Metric: **{metric}** ({'higher' if higher_better else 'lower'} is better), "
        f"computed on the hidden labels of `test.csv`.\n"
    )


def prepare_one(spec, data_root: Path, private_root: Path) -> dict:
    out = data_root / spec.task_id / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    private = private_root / spec.task_id   # grader-only, outside the public tree
    private.mkdir(parents=True, exist_ok=True)
    df, default_target, upstream_desc = _download_openml_csv(spec.dataset_id)
    # Prefer OpenML's authoritative default_target_attribute; fall back to the
    # spec's hand-written target only if the metadata didn't name one.
    target = default_target if (default_target and default_target in df.columns) else spec.target
    if target not in df.columns:
        raise ValueError(
            f"target {target!r} (spec={spec.target!r}, openml_default={default_target!r}) "
            f"not in columns: {list(df.columns)[:20]}"
        )
    df = df.reset_index(drop=True)
    if "row_id" in df.columns:   # a few datasets ship their own row_id
        df = df.rename(columns={"row_id": "row_id_orig"})
    df.insert(0, "row_id", np.arange(len(df)))

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    n_test = int(round(len(df) * TEST_FRAC))
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])

    train = df.iloc[train_idx].copy()
    test_full = df.iloc[test_idx].copy()
    test_features = test_full.drop(columns=[target])              # agent sees no labels
    answers = test_full[["row_id", target]].copy()                # grader-only

    # submission contract: classification = one probability column per class
    # (columns NAMED by class value — no positive-class convention to guess,
    # and multiclass log_loss gets the per-class probs it needs); regression =
    # a single `prediction` value column.
    classes = None
    sample = test_full[["row_id"]].copy()
    if spec.kind == "regression":
        # TRAIN mean only — the full-column mean would fold the heldout
        # labels' aggregate into a public file (cubic, PR #9)
        sample["prediction"] = float(train[target].mean())
    else:
        classes = sorted(map(str, pd.unique(df[target].dropna())))
        if any(c == "row_id" for c in classes):
            raise ValueError(f"class value collides with id column: {classes}")
        for c in classes:
            sample[c] = 1.0 / len(classes)

    train.to_csv(out / "train.csv", index=False)
    test_features.to_csv(out / "test.csv", index=False)
    answers.to_csv(private / "answers.csv", index=False)
    sample.to_csv(out / "sample_submission.csv", index=False)
    # Drop any legacy answers file written into prepared/ by an older prepare.py
    # (pre-firewall layout) so a re-prep also closes the leak.
    legacy = out / "answers.csv"
    if legacy.exists():
        legacy.unlink()

    meta = {
        "task_id": spec.task_id, "dataset_id": spec.dataset_id,
        "metric": spec.metric, "higher_better": spec.higher_better, "kind": spec.kind,
        "target": target, "id_col": "row_id", "classes": classes,
        "n_train": int(len(train)), "n_test": int(len(test_features)),
        "n_features": int(test_features.shape[1] - 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    (out / "description.md").write_text(build_description(
        spec.task_id, upstream_desc, spec.kind, spec.dataset_id,
        target, spec.metric, spec.higher_better))
    # a re-prep is a deliberate data change: drop the trust-on-first-use stamp
    # so the next load re-pins instead of failing on drift
    stamp = out / ".data_version"
    if stamp.exists():
        stamp.unlink()
    return meta


def main(argv):
    root = os.environ.get("OPENML_DATA_DIR")
    if not root:
        raise SystemExit("set OPENML_DATA_DIR (project FS)")
    data_root = Path(root)
    proot = os.environ.get("OPENML_PRIVATE_DATA_DIR")
    private_root = Path(proot) if proot else data_root.parent / "private"
    ids = argv[1:] or [t.task_id for t in ALL_TASKS]
    ok, fail = [], []
    for tid in ids:
        spec = BY_ID.get(tid)
        if not spec:
            print(f"SKIP unknown task {tid}"); continue
        try:
            m = prepare_one(spec, data_root, private_root)
            print(f"OK  {tid:16s} train={m['n_train']:>7d} test={m['n_test']:>6d} "
                  f"feat={m['n_features']:>3d} metric={m['metric']}")
            ok.append(tid)
        except Exception as e:
            print(f"FAIL {tid:16s} {type(e).__name__}: {e}")
            fail.append(tid)
    print(f"\nprepared {len(ok)}/{len(ids)}  ({len(fail)} failed: {fail})")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
