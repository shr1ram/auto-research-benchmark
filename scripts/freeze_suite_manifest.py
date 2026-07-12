"""Freeze the openml_tabular task population as a versioned suite manifest.

    python scripts/freeze_suite_manifest.py --name openml-agent-tabular-v2 \
        [--audit-json audit.json] > suites/openml-agent-tabular-v2.json

The manifest is the citable identity of the benchmark ("derived from
CC18/AMLB/CTR23", never "results on" them — benchmark plan §2): suite
membership, dataset ids, targets, metrics + direction, split parameters, and
per-task data_version (from a verify_openml_prepared.py --json run, so the
frozen versions are the AUDITED ones). Freeze a new version whenever task
metadata changes (a data_version re-pin event); never edit a frozen manifest.
"""
from __future__ import annotations

import argparse
import json
import sys

from arbench.benchmarks.openml_tabular import prepare as prep
from arbench.benchmarks.openml_tabular.tasks import BY_ID


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--audit-json", default=None,
                    help="verify_openml_prepared.py --json output; supplies "
                         "per-task data_version")
    args = ap.parse_args()
    versions = {}
    if args.audit_json:
        rows = json.load(open(args.audit_json))
        errored = sorted(r["task_id"] for r in rows if r.get("errors"))
        if errored:
            # a manifest must only ever freeze a CLEAN audit — data_versions
            # of tasks that failed integrity checks are not citable
            raise SystemExit(f"refusing to freeze: audit has integrity "
                             f"errors for {errored}")
        versions = {r["task_id"]: r.get("data_version") for r in rows}
    manifest = {
        "name": args.name,
        "derived_from_suites": {"cc18": 99, "amlb-clf": 271,
                                "amlb-reg": 269, "ctr23": 353},
        "split": {"test_frac": prep.TEST_FRAC, "seed": prep.SEED,
                  "method": "seeded permutation, unstratified "
                            "(2026-07-12 audit: no degenerate splits)"},
        "n_tasks": len(BY_ID),
        "tasks": {tid: {"dataset_id": spec.dataset_id, "target": spec.target,
                        "kind": spec.kind, "metric": spec.metric,
                        "higher_better": spec.higher_better,
                        "data_version": versions.get(tid)}
                  for tid, spec in sorted(BY_ID.items())},
    }
    json.dump(manifest, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
