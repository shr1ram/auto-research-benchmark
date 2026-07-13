"""Curate the openml_tabular expansion from established OpenML suites.

Sources (suite ids are OpenML's):
    99   OpenML-CC18            (classification)
    271  AutoML Benchmark       (classification)
    269  AutoML Benchmark       (regression)
    353  OpenML-CTR23           (regression)

Paper-gates applied BEFORE any data download:
    - active dataset, dense format (no sparse ARFF)
    - 2,000 <= rows <= 600,000   (>= ~400 held-out rows; bounded prepare/exec)
    - features <= 2,000
    - classification: 2..50 classes (binary -> roc_auc, else log_loss)
    - not a flattened-image/pixel set (blocklist — vision content is the
      far_vision bin's job, not tabular's)
    - deduplicated by normalised dataset name across suites and against the
      existing curated 11

Emits three blocks on stdout: a report table, tasks.py spec lines, and
splits.yaml lines. Deterministic given OpenML's suite state; re-run to
regenerate. The task-admission gates (agent runs / in time / headroom) still
apply per task before any grid cell is trusted.

Usage: python scripts/curate_openml_suites.py > /tmp/curation.txt
"""
from __future__ import annotations

import json
import re
import time
import urllib.request

SUITES = {99: "cc18", 271: "amlb-clf", 269: "amlb-reg", 353: "ctr23"}

MIN_ROWS, MAX_ROWS = 2_000, 600_000
MAX_FEATURES = 2_000
MAX_CLASSES = 50

#: RAW-pixel datasets that are tabular in format only. Engineered-feature
#: image derivatives (mfeat-*, letter, texture, pendigits, and the obfuscated
#: AutoML-challenge sets) stay, same rationale as leaf-classification.
IMAGE_BLOCKLIST = {"mnist-784", "fashion-mnist", "cifar-10", "semeion",
                   "mfeat-pixel", "optdigits", "cifar-10-small",
                   "devnagari-script"}

#: normalised names: the curated 11 (under every alias) + same-data-two-names
#: pairs within the suites (keep one)
EXISTING = {"magictelescope", "magic-telescope", "miniboone", "bank-marketing",
            "cpu-activity", "cpu-act", "credit", "give-me-some-credit",
            "creditcard", "electricity", "electricity-normalized",
            "wine-quality", "white-wine", "wine-quality-white",
            "covertype", "covtype", "house-16h", "house-16",
            "superconduct", "superconductivity", "pol",
            "kings-county"}   # = house-sales (King County) under another name


def _get(url: str) -> dict:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise AssertionError


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def suite_data_ids(suite_id: int) -> list[int]:
    study = _get(f"https://www.openml.org/api/v1/json/study/{suite_id}")["study"]
    ids = study["data"]["data_id"]
    return [int(x) for x in (ids if isinstance(ids, list) else [ids])]


def dataset_info(data_id: int) -> dict:
    desc = _get(f"https://www.openml.org/api/v1/json/data/{data_id}")["data_set_description"]
    quals = _get(f"https://www.openml.org/api/v1/json/data/qualities/{data_id}")
    q = {e["name"]: e["value"] for e in quals["data_qualities"]["quality"]}

    def num(name):
        v = q.get(name)
        return int(float(v)) if v not in (None, "") else None

    return {"data_id": data_id, "name": desc["name"],
            "target": (desc.get("default_target_attribute") or "").strip(),
            "format": desc.get("format", ""), "status": desc.get("status", ""),
            "upload": desc.get("upload_date", "")[:10],
            "rows": num("NumberOfInstances"), "features": num("NumberOfFeatures"),
            "classes": num("NumberOfClasses")}


def admit(info: dict) -> str:
    """Empty string = admitted; else the rejection reason."""
    if info["status"] != "active":
        return f"status={info['status']}"
    if "sparse" in info["format"].lower():
        return "sparse format"
    if _norm(info["name"]) in IMAGE_BLOCKLIST:
        return "flattened-image content"
    if not info["target"] or "," in info["target"]:
        return f"no single default target ({info['target']!r})"
    if info["rows"] is None or not (MIN_ROWS <= info["rows"] <= MAX_ROWS):
        return f"rows={info['rows']}"
    if info["features"] is None or info["features"] > MAX_FEATURES:
        return f"features={info['features']}"
    classes = info["classes"] or 0
    if classes == 1 or classes > MAX_CLASSES:
        return f"classes={classes}"
    return ""


def main() -> None:
    seen: dict[str, dict] = {}
    rejected: list[tuple[str, str, str]] = []
    for suite_id, suite_name in SUITES.items():
        for data_id in suite_data_ids(suite_id):
            time.sleep(0.1)
            info = dataset_info(data_id)
            key = _norm(info["name"])
            if key in EXISTING:
                rejected.append((info["name"], suite_name, "already curated"))
                continue
            if key in seen:
                seen[key]["suites"].append(suite_name)
                continue
            reason = admit(info)
            if reason:
                rejected.append((info["name"], suite_name, reason))
                continue
            info["suites"] = [suite_name]
            seen[key] = info

    admitted = sorted(seen.values(), key=lambda i: _norm(i["name"]))
    print(f"# admitted {len(admitted)}, rejected {len(rejected)}\n")
    print(f"{'task_id':38s} {'data_id':>7s} {'rows':>7s} {'feat':>5s} "
          f"{'cls':>4s} {'kind':10s} {'upload':10s} suites")
    for i in admitted:
        classes = i["classes"] or 0
        kind = "regression" if classes == 0 else ("binary" if classes == 2 else "multiclass")
        print(f"{_norm(i['name'])[:38]:38s} {i['data_id']:7d} {i['rows']:7d} "
              f"{i['features']:5d} {classes:4d} {kind:10s} {i['upload']:10s} "
              f"{','.join(i['suites'])}")

    print("\n# ---- rejected (name, suite, reason)")
    for name, suite, reason in rejected:
        print(f"# {name[:40]:40s} {suite:9s} {reason}")

    print("\n# ---- tasks.py spec lines")
    for i in admitted:
        classes = i["classes"] or 0
        kind = "regression" if classes == 0 else ("binary" if classes == 2 else "multiclass")
        # metric is CONVENTION-DERIVED (AMLB), never authored: binary->AUC,
        # multiclass->log-loss, regression->RMSE (design-decisions "Task prose
        # is upstream-only" — the suites author the science)
        metric = "rmse" if kind == "regression" else ("roc_auc" if kind == "binary" else "log_loss")
        hb = "False" if metric in ("rmse", "log_loss") else "True"
        tid = _norm(i["name"])
        print(f'    OpenMLTaskSpec("{tid}", {i["data_id"]}, "{i["target"]}", '
              f'"{metric}", {hb}, "{kind}", "OpenML {i["name"]} ({",".join(i["suites"])})"),')

    print("\n# ---- splits.yaml lines")
    for i in admitted:
        classes = i["classes"] or 0
        kind = "regression" if classes == 0 else "classification"
        print(f"  {_norm(i['name'])}: {{family: tabular, kind: {kind}}}")


if __name__ == "__main__":
    main()
