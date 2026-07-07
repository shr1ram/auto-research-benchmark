"""The OpenML tabular task set — the accumulation spine's dense, move-rich substrate.

Curated from the difficulty-filtered tabular ecosystem (Grinsztajn et al. 2022, which
deliberately EXCLUDES tasks a simple model already solves — i.e. headroom by construction)
plus a few OpenML-CTR23 regression tasks for move-family diversity ([S2]). Each entry is
plain data: an OpenML *dataset* id, the target column, the metric, and its direction.

Data staging mirrors mlebench_lite: `prepare.py` materialises each task under
`$OPENML_DATA_DIR/<task_id>/prepared/{train.csv, test.csv}` (public tree; answers.csv
goes to the separate private root) with a fixed,
seeded proxy/held-out split. The benchmark validates the prepared layout at load time.

`SMALL_FIRST` leads with the smallest datasets for a cheap first end-to-end run.

Metrics: 'roc_auc' (binary, higher), 'log_loss' (multiclass, lower), 'rmse' (regression,
lower), 'accuracy' (higher). Matches the task-appropriate-metric heterogeneity Gate 2's
metric-relative headroom threshold is built to handle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class OpenMLTaskSpec:
    task_id: str            # our stable id (folder name), kebab-case
    dataset_id: int         # OpenML dataset id (data_id), used to download
    target: str             # target column name
    metric: str             # 'roc_auc' | 'log_loss' | 'rmse' | 'accuracy'
    higher_better: bool
    kind: str               # 'binary' | 'multiclass' | 'regression'
    note: str = ""          # short human hint
    # optional override if OpenML's default target col name differs; else use `target`
    positive_label: Optional[str] = None


# --- Grinsztajn "numerical classification" (difficulty-curated, high-headroom) ---
# dataset_ids are OpenML data_ids from the Grinsztajn et al. 2022 suite.
# target fields below are OpenML's default_target_attribute (verified at prepare
# time); prepare.py auto-detects from metadata, so these are advisory but kept
# accurate. dataset_ids verified: all 11 downloaded cleanly.
CLASSIFICATION: list[OpenMLTaskSpec] = [
    OpenMLTaskSpec("credit",            44089, "SeriousDlqin2yrs", "roc_auc", True, "binary",
                   "credit default; class imbalance + feature interactions"),
    OpenMLTaskSpec("electricity",       44120, "class",  "roc_auc", True,  "binary",
                   "electricity price up/down; temporal features matter"),
    OpenMLTaskSpec("covertype",         44121, "Y",      "roc_auc", True,  "binary",
                   "forest cover (binarised); nonlinear boundaries; LARGE (453k rows)"),
    OpenMLTaskSpec("pol",               44122, "binaryClass", "roc_auc", True, "binary",
                   "telecomms; feature selection sensitive"),
    OpenMLTaskSpec("house_16H",         44123, "binaryClass", "roc_auc", True, "binary",
                   "housing (binarised); scaling + interactions"),
    OpenMLTaskSpec("MagicTelescope",    44125, "class",  "roc_auc", True,  "binary",
                   "gamma vs hadron; class overlap"),
    OpenMLTaskSpec("bank-marketing",    44126, "Class",  "roc_auc", True,  "binary",
                   "term-deposit; strong imbalance -> imbalance moves"),
    OpenMLTaskSpec("MiniBooNE",         44128, "signal", "roc_auc", True,  "binary",
                   "particle ID; high-dim (50 feat), standardisation helps"),
]

# --- OpenML regression (different move-family: target transform, RMSE) ---
REGRESSION: list[OpenMLTaskSpec] = [
    OpenMLTaskSpec("cpu_activity",      44132, "usr",    "rmse", False, "regression",
                   "cpu usage; log-target + interactions"),
    OpenMLTaskSpec("wine_quality",      44136, "quality","rmse", False, "regression",
                   "wine score; ordinal-ish regression"),
    OpenMLTaskSpec("superconduct",      44148, "criticaltemp", "rmse", False, "regression",
                   "superconductor Tc; heavy feature engineering pays (79 feat)"),
]

ALL_TASKS: list[OpenMLTaskSpec] = CLASSIFICATION + REGRESSION

# Smallest-first for a cheap smoke run (by rough row count).
SMALL_FIRST: list[str] = [
    "wine_quality",     # ~6.5k rows, tiny
    "MagicTelescope",   # ~19k
    "bank-marketing",   # ~10k, imbalance
    "credit",           # curated hard
]

BY_ID: dict[str, OpenMLTaskSpec] = {t.task_id: t for t in ALL_TASKS}
