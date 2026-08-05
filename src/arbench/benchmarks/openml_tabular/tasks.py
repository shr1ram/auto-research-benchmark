"""The curated OpenML tabular task set: the in-domain benchmark.

Curated from the difficulty-filtered tabular ecosystem (Grinsztajn et al. 2022, which
deliberately EXCLUDES tasks a simple model already solves — i.e. headroom by construction)
plus a few OpenML-CTR23 regression tasks for diversity. Each entry is
plain data: an OpenML *dataset* id, the target column, the metric, and its direction.

Data staging: `prepare.py` materialises each task under
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

@dataclass(frozen=True)
class OpenMLTaskSpec:
    task_id: str            # our stable id (folder name), kebab-case
    dataset_id: int         # OpenML dataset id (data_id), used to download
    target: str             # target column name
    metric: str             # 'roc_auc' | 'log_loss' | 'rmse' | 'accuracy'
    higher_better: bool
    kind: str               # 'binary' | 'multiclass' | 'regression'
    provenance: str = ""    # curation record (source suite / original name);
                            # NEVER reaches prepared data or the agent


# --- Grinsztajn "numerical classification" (difficulty-curated, high-headroom) ---
# dataset_ids are OpenML data_ids from the Grinsztajn et al. 2022 suite.
# target fields below are OpenML's default_target_attribute (verified at prepare
# time); prepare.py auto-detects from metadata, so these are advisory but kept
# accurate. dataset_ids verified: all 11 downloaded cleanly.
CLASSIFICATION: list[OpenMLTaskSpec] = [
    OpenMLTaskSpec("credit",            44089, "SeriousDlqin2yrs", "roc_auc", True, "binary",
                   "OpenML credit (grinsztajn)"),
    OpenMLTaskSpec("electricity",       44120, "class",  "roc_auc", True,  "binary",
                   "OpenML electricity (grinsztajn)"),
    OpenMLTaskSpec("covertype",         44121, "Y",      "roc_auc", True,  "binary",
                   "OpenML covertype (grinsztajn)"),
    OpenMLTaskSpec("pol",               44122, "binaryClass", "roc_auc", True, "binary",
                   "OpenML pol (grinsztajn)"),
    OpenMLTaskSpec("house_16H",         44123, "binaryClass", "roc_auc", True, "binary",
                   "OpenML house_16H (grinsztajn)"),
    OpenMLTaskSpec("MagicTelescope",    44125, "class",  "roc_auc", True,  "binary",
                   "OpenML MagicTelescope (grinsztajn)"),
    OpenMLTaskSpec("bank-marketing",    44126, "Class",  "roc_auc", True,  "binary",
                   "OpenML bank-marketing (grinsztajn)"),
    OpenMLTaskSpec("MiniBooNE",         44128, "signal", "roc_auc", True,  "binary",
                   "OpenML MiniBooNE (grinsztajn)"),
]

# --- OpenML regression (different move-family: target transform, RMSE) ---
REGRESSION: list[OpenMLTaskSpec] = [
    OpenMLTaskSpec("cpu_activity",      44132, "usr",    "rmse", False, "regression",
                   "OpenML cpu_activity (grinsztajn-reg)"),
    OpenMLTaskSpec("wine_quality",      44136, "quality","rmse", False, "regression",
                   "OpenML wine_quality (grinsztajn-reg)"),
    OpenMLTaskSpec("superconduct",      44148, "criticaltemp", "rmse", False, "regression",
                   "OpenML superconduct (grinsztajn-reg)"),
]

# --- 2026-07 expansion: OpenML-CC18 + AutoML Benchmark (clf/reg) + CTR23,
# curated by scripts/curate_openml_suites.py (paper-gates + dedup applied;
# regenerate with that script). Task-admission gates still apply per task.
EXPANSION: list[OpenMLTaskSpec] = [
    OpenMLTaskSpec("abalone", 42726, "Class_number_of_rings", "rmse", False, "regression", "OpenML abalone (amlb-reg,ctr23)"),
    OpenMLTaskSpec("ada", 41156, "class", "roc_auc", True, "binary", "OpenML ada (amlb-clf)"),
    OpenMLTaskSpec("adult", 1590, "class", "roc_auc", True, "binary", "OpenML adult (cc18,amlb-clf)"),
    OpenMLTaskSpec("airlines", 1169, "Delay", "roc_auc", True, "binary", "OpenML airlines (amlb-clf)"),
    OpenMLTaskSpec("albert", 41147, "class", "roc_auc", True, "binary", "OpenML albert (amlb-clf)"),
    OpenMLTaskSpec("allstate-claims-severity", 42571, "loss", "rmse", False, "regression", "OpenML Allstate_Claims_Severity (amlb-reg)"),
    OpenMLTaskSpec("amazon-employee-access", 4135, "target", "roc_auc", True, "binary", "OpenML Amazon_employee_access (amlb-clf)"),
    OpenMLTaskSpec("apsfailure", 41138, "class", "roc_auc", True, "binary", "OpenML APSFailure (amlb-clf)"),
    OpenMLTaskSpec("auction-verification", 44958, "verification.time", "rmse", False, "regression", "OpenML auction_verification (ctr23)"),
    OpenMLTaskSpec("bioresponse", 4134, "target", "roc_auc", True, "binary", "OpenML Bioresponse (cc18,amlb-clf)"),
    OpenMLTaskSpec("black-friday", 41540, "Purchase", "rmse", False, "regression", "OpenML black_friday (amlb-reg)"),
    OpenMLTaskSpec("brazilian-houses", 42688, "total_(BRL)", "rmse", False, "regression", "OpenML Brazilian_houses (amlb-reg,ctr23)"),
    OpenMLTaskSpec("buzzinsocialmedia-twitter", 4549, "Annotation", "rmse", False, "regression", "OpenML Buzzinsocialmedia_Twitter (amlb-reg)"),
    OpenMLTaskSpec("california-housing", 44977, "medianHouseValue", "rmse", False, "regression", "OpenML california_housing (ctr23)"),
    OpenMLTaskSpec("christine", 41142, "class", "roc_auc", True, "binary", "OpenML christine (amlb-clf)"),
    OpenMLTaskSpec("churn", 40701, "class", "roc_auc", True, "binary", "OpenML churn (cc18,amlb-clf)"),
    OpenMLTaskSpec("click-prediction-small", 42733, "click", "roc_auc", True, "binary", "OpenML Click_prediction_small (amlb-clf)"),
    OpenMLTaskSpec("colleges", 42727, "percent_pell_grant", "rmse", False, "regression", "OpenML colleges (amlb-reg)"),
    OpenMLTaskSpec("connect-4", 40668, "class", "log_loss", False, "multiclass", "OpenML connect-4 (cc18,amlb-clf)"),
    OpenMLTaskSpec("cps88wages", 44984, "wage", "rmse", False, "regression", "OpenML cps88wages (ctr23)"),
    OpenMLTaskSpec("diabetes130us", 4541, "readmitted", "log_loss", False, "multiclass", "OpenML Diabetes130US (amlb-clf)"),
    OpenMLTaskSpec("diamonds", 42225, "price", "rmse", False, "regression", "OpenML diamonds (amlb-reg,ctr23)"),
    OpenMLTaskSpec("dna", 40670, "class", "log_loss", False, "multiclass", "OpenML dna (cc18,amlb-clf)"),
    OpenMLTaskSpec("elevators", 216, "Goal", "rmse", False, "regression", "OpenML elevators (amlb-reg)"),
    OpenMLTaskSpec("fabert", 41164, "class", "log_loss", False, "multiclass", "OpenML fabert (amlb-clf)"),
    OpenMLTaskSpec("fifa", 45012, "wage_eur", "rmse", False, "regression", "OpenML fifa (ctr23)"),
    OpenMLTaskSpec("first-order-theorem-proving", 1475, "Class", "log_loss", False, "multiclass", "OpenML first-order-theorem-proving (cc18,amlb-clf)"),
    OpenMLTaskSpec("fps-benchmark", 44992, "FPS", "rmse", False, "regression", "OpenML fps_benchmark (ctr23)"),
    OpenMLTaskSpec("gesturephasesegmentationprocessed", 4538, "Phase", "log_loss", False, "multiclass", "OpenML GesturePhaseSegmentationProcessed (cc18,amlb-clf)"),
    OpenMLTaskSpec("gina", 41158, "class", "roc_auc", True, "binary", "OpenML gina (amlb-clf)"),
    OpenMLTaskSpec("grid-stability", 44973, "stab", "rmse", False, "regression", "OpenML grid_stability (ctr23)"),
    OpenMLTaskSpec("har", 1478, "Class", "log_loss", False, "multiclass", "OpenML har (cc18)"),
    OpenMLTaskSpec("health-insurance", 44993, "whrswk", "rmse", False, "regression", "OpenML health_insurance (ctr23)"),
    OpenMLTaskSpec("house-sales", 42731, "price", "rmse", False, "regression", "OpenML house_sales (amlb-reg)"),
    OpenMLTaskSpec("internet-advertisements", 40978, "class", "roc_auc", True, "binary", "OpenML Internet-Advertisements (cc18,amlb-clf)"),
    OpenMLTaskSpec("isolet", 300, "class", "log_loss", False, "multiclass", "OpenML isolet (cc18)"),
    OpenMLTaskSpec("jannis", 41168, "class", "log_loss", False, "multiclass", "OpenML jannis (amlb-clf)"),
    OpenMLTaskSpec("jasmine", 41143, "class", "roc_auc", True, "binary", "OpenML jasmine (amlb-clf)"),
    OpenMLTaskSpec("jm1", 1053, "defects", "roc_auc", True, "binary", "OpenML jm1 (cc18)"),
    OpenMLTaskSpec("jungle-chess-2pcs-raw-endgame-complete", 41027, "class", "log_loss", False, "multiclass", "OpenML jungle_chess_2pcs_raw_endgame_complete (cc18,amlb-clf)"),
    OpenMLTaskSpec("kc1", 1067, "defects", "roc_auc", True, "binary", "OpenML kc1 (cc18,amlb-clf)"),
    OpenMLTaskSpec("kddcup09-appetency", 1111, "APPETENCY", "roc_auc", True, "binary", "OpenML KDDCup09_appetency (amlb-clf)"),
    OpenMLTaskSpec("kick", 41162, "IsBadBuy", "roc_auc", True, "binary", "OpenML kick (amlb-clf)"),
    OpenMLTaskSpec("kin8nm", 44980, "y", "rmse", False, "regression", "OpenML kin8nm (ctr23)"),
    OpenMLTaskSpec("kr-vs-kp", 3, "class", "roc_auc", True, "binary", "OpenML kr-vs-kp (cc18,amlb-clf)"),
    OpenMLTaskSpec("letter", 6, "class", "log_loss", False, "multiclass", "OpenML letter (cc18)"),
    OpenMLTaskSpec("madeline", 41144, "class", "roc_auc", True, "binary", "OpenML madeline (amlb-clf)"),
    OpenMLTaskSpec("madelon", 1485, "Class", "roc_auc", True, "binary", "OpenML madelon (cc18)"),
    OpenMLTaskSpec("mercedes-benz-greener-manufacturing", 42570, "y", "rmse", False, "regression", "OpenML Mercedes_Benz_Greener_Manufacturing (amlb-reg)"),
    OpenMLTaskSpec("mfeat-factors", 12, "class", "log_loss", False, "multiclass", "OpenML mfeat-factors (cc18,amlb-clf)"),
    OpenMLTaskSpec("mfeat-fourier", 14, "class", "log_loss", False, "multiclass", "OpenML mfeat-fourier (cc18)"),
    OpenMLTaskSpec("mfeat-karhunen", 16, "class", "log_loss", False, "multiclass", "OpenML mfeat-karhunen (cc18)"),
    OpenMLTaskSpec("mfeat-morphological", 18, "class", "log_loss", False, "multiclass", "OpenML mfeat-morphological (cc18)"),
    OpenMLTaskSpec("mfeat-zernike", 22, "class", "log_loss", False, "multiclass", "OpenML mfeat-zernike (cc18)"),
    OpenMLTaskSpec("miami-housing", 44983, "SALE_PRC", "rmse", False, "regression", "OpenML miami_housing (ctr23)"),
    OpenMLTaskSpec("naval-propulsion-plant", 44969, "gt_compressor_decay_state_coefficient", "rmse", False, "regression", "OpenML naval_propulsion_plant (ctr23)"),
    OpenMLTaskSpec("nomao", 1486, "Class", "roc_auc", True, "binary", "OpenML nomao (cc18,amlb-clf)"),
    OpenMLTaskSpec("numerai28-6", 23517, "attribute_21", "roc_auc", True, "binary", "OpenML numerai28.6 (cc18,amlb-clf)"),
    OpenMLTaskSpec("nyc-taxi-green-dec-2016", 42729, "tip_amount", "rmse", False, "regression", "OpenML nyc-taxi-green-dec-2016 (amlb-reg)"),
    OpenMLTaskSpec("okcupid-stem", 42734, "job", "log_loss", False, "multiclass", "OpenML okcupid-stem (amlb-clf)"),
    OpenMLTaskSpec("onlinenewspopularity", 42724, "shares", "rmse", False, "regression", "OpenML OnlineNewsPopularity (amlb-reg)"),
    OpenMLTaskSpec("ozone-level-8hr", 1487, "Class", "roc_auc", True, "binary", "OpenML ozone-level-8hr (cc18,amlb-clf)"),
    OpenMLTaskSpec("pendigits", 32, "class", "log_loss", False, "multiclass", "OpenML pendigits (cc18)"),
    OpenMLTaskSpec("philippine", 41145, "class", "roc_auc", True, "binary", "OpenML philippine (amlb-clf)"),
    OpenMLTaskSpec("phishingwebsites", 4534, "Result", "roc_auc", True, "binary", "OpenML PhishingWebsites (cc18,amlb-clf)"),
    OpenMLTaskSpec("phoneme", 1489, "Class", "roc_auc", True, "binary", "OpenML phoneme (cc18,amlb-clf)"),
    OpenMLTaskSpec("physiochemical-protein", 44963, "RMSD", "rmse", False, "regression", "OpenML physiochemical_protein (ctr23)"),
    OpenMLTaskSpec("porto-seguro", 42742, "target", "roc_auc", True, "binary", "OpenML porto-seguro (amlb-clf)"),
    OpenMLTaskSpec("pumadyn32nh", 44981, "thetadd6", "rmse", False, "regression", "OpenML pumadyn32nh (ctr23)"),
    OpenMLTaskSpec("quake", 550, "col_4", "rmse", False, "regression", "OpenML quake (amlb-reg)"),
    OpenMLTaskSpec("sarcos", 44976, "V22", "rmse", False, "regression", "OpenML sarcos (ctr23)"),
    OpenMLTaskSpec("sat11-hand-runtime-regression", 41980, "runtime", "rmse", False, "regression", "OpenML SAT11-HAND-runtime-regression (amlb-reg)"),
    OpenMLTaskSpec("satellite", 40900, "Target", "roc_auc", True, "binary", "OpenML Satellite (amlb-clf)"),
    OpenMLTaskSpec("satimage", 182, "class", "log_loss", False, "multiclass", "OpenML satimage (cc18)"),
    OpenMLTaskSpec("segment", 40984, "class", "log_loss", False, "multiclass", "OpenML segment (cc18,amlb-clf)"),
    OpenMLTaskSpec("shuttle", 40685, "class", "log_loss", False, "multiclass", "OpenML shuttle (amlb-clf)"),
    OpenMLTaskSpec("sick", 38, "Class", "roc_auc", True, "binary", "OpenML sick (cc18)"),
    OpenMLTaskSpec("space-ga", 507, "ln(VOTES/POP)", "rmse", False, "regression", "OpenML space_ga (amlb-reg,ctr23)"),
    OpenMLTaskSpec("spambase", 44, "class", "roc_auc", True, "binary", "OpenML spambase (cc18)"),
    OpenMLTaskSpec("splice", 46, "Class", "log_loss", False, "multiclass", "OpenML splice (cc18)"),
    OpenMLTaskSpec("sylvine", 41146, "class", "roc_auc", True, "binary", "OpenML sylvine (amlb-clf)"),
    OpenMLTaskSpec("texture", 40499, "Class", "log_loss", False, "multiclass", "OpenML texture (cc18)"),
    OpenMLTaskSpec("topo-2-1", 422, "oz267", "rmse", False, "regression", "OpenML topo_2_1 (amlb-reg)"),
    OpenMLTaskSpec("video-transcoding", 44974, "utime", "rmse", False, "regression", "OpenML video_transcoding (ctr23)"),
    OpenMLTaskSpec("volkert", 41166, "class", "log_loss", False, "multiclass", "OpenML volkert (amlb-clf)"),
    OpenMLTaskSpec("wall-robot-navigation", 1497, "Class", "log_loss", False, "multiclass", "OpenML wall-robot-navigation (cc18)"),
    OpenMLTaskSpec("wave-energy", 44975, "energy_total", "rmse", False, "regression", "OpenML wave_energy (ctr23)"),
    OpenMLTaskSpec("wilt", 40983, "class", "roc_auc", True, "binary", "OpenML wilt (cc18,amlb-clf)"),
    OpenMLTaskSpec("yolanda", 42705, "101", "rmse", False, "regression", "OpenML Yolanda (amlb-reg)"),
    OpenMLTaskSpec("yprop-4-1", 416, "oz252", "rmse", False, "regression", "OpenML yprop_4_1 (amlb-reg)"),
]

ALL_TASKS: list[OpenMLTaskSpec] = CLASSIFICATION + REGRESSION + EXPANSION

# Smallest-first for a cheap smoke run (by rough row count).
SMALL_FIRST: list[str] = [
    "mfeat-factors",         # 2,000 rows (smallest in the pool), multiclass
    "auction-verification",  # 2,043 rows, regression
    "kc1",                   # 2,109 rows, binary
    "quake",                 # 2,178 rows, regression
]

BY_ID: dict[str, OpenMLTaskSpec] = {t.task_id: t for t in ALL_TASKS}
