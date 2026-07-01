"""OpenML tabular benchmark — the accumulation spine's dense tabular substrate.

Mirrors mlebench_lite: it hands out benchmark-agnostic Tasks and grades submissions,
never knowing which autoresearch system produced them. Data is staged by
`prepare.py` under `$OPENML_DATA_DIR/<task_id>/prepared/` (see that file for layout).

Grading reads `answers.csv` (held-out ground truth) and the agent's `submission.csv`,
computes the task's metric with sklearn, and returns a clean Score — never raises.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from arbench.core.benchmark import Benchmark
from arbench.core.task import Task
from arbench.core.result import Score
from arbench.core.registry import register_benchmark
from arbench.benchmarks.openml_tabular.tasks import ALL_TASKS, SMALL_FIRST, BY_ID


def _data_root() -> Path | None:
    r = os.environ.get("OPENML_DATA_DIR", "")
    return Path(r) if r else None


class OpenMLTabular(Benchmark):
    name = "openml_tabular"

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir) if data_dir else _data_root()

    def _prepared(self, task_id: str) -> Path:
        if not self.data_dir:
            raise RuntimeError("OPENML_DATA_DIR not set (point it at the project FS).")
        return self.data_dir / task_id / "prepared"

    def list_tasks(self) -> Iterable[str]:
        seen, ordered = set(), []
        for tid in SMALL_FIRST + [t.task_id for t in ALL_TASKS]:
            if tid not in seen:
                seen.add(tid); ordered.append(tid)
        return ordered

    def load_task(self, task_id: str) -> Task:
        spec = BY_ID.get(task_id)
        if not spec:
            raise RuntimeError(f"unknown openml_tabular task {task_id!r}; "
                               f"have {sorted(BY_ID)}")
        prep = self._prepared(task_id)
        if not (prep / "meta.json").exists():
            raise RuntimeError(
                f"task {task_id!r} is not prepared at {prep}.\n"
                f"Prepare it:  OPENML_DATA_DIR=... python -m "
                f"arbench.benchmarks.openml_tabular.prepare {task_id}"
            )
        meta = json.loads((prep / "meta.json").read_text())
        goal = (prep / "description.md").read_text()
        goal += (f"\nRead the training data under: {prep}\n"
                 f"Write your submission as submission.csv in your working directory.\n")
        return Task(
            task_id=task_id,
            benchmark=self.name,
            goal=goal,
            eval=f"Metric: {meta['metric']} "
                 f"({'higher' if meta['higher_better'] else 'lower'} is better).",
            data_dir=prep,
            submission_filename="submission.csv",
            metadata={
                "metric": meta["metric"],
                "is_lower_better": not meta["higher_better"],
                "kind": meta["kind"],
                "target": meta["target"],
                "id_col": meta["id_col"],
                "sample_submission": str(prep / "sample_submission.csv"),
                "answers_path": str(prep / "answers.csv"),
            },
        )

    def grade(self, task: Task, submission_path: Path) -> Score:
        meta = task.metadata
        hb = not meta.get("is_lower_better", False)
        try:
            import numpy as np
            import pandas as pd
            ans = pd.read_csv(meta["answers_path"])
            sub = pd.read_csv(submission_path)
        except Exception as e:
            return Score.invalid(f"could not read submission/answers: {e}", is_higher_better=hb)

        id_col, target = meta["id_col"], meta["target"]
        if id_col not in sub.columns or "prediction" not in sub.columns:
            return Score.invalid(
                f"submission must have columns [{id_col}, prediction]; got {list(sub.columns)}",
                is_higher_better=hb)
        # align on id, guard against dup/missing ids
        sub = sub.drop_duplicates(subset=[id_col])
        merged = ans.merge(sub[[id_col, "prediction"]], on=id_col, how="left")
        if merged["prediction"].isna().any():
            n = int(merged["prediction"].isna().sum())
            return Score.invalid(f"submission missing predictions for {n} test rows",
                                 is_higher_better=hb)

        y_true = merged[target].values
        y_pred = merged["prediction"].values
        metric = meta["metric"]
        try:
            from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score
            if metric == "roc_auc":
                # y_true may be strings/bools -> factorise to 0/1 (positive = max label)
                classes = pd.unique(pd.Series(y_true))
                if len(classes) != 2:
                    return Score.invalid(f"roc_auc needs 2 classes, got {len(classes)}", is_higher_better=hb)
                pos = sorted(map(str, classes))[-1]
                yt = (pd.Series(y_true).astype(str) == pos).astype(int).values
                val = float(roc_auc_score(yt, y_pred.astype(float)))
            elif metric == "log_loss":
                val = float(log_loss(y_true, y_pred))
            elif metric == "rmse":
                val = float(mean_squared_error(y_true.astype(float), y_pred.astype(float)) ** 0.5)
            elif metric == "accuracy":
                val = float(accuracy_score(pd.Series(y_true).astype(str),
                                           pd.Series(y_pred).astype(str)))
            else:
                return Score.invalid(f"unknown metric {metric}", is_higher_better=hb)
        except Exception as e:
            return Score.invalid(f"metric computation failed: {e}", is_higher_better=hb)

        return Score(value=val, valid=True, is_higher_better=hb,
                     details={"metric": metric, "kind": meta.get("kind"),
                              "n_test": int(len(merged))})


register_benchmark("openml_tabular", OpenMLTabular)
