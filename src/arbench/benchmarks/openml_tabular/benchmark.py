"""OpenML tabular benchmark — the accumulation spine's dense tabular substrate.

Mirrors mlebench_lite: it hands out benchmark-agnostic Tasks and grades submissions,
never knowing which autoresearch system produced them. Data is staged by
`prepare.py` under `$OPENML_DATA_DIR/<task_id>/prepared/` (see that file for layout).

Grading reads `answers.csv` (held-out ground truth) and the agent's `submission.csv`,
computes the task's metric with sklearn, and returns a clean Score — never raises.

FIREWALL: the agent's data_dir is `$OPENML_DATA_DIR/<task>/prepared/` — a PUBLIC
tree that can be bind-mounted whole into containered runs. The ground truth lives
under a separate PRIVATE root: `$OPENML_PRIVATE_DATA_DIR/<task>/answers.csv`
(default: the sibling `private/` of the public root, i.e. data/openml/{public,private}).
It is never handed out and never named in Task metadata. load_task refuses to serve a task whose prepared/
still contains a legacy answers.csv (pre-firewall layout) — re-prep or migrate.
"""
from __future__ import annotations

import json
import os
import math

from pathlib import Path
from typing import Iterable

from arbench.core.benchmark import Benchmark
from arbench.core.task import Task
from arbench.core.result import Score
from arbench.core.data_version import compute_data_version, verify_data_version
from arbench.core.splits import split_of
from arbench.benchmarks.openml_tabular.tasks import ALL_TASKS, SMALL_FIRST, BY_ID


def _data_root() -> Path | None:
    r = os.environ.get("OPENML_DATA_DIR", "")
    return Path(r) if r else None


class OpenMLTabular(Benchmark):
    name = "openml_tabular"

    def __init__(self, data_dir: str | None = None, private_data_dir: str | None = None):
        self.data_dir = Path(data_dir) if data_dir else _data_root()
        priv = private_data_dir or os.environ.get("OPENML_PRIVATE_DATA_DIR", "")
        self.private_dir = Path(priv) if priv else None

    def _prepared(self, task_id: str) -> Path:
        if not self.data_dir:
            raise RuntimeError("OPENML_DATA_DIR not set (point it at the project FS).")
        return self.data_dir / task_id / "prepared"

    def _answers(self, task_id: str) -> Path:
        """Grader-only ground truth: <private root>/<task>/answers.csv, in a tree
        SEPARATE from the public root the agent (and the container bind) sees.
        Never expose this path to a Task."""
        if self.private_dir:
            return self.private_dir / task_id / "answers.csv"
        if not self.data_dir:
            raise RuntimeError("OPENML_DATA_DIR not set (point it at the project FS).")
        return self.data_dir.parent / "private" / task_id / "answers.csv"

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
        if (prep / "answers.csv").exists():
            raise RuntimeError(
                f"FIREWALL: {prep / 'answers.csv'} exists — the held-out answers "
                f"are inside the agent-visible data dir (legacy layout). Move it "
                f"to {self._answers(task_id)} (or re-run prepare.py) before "
                f"serving this task."
            )
        data_version = verify_data_version(prep)   # loud on drift (plan §2)
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
                "classes": meta.get("classes"),
                "sample_submission": str(prep / "sample_submission.csv"),
                "data_version": data_version,
                "split": split_of(self.name, task_id),
            },
        )

    def validate_submission(self, task: Task,
                            submission_path: Path) -> tuple[bool, str | None]:
        """PUBLIC-data-only mirror of grade()'s validity gates (columns from
        meta, row ids from test.csv) — the per-attempt loop check never opens
        the answers file. Agreement with grade() is pinned by test."""
        meta = task.metadata
        id_col, kind = meta["id_col"], meta.get("kind")
        try:
            import pandas as pd
            sub = pd.read_csv(submission_path)
        except Exception as e:  # noqa: BLE001
            return False, f"could not read submission: {e}"
        if kind == "regression":
            expected = ["prediction"]
        else:
            expected = list(meta.get("classes") or [])
            if not expected:
                return False, ("meta.json has no classes for a classification "
                               "task — re-run prepare/refresh for this task")
        missing = [c for c in [id_col] + expected if c not in sub.columns]
        if missing:
            return False, (f"submission must have columns {[id_col] + expected}; "
                           f"missing {missing} (got {list(sub.columns)})")
        try:
            test_ids = pd.read_csv(task.data_dir / "test.csv", usecols=[id_col])
        except Exception as e:  # noqa: BLE001
            return False, f"could not read test.csv ids: {e}"
        sub = sub.drop_duplicates(subset=[id_col])
        merged = test_ids.merge(sub[[id_col] + expected], on=id_col, how="left")
        if merged[expected].isna().any().any():
            n = int(merged[expected].isna().any(axis=1).sum())
            return False, f"submission missing predictions for {n} test rows"
        import numpy as np
        if kind == "regression":
            try:
                preds = merged["prediction"].to_numpy(dtype=float)
            except Exception as e:  # noqa: BLE001
                return False, f"predictions must be numeric: {e}"
            if not np.isfinite(preds).all():
                # grade()'s sklearn metrics raise on inf/nan — accepting them
                # here was a validate/grade disagreement (review find)
                return False, "predictions must be finite (no inf/nan)"
        else:
            try:
                probs = merged[expected].to_numpy(dtype=float)
            except Exception as e:  # noqa: BLE001
                return False, f"class probabilities must be numeric: {e}"
            if not np.isfinite(probs).all():
                return False, "class probabilities must be finite (no inf/nan)"
            if (probs < 0).any() or (probs.sum(axis=1) <= 0).any():
                return False, ("class probabilities must be non-negative "
                               "and sum > 0 per row")
        return True, None

    def grade(self, task: Task, submission_path: Path) -> Score:
        meta = task.metadata
        hb = not meta.get("is_lower_better", False)
        try:
            import numpy as np
            import pandas as pd
            ans = pd.read_csv(self._answers(task.task_id))
            sub = pd.read_csv(submission_path)
        except Exception as e:
            return Score.invalid(f"could not read submission/answers: {e}", is_higher_better=hb)

        id_col, target, metric = meta["id_col"], meta["target"], meta["metric"]
        kind = meta.get("kind")

        # expected columns come from the contract, mechanically: regression =
        # [id, prediction]; classification = [id] + one probability column per
        # class, NAMED by class value (sample_submission.csv shows the same) —
        # no positive-class convention exists to get wrong.
        if kind == "regression":
            expected = ["prediction"]
        else:
            expected = list(meta.get("classes") or [])
            if not expected:
                return Score.invalid(
                    "meta.json has no classes for a classification task — "
                    "re-run prepare/refresh for this task", is_higher_better=hb)
        missing = [c for c in [id_col] + expected if c not in sub.columns]
        if missing:
            return Score.invalid(
                f"submission must have columns {[id_col] + expected}; "
                f"missing {missing} (got {list(sub.columns)})", is_higher_better=hb)
        # align on id, guard against dup/missing ids
        sub = sub.drop_duplicates(subset=[id_col])
        merged = ans.merge(sub[[id_col] + expected], on=id_col, how="left")
        if merged[expected].isna().any().any():
            n = int(merged[expected].isna().any(axis=1).sum())
            return Score.invalid(f"submission missing predictions for {n} test rows",
                                 is_higher_better=hb)

        y_true = merged[target].values
        try:
            from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score
            if kind != "regression":
                probs = merged[expected].to_numpy(dtype=float)
                sums = probs.sum(axis=1)
                if (probs < 0).any() or (sums <= 0).any():
                    return Score.invalid("class probabilities must be non-negative "
                                         "and sum > 0 per row", is_higher_better=hb)
                probs = probs / sums[:, None]   # tolerate unnormalised rows
                yt = pd.Series(y_true).astype(str)
                unknown = sorted(set(yt) - set(expected))
                if unknown:
                    return Score.invalid(f"answers contain classes {unknown} not in "
                                         f"meta classes {expected}", is_higher_better=hb)
            if metric == "roc_auc":
                if len(expected) != 2:
                    return Score.invalid(f"roc_auc needs 2 classes, got {len(expected)}",
                                         is_higher_better=hb)
                pos = expected[-1]   # scored from that class's OWN column
                val = float(roc_auc_score((yt == pos).astype(int).values,
                                          probs[:, expected.index(pos)]))
            elif metric == "log_loss":
                val = float(log_loss(yt, probs, labels=expected))
            elif metric == "accuracy":
                pred = pd.Series(np.asarray(expected)[probs.argmax(axis=1)])
                val = float(accuracy_score(yt, pred))
            elif metric == "rmse":
                y_pred = merged["prediction"].values
                val = float(mean_squared_error(y_true.astype(float), y_pred.astype(float)) ** 0.5)
            else:
                return Score.invalid(f"unknown metric {metric}", is_higher_better=hb)
        except Exception as e:
            return Score.invalid(f"metric computation failed: {e}", is_higher_better=hb)

        if not math.isfinite(val):
            # roc_auc on single-class truth WARNS and returns nan instead of
            # raising — a "valid" NaN silently poisons every downstream
            # ranking that trusts valid=True (review find)
            return Score.invalid(f"metric {metric} returned non-finite {val!r}",
                                 is_higher_better=hb)
        return Score(value=val, valid=True, is_higher_better=hb,
                     details={"metric": metric, "kind": meta.get("kind"),
                              "n_test": int(len(merged)),
                              # recomputed HERE (the audit hook): catches data
                              # that changed between load and grade
                              "data_version": compute_data_version(task.data_dir)
                              if task.data_dir else ""})

