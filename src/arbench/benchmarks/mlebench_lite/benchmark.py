"""MLE-Bench Lite benchmark, wrapping OpenAI's `mlebench` package.

We reuse mlebench's registry (task metadata + prepared data layout) and its
grader (`grade_csv`) rather than reimplementing either. The only thing we add is
the adapter-facing Task view and a clean Score.

`mlebench` and a prepared competition must be present on the run box; if they are
not, load_task/grade raise a clear, actionable error instead of failing deep in
an import.

FIREWALL — the private split lives under a SEPARATE root. mlebench's stock layout
puts `prepared/private/` (held-out answers) one `../private` away from the
`prepared/public/` dir handed to the agent, and agent-written code runs
unsandboxed. We therefore keep two roots:

  * public root  (MLEBENCH_DATA_DIR):          <root>/<task>/prepared/public
  * private root (MLEBENCH_PRIVATE_DATA_DIR):  <root>-private/<task>/prepared/private

load_task resolves paths against the public root; grade() builds its Competition
against the private root (mlebench's Registry derives every answers path from its
data_dir, so pointing a second Registry at the private root is all it takes).
If MLEBENCH_PRIVATE_DATA_DIR is unset we fall back to `<data_dir>-private` when
that directory exists, else to the legacy single-root layout. In firewalled mode
load_task refuses to serve a task whose PUBLIC tree still contains a private/
dir (unmigrated data = live leak).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from arbench.core.benchmark import Benchmark
from arbench.core.task import Task
from arbench.core.result import Score
from arbench.core.data_version import compute_data_version, verify_data_version
from arbench.core.splits import split_of
from arbench.benchmarks.mlebench_lite.tasks import LITE_COMPETITIONS, SMALL_FIRST


def _require_mlebench():
    try:
        from mlebench.registry import registry  # noqa
        from mlebench.grade import grade_csv  # noqa
        return registry, grade_csv
    except Exception as e:  # pragma: no cover - depends on box install
        raise RuntimeError(
            "mlebench is not importable. Install it on the run box:\n"
            "  uv sync --extra mlebench   (vendor/mle-bench)\n"
            f"(underlying import error: {e})"
        ) from e


class MLEBenchLite(Benchmark):
    name = "mlebench_lite"

    def __init__(self, data_dir: str | None = None, private_data_dir: str | None = None):
        # Where prepared competition data lives. Default to MLEBENCH_DATA_DIR env
        # (we point this at the project FS on the box, never NFS home).
        self.data_dir = Path(
            data_dir or os.environ.get("MLEBENCH_DATA_DIR", "")
        ) if (data_dir or os.environ.get("MLEBENCH_DATA_DIR")) else None
        # Grader-only root holding each task's private split. Resolution:
        # explicit arg > env > sibling `<data_dir>-private`. There is NO
        # single-root fallback: answers living in the agent-bindable tree
        # would silently void the mount-level firewall (removed 2026-07-08).
        priv = private_data_dir or os.environ.get("MLEBENCH_PRIVATE_DATA_DIR", "")
        if priv:
            self.private_data_dir = Path(priv)
        elif self.data_dir is not None and Path(f"{self.data_dir}-private").is_dir():
            self.private_data_dir = Path(f"{self.data_dir}-private")
        else:
            self.private_data_dir = None

    def _registry(self):
        registry, _ = _require_mlebench()
        return registry.set_data_dir(self.data_dir) if self.data_dir else registry

    def _grading_registry(self):
        """A second mlebench Registry rooted at the private data dir, so
        Competition.answers / private_dir resolve OUTSIDE the agent-reachable
        tree. Falls back to the public registry in legacy single-root mode."""
        registry, _ = _require_mlebench()
        return registry.set_data_dir(self.private_data_dir) if self.private_data_dir else registry

    def _require_private_root(self) -> None:
        if self.private_data_dir is None:
            raise RuntimeError(
                "no private data root: set MLEBENCH_PRIVATE_DATA_DIR (or create "
                "the sibling '<data_dir>-private') — the legacy single-root mode "
                "was removed; answers must never live in the agent-bindable tree")

    def _firewalled(self) -> bool:
        return (self.private_data_dir is not None
                and self.private_data_dir != self.data_dir)

    def list_tasks(self) -> Iterable[str]:
        # The Lite split, smallest-first so the cheap smoke tasks lead.
        seen, ordered = set(), []
        for cid in SMALL_FIRST + LITE_COMPETITIONS:
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)
        return ordered

    def load_task(self, task_id: str) -> Task:
        self._require_private_root()
        comp = self._registry().get_competition(task_id)
        public_dir = Path(comp.public_dir)
        if not public_dir.exists():
            raise RuntimeError(
                f"competition {task_id!r} is not prepared at {public_dir}.\n"
                f"Prepare it first:  mlebench prepare -c {task_id}\n"
                "(requires Kaggle API creds in ~/.kaggle/kaggle.json and "
                "accepting the competition rules on kaggle.com)."
            )
        if self._firewalled() and Path(comp.private_dir).exists():
            raise RuntimeError(
                f"FIREWALL: {comp.private_dir} still exists inside the public data "
                f"root — the held-out split is one '../private' away from the "
                f"agent's data dir. Migrate it to "
                f"{self.private_data_dir / task_id / 'prepared' / 'private'} "
                f"before serving this task."
            )
        data_version = verify_data_version(public_dir)   # loud on drift (plan §2)
        goal = (
            f"{comp.description}\n\n"
            f"Read the training data under: {public_dir}\n"
            f"Produce a submission CSV matching the format of: "
            f"{comp.sample_submission}"
        )
        return Task(
            task_id=task_id,
            benchmark=self.name,
            goal=goal,
            eval=f"Graded by the Kaggle '{task_id}' metric. "
                 f"{'Lower is better.' if getattr(comp.grader, 'is_lower_better', False) else 'Higher is better.'}",
            data_dir=public_dir,
            submission_filename="submission.csv",
            metadata={
                "competition_id": task_id,
                "sample_submission": str(comp.sample_submission),
                "is_lower_better": bool(getattr(comp.grader, "is_lower_better", False)),
                "data_version": data_version,
                "split": split_of(self.name, task_id),
            },
        )

    def grade(self, task: Task, submission_path: Path) -> Score:
        self._require_private_root()
        registry, grade_csv = _require_mlebench()
        # Grade against the PRIVATE root: mlebench derives answers/private_dir
        # from its Registry's data_dir, and grade_csv only touches the private
        # side (is_dataset_prepared(grading_only=True) skips public checks).
        reg = self._grading_registry()
        try:
            comp = reg.get_competition(task.task_id)
            report = grade_csv(Path(submission_path), comp)
        except Exception as e:  # grading must never raise out of the harness
            return Score.invalid(f"grader error: {e}",
                                 is_higher_better=not task.metadata.get("is_lower_better", False))

        is_higher_better = not bool(getattr(report, "is_lower_better", False))
        # Distinguish "field absent" (a harness/version mismatch worth surfacing)
        # from a genuine invalid submission.
        if not hasattr(report, "valid_submission"):
            return Score.invalid(
                "mlebench report has no 'valid_submission' field "
                f"(available: {sorted(vars(report))[:12] if hasattr(report,'__dict__') else '?'})",
                is_higher_better=is_higher_better)
        if not getattr(report, "valid_submission", False):
            return Score.invalid("mlebench marked submission invalid",
                                 is_higher_better=is_higher_better)
        # valid_submission=True but no score => still invalid (don't emit a
        # 'valid' Score with value=None, which breaks the valid/None contract).
        if report.score is None:
            return Score.invalid("mlebench returned no score despite valid_submission",
                                 is_higher_better=is_higher_better)
        return Score(
            value=float(report.score),
            valid=True,
            is_higher_better=is_higher_better,
            details={
                "any_medal": bool(getattr(report, "any_medal", False)),
                "gold_medal": bool(getattr(report, "gold_medal", False)),
                "silver_medal": bool(getattr(report, "silver_medal", False)),
                "bronze_medal": bool(getattr(report, "bronze_medal", False)),
                "above_median": bool(getattr(report, "above_median", False)),
                "gold_threshold": getattr(report, "gold_threshold", None),
                "median_threshold": getattr(report, "median_threshold", None),
                # recomputed HERE (the audit hook): catches data that changed
                # between load and grade
                "data_version": compute_data_version(task.data_dir)
                if task.data_dir and task.data_dir.exists() else "",
            },
        )

