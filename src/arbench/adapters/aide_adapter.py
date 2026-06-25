"""Adapter that drives the (forked) AIDE autoresearch system.

AIDE exposes a programmatic `Experiment(data_dir, goal, eval).run(steps)` API and
runs candidate solution code inside its own per-experiment workspace. The MLE-bench
convention AIDE follows is that the solution code writes `submission.csv` into its
working directory; AIDE keeps the best node's artifacts around.

This adapter:
  1. configures AIDE's LLM backend (LiteLLM proxy + Kimi by default) via env,
  2. runs the experiment with data_dir/goal/eval from the harness Task,
  3. locates the produced submission.csv in AIDE's workspace tree and copies it
     to where the harness grader expects it.

It deliberately uses only AIDE's public surface + filesystem outputs, so a fork
that changes internals still works as long as it still emits a submission.csv.
"""
from __future__ import annotations

import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

from arbench.core.adapter import AutoResearchAdapter
from arbench.core.task import Task
from arbench.core.registry import register_adapter
from arbench.llm.backends import configure_aide_env, resolve_backend


class AIDEAdapter(AutoResearchAdapter):
    name = "aide"

    def __init__(
        self,
        steps: int = 10,
        backend: str | None = None,   # "litellm" | "claude_p" | "local"; default from env
        model: str | None = None,     # overrides the backend's default model
    ):
        self.steps = int(os.environ.get("AIDE_STEPS", steps))
        self.backend = backend or os.environ.get("ARBENCH_LLM_BACKEND", "litellm")
        self.model = model

    def prepare(self, task: Task, workspace: Path) -> None:
        # Point AIDE's OpenAI-compatible client at the chosen backend (LiteLLM/
        # Kimi by default). Mutates os.environ so the AIDE import picks it up.
        configure_aide_env(self.backend, model=self.model)

    def run(self, task: Task, workspace: Path) -> Path:
        workspace = Path(workspace)
        backend = resolve_backend(self.backend, model=self.model)

        # AIDE writes logs/workspaces relative to CWD; pin them under our workspace.
        aide_root = workspace / "aide_run"
        aide_root.mkdir(parents=True, exist_ok=True)
        prev_cwd = Path.cwd()
        os.chdir(aide_root)
        try:
            # AIDE selects its model from OmegaConf, which merges OmegaConf.from_cli()
            # (i.e. sys.argv dotlist overrides) at import/config-load time. The
            # default is o4-mini, which the LiteLLM proxy does NOT serve — so we
            # MUST override agent.code.model / agent.feedback.model to the backend's
            # model (Kimi-K2.6), else AIDE 404s. We inject the overrides into argv
            # for the duration of the config load + run.
            overrides = [
                f"agent.code.model={backend.model}",
                f"agent.feedback.model={backend.feedback_model}",
                f"agent.steps={self.steps}",
                f"report.model={backend.model}",
            ]
            with _aide_cli_overrides(overrides):
                # Import here so a machine without AIDE can still import the package,
                # and so the (re)load picks up our argv overrides.
                from aide import Experiment
                exp = Experiment(
                    data_dir=str(task.data_dir),
                    goal=task.goal,
                    eval=task.eval,
                )
                solution = exp.run(steps=self.steps)
                _ = solution  # best code also persisted in the journal/workspace
        finally:
            os.chdir(prev_cwd)

        dest = task.submission_path(workspace)
        found = _find_submission(aide_root)
        if found is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(found, dest)
        return dest


@contextmanager
def _aide_cli_overrides(overrides: list[str]):
    """Temporarily append OmegaConf dotlist overrides to sys.argv so AIDE's
    OmegaConf.from_cli() picks up our model selection. Restores argv after."""
    saved = sys.argv[:]
    try:
        sys.argv = sys.argv + overrides
        yield
    finally:
        sys.argv = saved


def _find_submission(aide_root: Path) -> Path | None:
    """Locate the submission AIDE produced. Prefer an explicit best-submission
    dir, else the newest submission.csv anywhere under the run tree."""
    candidates: list[Path] = []
    for pattern in ("**/best_submission/submission.csv",
                    "**/best_solution/submission.csv",
                    "**/submission/submission.csv",
                    "**/submission.csv"):
        candidates.extend(aide_root.glob(pattern))
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        return None
    # Newest wins (AIDE rewrites the best submission as it improves).
    return max(candidates, key=lambda p: p.stat().st_mtime)


register_adapter("aide", AIDEAdapter)
