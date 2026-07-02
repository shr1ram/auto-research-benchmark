"""Adapter that drives the continual-auto-research ("car") system.

Unlike AIDE (a code-search agent with its own workspace machinery), car exposes a
generic hill-climbing loop: ``HillClimber(propose, runner).run(max_iter)``. The
loop is *substrate-agnostic* — it only knows propose -> run -> score -> climb. To
solve an MLE-bench task with it we supply:

  * a PROPOSER (an LLM) that, given the task goal + recent attempts, emits a
    COMPLETE Python solution script that reads the prepared data, trains a model,
    writes ``submission.csv``, and prints ``SCORE=<cv_metric>`` as its last line;
  * a local exec RUNNER that writes that script into the workspace, runs it on the
    box's GPU, and scores it from the ``SCORE=`` sentinel (car.scoring already
    parses this).

The score car optimises is the candidate's own CV/proxy. The harness grader runs
separately on the final ``submission.csv`` against the held-out Kaggle test — so
the proxy-vs-held-out gap (reward-hacking signal, RQ [E2]) falls out for free.

Like the AIDE adapter, this uses only car's public surface (HillClimber +
proposers + scoring) and the filesystem, so internal changes don't break it as
long as a candidate still leaves a submission.csv + prints SCORE=.

SECURITY NOTE — LLM-written solution scripts run UNSANDBOXED as the login user
(full containerization is out of scope for this harness). What we do mitigate:
each script runs with cwd inside its own iteration dir and a cleaned environment
(SSH agent / API keys / tokens stripped — see _clean_child_env), and the held-out
answers live outside the data dir the script is pointed at. Residual risk: the
script can still read anything the login user can read on the shared FS and open
network connections; treat solution code as untrusted-but-observed, not confined.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from arbench.core.adapter import AutoResearchAdapter
from arbench.core.task import Task
from arbench.core.registry import register_adapter
from arbench.core.trace import record_llm_call
from arbench.llm.backends import configure_aide_env, resolve_backend


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_IMAGE_KEYWORDS = ("image", "cactus", "leaf", "leaves", "x-ray", "xray", "histo",
                   "dog", "cat", "digit", "mnist", "cifar", "satellite", "aerial",
                   "photo", "picture", "pixel", "convolutional")


def _looks_like_image_task(task: Task) -> bool:
    """Heuristic: is this an image task (so the proposer should train a CNN)?

    Two cheap signals, OR'd: (1) the task id / goal / eval text mentions images;
    (2) the data dir actually contains image files. Either is enough — the cost of
    a false positive (nudging toward a CNN on non-image data) is low because the
    proposer still sees the real data, while a false negative produced a constant
    chance-level submission on aerial-cactus.
    """
    text = " ".join(
        str(x or "") for x in (task.task_id, getattr(task, "goal", ""),
                                getattr(task, "eval", ""))
    ).lower()
    if any(k in text for k in _IMAGE_KEYWORDS):
        return True
    try:
        data_dir = Path(task.data_dir)
        for i, p in enumerate(data_dir.rglob("*")):
            if i > 2000:  # bound the walk on large datasets
                break
            if p.suffix.lower() in _IMAGE_EXTS:
                return True
    except Exception:
        pass
    return False


# System prompt: turn the proposer into an MLE-bench solution writer. It must emit
# a self-contained script and the SCORE= contract; the runner enforces nothing
# else, so the contract lives here.
_CAR_MLE_SYSTEM = """\
You are a Kaggle/MLE-bench solver inside a hill-climbing loop. You are given the
task description, the path to the prepared data, and the best attempt so far with
its measured CV score. Propose ONE improved, COMPLETE, self-contained Python
solution.

HARD REQUIREMENTS (a violation scores the attempt as failed):
1. Output ONLY a single ```python ...``` code block. No prose outside it.
2. The script reads the training/test data from the data directory given in the
   task (do not hardcode a different path).
3. The script writes its predictions to `submission.csv` IN THE CURRENT WORKING
   DIRECTORY, matching the sample submission's columns and row order.
4. The script computes an honest cross-validation / held-out-split estimate of the
   competition metric and prints it as the FINAL stdout line, EXACTLY:
       SCORE=<number>
   (line-anchored, e.g. `SCORE=0.7421`). Print nothing after it.
   CRITICAL — SCORE must be the RAW competition metric in its NATURAL direction, the
   SAME value a leaderboard would show. Do NOT transform it: never print 1-metric,
   never negate it, never convert a "higher-is-better" metric into a loss. The loop
   is told separately whether higher or lower is better (see OPTIMIZATION DIRECTION
   in the task) and handles that itself. E.g. for ROC AUC print `SCORE=<auc>` (NOT
   `1-auc`); for RMSE print `SCORE=<rmse>`. Getting this wrong makes the loop optimise
   the OPPOSITE of the real objective.
5. Be robust AND FAST: set seeds, guard against missing columns, and keep runtime
   within the RUNTIME BUDGET stated in the task (a script that exceeds it is killed
   and scored as failed). Prefer a strong SIMPLE baseline first (cheap features,
   a fast model, few CV folds), then improve only if time allows. Favour
   linear/logistic or a small GBM over anything that scales badly with feature count.

To improve over the incumbent, change ONE meaningful thing (a research move:
better features, a different model class, handling imbalance/leakage, tuning,
ensembling) rather than rewriting blindly.
"""


class _MetricExtractor:
    """Independently extract the run's metric from its output — AIDE-style.

    AIDE does NOT trust a script's self-reported number; a separate feedback LLM
    call reads the execution output and returns {is_bug, metric, lower_is_better},
    and a buggy/None metric maps to the worst value. We mirror that here so the
    baseline (AIDE) and the intervention (CAR) derive their score the SAME way —
    otherwise a comparison would confound "memory" with "scoring robustness".

    Returns (metric: float|None, lower_is_better: bool|None, is_bug: bool, raw_text).
    metric is the RAW competition metric in its natural direction; the caller maps
    None/is_bug to a failed (non-finite) score for the hill-climber.
    """

    _SYS = (
        "You are evaluating the stdout/stderr of an ML solution script. From the "
        "output, extract the model's FINAL cross-validation / held-out score for the "
        "competition metric. Respond with STRICT JSON only: "
        '{"is_bug": <bool>, "metric": <number or null>, "lower_is_better": <bool>}. '
        "is_bug=true if the script errored, produced no usable metric, or the metric "
        "looks invalid. metric = the RAW metric value as the script reports it (do NOT "
        "transform it). lower_is_better = whether a LOWER value of THIS metric is better "
        "(true for losses/error like RMSE/logloss; false for AUC/accuracy/F1). "
        "If you cannot find a metric, is_bug=true and metric=null."
    )

    def __init__(self, model: str, base_url: str, api_key: str):
        self.model, self.base_url, self.api_key = model, base_url, api_key
        self._client = None

    def _client_(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    # Force structured output via function-calling, exactly like AIDE's
    # review_func_spec — the model MUST return these fields as tool-call args, so
    # it cannot narrate prose that free-text json.loads would choke on. Keeping this
    # identical to AIDE is the whole point of the extractor: the CAR intervention
    # and the AIDE baseline must derive their score the same way.
    _TOOL = {
        "type": "function",
        "function": {
            "name": "submit_review",
            "description": "Submit a review of the training script's output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "is_bug": {"type": "boolean",
                               "description": "true if the execution failed / has a bug, else false."},
                    "metric": {"type": ["number", "null"],
                               "description": "the RAW validation metric the script reports "
                                              "(do NOT transform it); null if none / buggy."},
                    "lower_is_better": {"type": "boolean",
                               "description": "true if a LOWER value of THIS metric is better "
                                              "(RMSE/logloss), false for AUC/accuracy/F1."},
                },
                "required": ["is_bug", "metric", "lower_is_better"],
            },
        },
    }

    def extract(self, eval_desc: str, output: str):
        """AIDE-style independent metric extraction via forced function-calling.

        Two failure classes, kept distinct like AIDE:
          - the LLM ran and judged the solution bad / metric-less -> is_bug=True
            (a legitimate verdict; the controller treats it as a failed run);
          - the extractor CALL ITSELF failed (gateway 504 / connection death /
            timeout) -> RAISE. A transport failure is NOT the solution's fault and
            must not be silently laundered into "the solution is buggy".
            _LocalExecRunner.run catches the raise and fails that ITERATION
            (loud, recorded) — never the whole run, so the best submission from
            earlier iterations still gets graded.
        """
        import json as _json
        user = (f"COMPETITION METRIC: {eval_desc}\n\n"
                f"--- script output (tail) ---\n{output[-3500:]}")
        # Forced tool-choice: the model must call submit_review with structured
        # args. This is short (no long codegen), so the alibaba-ga idle-504 risk is
        # low; if a transport error DOES occur it propagates (loud), by design —
        # the runner catches it and fails the ITERATION, not the run.
        t0 = time.monotonic()
        resp = self._client_().chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self._SYS},
                      {"role": "user", "content": user}],
            temperature=0, timeout=120,
            tools=[self._TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_review"}},
        )
        msg = resp.choices[0].message
        # Telemetry: the extractor is a real LLM call this arm pays for — record
        # it like AIDE's feedback calls so run.json/cost aren't one-sided.
        usage = getattr(resp, "usage", None)
        record_llm_call(
            model=self.model, role="feedback",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=time.monotonic() - t0,
            system=self._SYS, user=user,
            response=(msg.content or "") or str(getattr(msg, "tool_calls", "")),
            extra={"source": "car_metric_extractor"},
        )
        tcs = getattr(msg, "tool_calls", None)
        if not tcs:
            # Model answered but didn't call the tool -> no usable metric (genuine
            # is_bug, like AIDE's metric=None). NOT a transport failure.
            return None, None, True, f"no tool call; content={((msg.content or '')[:200])!r}"
        try:
            args = _json.loads(tcs[0].function.arguments or "{}")
        except _json.JSONDecodeError:
            return None, None, True, f"unparseable tool args: {tcs[0].function.arguments[:200]!r}"
        metric = args.get("metric")
        metric = float(metric) if isinstance(metric, (int, float)) else None
        return metric, args.get("lower_is_better"), bool(args.get("is_bug", metric is None)), _json.dumps(args)


# Exact env var names that carry live credentials/handles a solution script has
# no business inheriting; plus a suffix pattern for the API-key/token family.
_ENV_DENY_EXACT = {
    "SSH_AUTH_SOCK", "SSH_AGENT_PID", "SSH_CLIENT", "SSH_CONNECTION", "SSH_TTY",
    "ARBENCH_LLM_TRACE",  # the child must not append to the run's LLM trace
}
_ENV_DENY_SUFFIX = re.compile(r"(_API_KEY|_APIKEY|_TOKEN|_SECRET|_PASSWORD|_CREDENTIALS)$")


def _clean_child_env(data_dir: Path) -> dict[str, str]:
    """Environment for the LLM-written solution subprocess: the parent env minus
    credentials-bearing variables. Deny-list (not allow-list) on purpose — torch/
    CUDA/OpenML/PATH/HOME and friends must keep working, and the box env is the
    only reliable source of those. Full sandboxing is out of scope (see module
    docstring)."""
    env = {
        k: v for k, v in os.environ.items()
        if k not in _ENV_DENY_EXACT and not _ENV_DENY_SUFFIX.search(k)
    }
    env["MLEBENCH_DATA_DIR"] = str(data_dir)
    env["DATA_DIR"] = str(data_dir)  # also expose the common name
    return env


def _extract_code(proposal: str) -> str:
    """Pull the Python from a ```python ...``` block; fall back to the raw text if
    the model forgot the fence (best-effort — a non-script proposal will just fail
    to run and score as a failed attempt)."""
    text = proposal or ""
    fence = "```"
    if fence in text:
        # take the content of the first fenced block, dropping an optional
        # ```python / ```py language tag on the opening fence.
        after = text.split(fence, 1)[1]
        body = after.split(fence, 1)[0]
        first_nl = body.find("\n")
        if first_nl != -1 and body[:first_nl].strip().lower() in ("python", "py", ""):
            body = body[first_nl + 1:]
        return body.strip()
    return text.strip()


class _LocalExecRunner:
    """Run a proposed solution script locally in the workspace and score it.

    Contract (car's Runner protocol): ``run(proposal, iteration) -> (score, output)``.
    Writes the script, executes it with the data dir on the environment, parses the
    ``SCORE=`` sentinel via car.scoring, and snapshots the submission.csv this
    iteration produced so the adapter can grade the BEST iteration's submission
    (not merely the last)."""

    def __init__(self, workspace: Path, data_dir: Path, timeout_s: int = 600,
                 extractor: "Optional[_MetricExtractor]" = None, eval_desc: str = ""):
        self.workspace = Path(workspace)
        self.data_dir = Path(data_dir)
        # AIDE-style independent metric extraction (preferred over trusting the
        # script's SCORE= self-report). None => fall back to the SCORE= sentinel.
        self.extractor = extractor
        self.eval_desc = eval_desc
        # Per-iteration execution ceiling. Default 10 min: generous enough for an
        # honest CV on a cheap task, tight enough that a runaway/hung script can't
        # burn the whole run. Override with CAR_EXEC_TIMEOUT_S. (AIDE's own cap is
        # 3600s but it steers the model to fast code + feeds back exec results;
        # we do the latter via last_exec_feedback below.)
        self.timeout_s = int(os.environ.get("CAR_EXEC_TIMEOUT_S", timeout_s))
        self.best_dir = self.workspace / "car_best"      # snapshot of best submission
        self.iters_dir = self.workspace / "car_iters"    # per-iter scripts + logs
        self.iters_dir.mkdir(parents=True, exist_ok=True)
        self.last_trace: Optional[dict] = None
        # Structured outcome of the most recent run, fed back to the proposer so it
        # can self-correct (the key thing AIDE has that a bare hill-climber lacks).
        self.last_exec_feedback: Optional[str] = None

    def run(self, proposal: str, iteration: int) -> Tuple[Optional[float], str]:
        from continual_auto_research.core import scoring

        code = _extract_code(proposal)
        rundir = self.iters_dir / f"iter_{iteration:03d}"
        rundir.mkdir(parents=True, exist_ok=True)
        script = rundir / "solution.py"
        script.write_text(code)

        # Clear any stale submission so a script that fails to write one doesn't
        # get scored on a previous iteration's file.
        sub = rundir / "submission.csv"
        if sub.exists():
            sub.unlink()

        # Cleaned env: no SSH agent / API keys / tokens for untrusted solution
        # code (see _clean_child_env). cwd is the iteration dir inside the
        # workspace, so relative writes land there.
        env = _clean_child_env(self.data_dir)

        status = "ok"
        stderr_text = ""
        try:
            proc = subprocess.run(
                [sys.executable, "solution.py"],
                cwd=str(rundir), env=env,
                capture_output=True, text=True, timeout=self.timeout_s,
            )
            stderr_text = proc.stderr or ""
            output = (proc.stdout or "") + "\n[stderr]\n" + stderr_text
            if proc.returncode != 0:
                status = f"nonzero_exit({proc.returncode})"
        except subprocess.TimeoutExpired as exc:
            status = f"timeout_{self.timeout_s}s"
            stderr_text = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            output = f"TIMEOUT after {self.timeout_s}s\n{exc.stdout or ''}\n{stderr_text}"
        except Exception as exc:  # noqa: BLE001
            status = f"runner_error({type(exc).__name__})"
            output = f"runner error: {type(exc).__name__}: {exc}"

        # Persist the script's stdout/stderr so a failed iteration is debuggable
        # (and so the experiment run keeps a per-iteration audit trail).
        try:
            (rundir / "exec_log.txt").write_text(output)
        except Exception:  # noqa: BLE001 — logging must never break the run
            pass

        wrote_sub = sub.exists()

        # Score the run. PREFER AIDE-style independent extraction (read the output,
        # return the raw metric + buggy flag); fall back to the SCORE= sentinel only
        # if no extractor is configured or it errored. A buggy/None metric => None
        # (the controller treats a non-finite score as a failed, non-plateauing run).
        extractor_note = ""
        extractor_failed = False
        if self.extractor is not None:
            try:
                metric, lower_better, is_bug, ex_txt = self.extractor.extract(self.eval_desc, output)
            except Exception as exc:  # transport/gateway death, NOT the solution's fault
                # Fail THIS ITERATION only (loud), never the whole run — earlier
                # iterations' best submission must still get graded at the end.
                extractor_failed = True
                metric, lower_better, is_bug = None, None, False
                ex_txt = f"EXTRACTOR TRANSPORT FAILURE: {type(exc).__name__}: {exc}"
                status = f"extractor_error({type(exc).__name__})" if status == "ok" else status
                print(f"[continual] iter {iteration}: metric-extractor call failed "
                      f"({type(exc).__name__}: {exc}) — iteration scored as failed, "
                      f"run continues", file=sys.stderr, flush=True)
            extractor_note = (f" | extractor: bug={is_bug} metric={metric} "
                              f"lower_better={lower_better}"
                              + (" TRANSPORT-FAILED" if extractor_failed else ""))
            # Persist the extractor's INDEPENDENT verdict (auditable trail — the
            # score the loop actually used did not come from the script's self-report).
            try:
                import json as _json
                (rundir / "extractor.json").write_text(_json.dumps(
                    {"metric": metric, "lower_better": lower_better,
                     "is_bug": is_bug, "raw": ex_txt}, indent=2))
            except Exception:  # noqa: BLE001
                pass
            if is_bug or metric is None:
                score = None
            else:
                score = metric
        else:
            # No extractor: trust the SCORE= sentinel (legacy path).
            score = scoring.resolve_score(str(rundir), output)

        # Build the feedback the NEXT proposal will see, so the loop self-corrects
        # like AIDE does (exec outcome -> model). Surface the failure mode crisply
        # plus a stderr tail; on success, confirm the score + submission.
        if status == "ok" and score is not None and wrote_sub:
            self.last_exec_feedback = (
                f"Iter {iteration}: ran OK, metric={score}, submission.csv written.{extractor_note}"
            )
        else:
            why = []
            if status != "ok":
                why.append(f"execution {status}")
            if score is None:
                why.append("metric-extractor call failed (infrastructure, not the code)"
                           if extractor_failed else "no valid SCORE= line printed")
            if not wrote_sub:
                why.append("no submission.csv written")
            tail = (stderr_text or output)[-800:].strip()
            self.last_exec_feedback = (
                f"Iter {iteration} FAILED ({'; '.join(why) or 'unknown'}). "
                f"Keep the next solution FASTER and SIMPLER (must finish well under "
                f"{self.timeout_s}s), and fix the error below. Last error/output tail:\n{tail}"
            )

        self.last_trace = {
            "runner": "car_local_exec", "iteration": iteration,
            "script": str(script), "score": score, "status": status,
            "wrote_submission": wrote_sub, "extractor": extractor_note.strip(" |"),
            "output": output[-4000:],  # tail, for the trace window
        }
        return score, output

    def snapshot_if_best(self, iteration: int, is_better: bool) -> None:
        """Copy this iteration's submission.csv into car_best/ when it's the new
        incumbent, so the adapter grades the best submission produced."""
        if not is_better:
            return
        src = self.iters_dir / f"iter_{iteration:03d}" / "submission.csv"
        if src.is_file():
            self.best_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, self.best_dir / "submission.csv")


class ContinualAdapter(AutoResearchAdapter):
    """Drive car's HillClimber on an MLE-bench task with an LLM proposer that
    writes full solution scripts, executed locally on the box's GPU."""

    name = "continual"

    def __init__(self, steps: int = 10, backend: str | None = None, model: str | None = None):
        self.steps = int(os.environ.get("CAR_STEPS", steps))
        self.backend = backend or os.environ.get("ARBENCH_LLM_BACKEND", "litellm")
        self.model = model

    def prepare(self, task: Task, workspace: Path) -> None:
        # Reuse the same backend env wiring as AIDE (LiteLLM/Kimi by default).
        configure_aide_env(self.backend, model=self.model)
        if not os.environ.get("OPENAI_BASE_URL"):
            raise RuntimeError(
                "OPENAI_BASE_URL is unset; the OpenAI-compatible proposer would "
                "mis-route a non-OpenAI model. Set the backend env first."
            )

    def run(self, task: Task, workspace: Path) -> Path:
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        from continual_auto_research import HillClimber
        from continual_auto_research.core.proposers import OpenAICompatProposer

        backend = resolve_backend(self.backend, model=self.model)
        proposer = OpenAICompatProposer(
            model=backend.model,
            base_url=backend.base_url,
            api_key=backend.api_key,
            system=_CAR_MLE_SYSTEM,
        )

        # Fold the task into the proposer context so the LLM sees goal + data dir.
        lower_better = bool(task.metadata.get("is_lower_better"))
        direction_line = (
            "OPTIMIZATION DIRECTION: LOWER SCORE is better (the loop minimises)."
            if lower_better else
            "OPTIMIZATION DIRECTION: HIGHER SCORE is better (the loop maximises)."
        ) + " Print SCORE= as the RAW metric in this direction — do not invert it.\n"
        # Modality hint. The system prompt steers toward simple/GBM/linear models,
        # which is right for tabular but makes the proposer punt on IMAGE tasks
        # (it wrote a constant 0.5 submission for aerial-cactus rather than a CNN).
        # When the data looks like images, override that steer with an explicit
        # "train a small CNN on the GPU" instruction.
        modality_line = ""
        if _looks_like_image_task(task):
            modality_line = (
                "MODALITY: this is an IMAGE classification task. The 'prefer a simple "
                "GBM/linear model' guidance does NOT apply — a constant/trivial "
                "submission scores at chance. Train a small convolutional network "
                "(e.g. a compact CNN or a lightweight pretrained backbone like "
                "resnet18) on the GPU with torch/torchvision: read the image files, "
                "use a DataLoader, train for a few epochs with early data, and write "
                "calibrated probabilities to submission.csv. Keep it within the "
                "runtime budget (small input size, few epochs) but it MUST actually "
                "train on the images.\n"
                "DATA NOTE: image files are often shipped as ZIP archives (e.g. "
                "train.zip / test.zip) in the data directory and are NOT pre-extracted. "
                "Check for .zip files and extract them (zipfile) into a writable temp "
                "dir under the CURRENT working directory before reading images; do not "
                "assume loose .jpg/.png files already exist on disk.\n"
            )
        task_preamble = (
            f"TASK: {task.task_id}\n"
            f"DATA DIRECTORY (read inputs from here): {task.data_dir}\n"
            f"EVALUATION: {task.eval}\n"
            f"{direction_line}"
            f"{modality_line}\n"
            f"{task.goal}\n\n"
            "----- hill-climbing context (incumbent + recent attempts) -----\n"
        )

        direction = "min" if task.metadata.get("is_lower_better") else "max"
        # AIDE-style independent metric extraction (same backend as the proposer),
        # unless disabled via CAR_DISABLE_EXTRACTOR=1 (falls back to the SCORE= sentinel).
        extractor = None
        if os.environ.get("CAR_DISABLE_EXTRACTOR") != "1":
            extractor = _MetricExtractor(backend.model, backend.base_url, backend.api_key)
        runner = _LocalExecRunner(
            workspace, task.data_dir, extractor=extractor, eval_desc=task.eval,
        )

        # Tell the proposer the runtime ceiling (AIDE does the same) so it self-
        # limits code complexity rather than writing something that times out.
        budget_line = (
            f"RUNTIME BUDGET: each solution must finish well under "
            f"{runner.timeout_s} seconds or it is killed and scored as failed.\n\n"
        )

        def propose(context: str) -> str:
            # Prepend the previous iteration's execution outcome so the proposer
            # can fix failures/timeouts — the feedback loop a bare hill-climber's
            # score-only context lacks, and AIDE has via its feedback model.
            fb = runner.last_exec_feedback
            fb_block = (f"----- previous attempt's execution result -----\n{fb}\n\n"
                        if fb else "")
            full = task_preamble + budget_line + fb_block + (context or "")
            t0 = time.monotonic()
            out = proposer(full)
            latency = time.monotonic() - t0
            # expose the LLM I/O for the trace window
            lt = getattr(proposer, "last_trace", None) or {}
            propose.last_trace = lt or None
            # Telemetry: record the proposer call into the run's LLM trace, so
            # the continual arm's run.json carries n_llm_calls/tokens/cost like
            # the AIDE arm does (token counts come from the proposer's streamed
            # usage chunk; 0 if the CAR version doesn't surface usage yet).
            record_llm_call(
                model=backend.model, role="code",
                prompt_tokens=lt.get("prompt_tokens") or 0,
                completion_tokens=lt.get("completion_tokens") or 0,
                latency_s=lt.get("latency_s") or latency,
                system=_CAR_MLE_SYSTEM, user=full, response=out,
                extra={"source": "car_proposer"},
            )
            return out
        propose.last_trace = None

        hc = HillClimber(propose=propose, runner=runner, direction=direction)

        dest = task.submission_path(workspace)
        try:
            # Stream so we can snapshot the best submission as the incumbent changes.
            # NOTE: car's scored event uses the key "improved" (a separate "accepted"
            # event also fires, but the scored event is where iteration+improved meet).
            best_iter = None
            for ev in hc.stream(max_iter=self.steps, patience=self.steps):
                if ev.get("type") == "scored":
                    improved = bool(ev.get("improved"))
                    runner.snapshot_if_best(ev.get("iteration"), improved)
                    if improved:
                        best_iter = ev.get("iteration")
        finally:
            # ALWAYS place the best submission where the grader expects it — even
            # if the loop died mid-run, the best iteration so far is gradeable
            # (the runner grades dest despite an adapter error). Prefer the
            # best-snapshot; else fall back to the most recent iteration's file.
            dest.parent.mkdir(parents=True, exist_ok=True)
            best_sub = runner.best_dir / "submission.csv"
            chosen = best_sub if best_sub.is_file() else _latest_submission(runner.iters_dir)
            if chosen is not None and chosen.is_file():
                shutil.copyfile(chosen, dest)
        # If nothing was produced, return dest anyway (missing file -> grader marks
        # it invalid, which is the correct "the system failed to solve it" outcome).
        return dest


def _latest_submission(iters_dir: Path) -> Optional[Path]:
    subs = [p for p in iters_dir.glob("iter_*/submission.csv") if p.is_file()]
    if not subs:
        return None
    return max(subs, key=lambda p: p.stat().st_mtime)


register_adapter("continual", ContinualAdapter)

