"""Full-run capture: everything needed to reconstruct and cost a benchmark run.

A run produces, inside its workspace, a self-contained bundle:

    <workspace>/
      run.json                 # RunTrace: metadata + aggregates (the index)
      llm_calls.jsonl          # one record per LLM call (prompt, response, tokens, timing)
      submission.csv           # the graded artifact
      run.log                  # stdout/stderr of the adapter
      artifacts/               # copied AIDE journal, best solution, tree plot, etc.

The per-call JSONL is written by the LLM backend during the run (the AIDE fork
appends to $ARBENCH_LLM_TRACE); arbench reads it back to aggregate tokens, cost,
and timing. Everything here is plain JSON so a run is fully reconstructable
offline, with no live services.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from arbench.llm.pricing import estimate_cost


LLM_TRACE_ENV = "ARBENCH_LLM_TRACE"   # path the backend appends call records to


# ──────────────────────────────────────────────────────────────────────────
# Per-call record (written by the backend, one JSON object per line)
# ──────────────────────────────────────────────────────────────────────────
def record_llm_call(
    *,
    model: str,
    role: str,                 # "code" | "feedback" | "report" | "other"
    prompt_tokens: int,
    completion_tokens: int,
    latency_s: float,
    system: Optional[str] = None,
    user: Optional[str] = None,
    response: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Append one LLM-call record to the JSONL named by $ARBENCH_LLM_TRACE.
    No-op if the env var is unset (so the backend is safe outside arbench)."""
    path = os.environ.get(LLM_TRACE_ENV)
    if not path:
        return
    rec = {
        "ts": time.time(),
        "model": model,
        "role": role,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "latency_s": round(float(latency_s or 0.0), 4),
        # Full text for reconstruction. Truncation is the caller's choice; we
        # store whatever is passed (set ARBENCH_TRACE_FULL=0 in the backend to omit).
        "system": system,
        "user": user,
        "response": response,
        "extra": extra or {},
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass  # tracing must never break a run


def load_llm_calls(path: Path) -> list[dict]:
    calls = []
    if not Path(path).exists():
        return calls
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    calls.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return calls


# ──────────────────────────────────────────────────────────────────────────
# Run metadata + the aggregated trace
# ──────────────────────────────────────────────────────────────────────────
def _git_sha(repo_dir: str | Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class RunMeta:
    adapter: str
    benchmark: str
    task_id: str
    model: str
    backend: str
    steps: Optional[int]
    started_at: float
    finished_at: Optional[float] = None
    wall_clock_s: Optional[float] = None
    host: str = field(default_factory=platform.node)
    python: str = field(default_factory=platform.python_version)
    arbench_sha: Optional[str] = None
    adapter_repo_sha: Optional[str] = None   # e.g. the AIDE fork HEAD
    env_snapshot: dict[str, str] = field(default_factory=dict)


@dataclass
class RunTrace:
    meta: RunMeta
    score: dict[str, Any]
    # aggregates derived from llm_calls.jsonl
    n_llm_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_llm_latency_s: float = 0.0
    cost: dict[str, Any] = field(default_factory=dict)
    per_role: dict[str, Any] = field(default_factory=dict)
    submission_path: Optional[str] = None
    artifacts: list[str] = field(default_factory=list)
    adapter_error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def write(self, workspace: Path) -> Path:
        out = Path(workspace) / "run.json"
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return out


def build_trace(
    *,
    meta: RunMeta,
    score: dict,
    workspace: Path,
    submission_path: Optional[Path],
    adapter_error: Optional[str],
    arbench_repo: str | Path,
    adapter_repo: str | Path | None = None,
) -> RunTrace:
    """Aggregate llm_calls.jsonl + metadata into a RunTrace and persist run.json."""
    workspace = Path(workspace)
    calls = load_llm_calls(workspace / "llm_calls.jsonl")

    pt = sum(c.get("prompt_tokens", 0) for c in calls)
    ct = sum(c.get("completion_tokens", 0) for c in calls)
    lat = sum(c.get("latency_s", 0.0) for c in calls)

    # cost: estimate per model, sum
    by_model: dict[str, list[int]] = {}
    for c in calls:
        m = c.get("model", meta.model)
        by_model.setdefault(m, [0, 0])
        by_model[m][0] += c.get("prompt_tokens", 0)
        by_model[m][1] += c.get("completion_tokens", 0)
    total_usd = 0.0
    model_costs = []
    for m, (p, comp) in by_model.items():
        ce = estimate_cost(m, p, comp)
        total_usd += ce.total_usd
        model_costs.append(ce.to_dict())

    # per-role aggregation
    per_role: dict[str, dict] = {}
    for c in calls:
        r = c.get("role", "other")
        slot = per_role.setdefault(r, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        slot["calls"] += 1
        slot["prompt_tokens"] += c.get("prompt_tokens", 0)
        slot["completion_tokens"] += c.get("completion_tokens", 0)

    meta.arbench_sha = _git_sha(arbench_repo)
    if adapter_repo:
        meta.adapter_repo_sha = _git_sha(adapter_repo)

    trace = RunTrace(
        meta=meta,
        score=score,
        n_llm_calls=len(calls),
        total_prompt_tokens=pt,
        total_completion_tokens=ct,
        total_llm_latency_s=round(lat, 3),
        cost={"total_usd": round(total_usd, 6), "by_model": model_costs,
              "basis": "openrouter-equivalent"},
        per_role=per_role,
        submission_path=str(submission_path) if submission_path else None,
        artifacts=sorted(p.name for p in (workspace / "artifacts").glob("*")) if (workspace / "artifacts").exists() else [],
        adapter_error=adapter_error,
    )
    trace.write(workspace)
    return trace
