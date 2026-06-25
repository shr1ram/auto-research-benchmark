"""Tests for the full-run capture: LLM-call trace, cost estimate, run bundle.

Uses a fake adapter that emits trace records exactly as the AIDE fork does
(appending to $ARBENCH_LLM_TRACE), so we exercise the whole capture path without
AIDE/mlebench installed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from arbench.core.task import Task
from arbench.core.result import Score
from arbench.core.adapter import AutoResearchAdapter
from arbench.core.benchmark import Benchmark
from arbench.core.runner import run_one
from arbench.core.trace import record_llm_call, LLM_TRACE_ENV
from arbench.llm.pricing import estimate_cost


class FakeBenchmark(Benchmark):
    name = "fake"

    def list_tasks(self):
        return ["t1"]

    def load_task(self, task_id):
        return Task(task_id=task_id, benchmark=self.name, goal="g", eval="e")

    def grade(self, task, submission_path):
        return Score(value=0.9, valid=True, is_higher_better=True)


class TracingAdapter(AutoResearchAdapter):
    """Mimics AIDE: makes 'LLM calls' that append trace records, then writes a
    submission. Uses the same env-var contract the real backend uses."""
    name = "tracer"
    model = "Kimi-K2.6"
    backend = "litellm"
    steps = 3

    def run(self, task, workspace):
        # two coder calls + one feedback call, like a small AIDE run
        record_llm_call(model="Kimi-K2.6", role="code",
                        prompt_tokens=1000, completion_tokens=2000, latency_s=12.5,
                        system="sys", user="draft a model", response="import pandas")
        record_llm_call(model="Kimi-K2.6", role="code",
                        prompt_tokens=1500, completion_tokens=2500, latency_s=18.0,
                        system="sys", user="improve it", response="import xgboost")
        record_llm_call(model="Kimi-K2.6", role="feedback",
                        prompt_tokens=500, completion_tokens=300, latency_s=4.0)
        p = task.submission_path(workspace)
        p.write_text("ok")
        return p


# ── pricing ─────────────────────────────────────────────────────────────────

def test_cost_estimate_kimi():
    ce = estimate_cost("Kimi-K2.6", 1_000_000, 1_000_000)
    assert ce.priced is True
    # 0.60 prompt + 2.50 completion per 1M
    assert abs(ce.total_usd - (0.60 + 2.50)) < 1e-9


def test_cost_estimate_unknown_model_flagged():
    ce = estimate_cost("some-random-model", 1000, 1000)
    assert ce.priced is False
    assert ce.total_usd > 0


# ── trace record sink ─────────────────────────────────────────────────────────

def test_record_is_noop_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv(LLM_TRACE_ENV, raising=False)
    record_llm_call(model="m", role="code", prompt_tokens=1, completion_tokens=1, latency_s=1.0)
    # nothing written anywhere — just shouldn't raise


def test_record_appends_jsonl(tmp_path, monkeypatch):
    p = tmp_path / "calls.jsonl"
    monkeypatch.setenv(LLM_TRACE_ENV, str(p))
    record_llm_call(model="m", role="code", prompt_tokens=10, completion_tokens=20, latency_s=1.0)
    record_llm_call(model="m", role="feedback", prompt_tokens=5, completion_tokens=5, latency_s=0.5)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["prompt_tokens"] == 10 and rec["role"] == "code"


# ── full run bundle ───────────────────────────────────────────────────────────

def test_run_produces_full_trace_bundle(tmp_path):
    ws = tmp_path / "run1"
    r = run_one(TracingAdapter(), FakeBenchmark(), "t1", ws, trace=True)
    assert r.score.valid

    # run.json exists with the aggregates
    run_json = ws / "run.json"
    assert run_json.exists()
    t = json.loads(run_json.read_text())
    assert t["n_llm_calls"] == 3
    assert t["total_prompt_tokens"] == 3000
    assert t["total_completion_tokens"] == 4800
    assert t["total_llm_latency_s"] == 34.5
    # cost computed at openrouter-equiv rates, > 0
    assert t["cost"]["total_usd"] > 0
    assert t["cost"]["basis"] == "openrouter-equivalent"
    # per-role split present
    assert t["per_role"]["code"]["calls"] == 2
    assert t["per_role"]["feedback"]["calls"] == 1
    # metadata for reconstruction
    assert t["meta"]["model"] == "Kimi-K2.6"
    assert t["meta"]["backend"] == "litellm"
    assert t["meta"]["wall_clock_s"] is not None
    assert "host" in t["meta"] and "python" in t["meta"]

    # the per-call JSONL is preserved
    assert (ws / "llm_calls.jsonl").exists()
    assert len((ws / "llm_calls.jsonl").read_text().strip().splitlines()) == 3


def test_trace_can_be_disabled(tmp_path):
    ws = tmp_path / "run2"
    run_one(TracingAdapter(), FakeBenchmark(), "t1", ws, trace=False)
    assert not (ws / "run.json").exists()
