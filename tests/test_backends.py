"""Backend resolution tests — the LLM plumbing adapters ride on."""
from __future__ import annotations

import pytest

from arbench.llm.backends import resolve_backend, KIMI_MODEL, LITELLM_BASE_URL


def test_litellm_defaults_to_kimi(monkeypatch):
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-test")
    monkeypatch.delenv("ARBENCH_LLM_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_API_BASE_URL", raising=False)
    b = resolve_backend("litellm")
    assert b.model == KIMI_MODEL
    assert b.base_url == LITELLM_BASE_URL
    assert b.api_key == "sk-test"
    # OpenAI-compatible clients route off these env keys.
    env = b.env()
    assert env["OPENAI_BASE_URL"] == LITELLM_BASE_URL
    assert env["OPENAI_API_KEY"] == "sk-test"


def test_model_override(monkeypatch):
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-test")
    b = resolve_backend("litellm", model="deepseek-v4-flash")
    assert b.model == "deepseek-v4-flash"


def test_local_backend(monkeypatch):
    b = resolve_backend("local")
    assert "11435" in b.base_url
    assert b.api_key == "ollama"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        resolve_backend("nope")
