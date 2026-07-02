"""Unit tests for the continual adapter's pure helpers (no CAR install needed)."""
from __future__ import annotations

from pathlib import Path

from arbench.adapters.continual_adapter import _clean_child_env, _extract_code


def test_clean_child_env_strips_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 22")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-secret2")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("MY_SECRET", "shh")
    monkeypatch.setenv("ARBENCH_LLM_TRACE", "/somewhere/llm_calls.jsonl")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("OPENML_DATA_DIR", "/data/openml")

    env = _clean_child_env(tmp_path)

    for denied in ("SSH_AUTH_SOCK", "SSH_CONNECTION", "OPENAI_API_KEY",
                   "CUSTOM_API_KEY", "HF_TOKEN", "MY_SECRET", "ARBENCH_LLM_TRACE"):
        assert denied not in env
    # the things training code genuinely needs survive
    assert "PATH" in env
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["OPENML_DATA_DIR"] == "/data/openml"
    assert env["DATA_DIR"] == str(tmp_path)
    assert env["MLEBENCH_DATA_DIR"] == str(tmp_path)


def test_extract_code_fenced_block():
    text = "here you go\n```python\nprint('hi')\n```\ntrailing prose"
    assert _extract_code(text) == "print('hi')"
