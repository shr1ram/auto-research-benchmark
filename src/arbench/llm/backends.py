"""Pluggable LLM backends for driving autoresearch systems.

Three backends, all reachable the same way because most OpenAI-API
clients route to an OpenAI-compatible endpoint whenever OPENAI_BASE_URL is set
and the model name is not a gpt-/claude-/gemini- prefix:

  - "litellm"  : the team LiteLLM proxy (litellm.yangtzeailab.com/v1).
                 DEFAULT model = Kimi-K2.6 (per project decision), also serves
                 deepseek-v4-flash.
  - "claude_p" : Claude via an OpenAI-compatible base URL (e.g. claude-code
                 proxy / Anthropic-compatible gateway). Model name keeps a
                 "claude-" prefix so a native Anthropic path can also be used.
  - "local"    : local Ollama on the UCL GPU box (127.0.0.1:11435/v1), e.g.
                 qwen3.6-64k:27b-q4_K_M.

`configure_llm_env` sets the env vars OpenAI-compatible clients read;
`resolve_backend` returns the
resolved (base_url, model, api_key) for logging / programmatic use.

Secrets are read from the environment (loaded from the box's env-profiles), never
hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


# The team LiteLLM proxy. Confirmed live model ids: deepseek-v4-flash, Kimi-K2.6.
LITELLM_BASE_URL = "https://litellm.yangtzeailab.com/v1"
KIMI_MODEL = "Kimi-K2.6"          # project default
DEEPSEEK_MODEL = "deepseek-v4-flash"

# Local Ollama on the UCL GPU box.
LOCAL_BASE_URL = "http://127.0.0.1:11435/v1"
LOCAL_MODEL = "qwen3.6-64k:27b-q4_K_M"


@dataclass
class Backend:
    name: str
    base_url: str
    model: str
    api_key: str
    # The model used for lighter feedback/report calls (default = model).
    feedback_model: str

    def env(self) -> dict[str, str]:
        """The environment an OpenAI-compatible client reads."""
        return {
            "OPENAI_BASE_URL": self.base_url,
            "OPENAI_API_KEY": self.api_key,
            # Mirror onto names other clients use, harmless if unread.
            "ARBENCH_LLM_MODEL": self.model,
            "ARBENCH_LLM_FEEDBACK_MODEL": self.feedback_model,
        }


def _key_from_env(*names: str, default: str = "sk-noauth") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def resolve_backend(backend: str, model: str | None = None) -> Backend:
    backend = (backend or "litellm").lower()

    if backend == "litellm":
        m = model or os.environ.get("ARBENCH_LLM_MODEL") or KIMI_MODEL
        key = _key_from_env("CUSTOM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
        base = os.environ.get("DEFAULT_API_BASE_URL", LITELLM_BASE_URL)
        return Backend("litellm", base, m, key, feedback_model=m)

    if backend == "claude_p":
        # OpenAI-compatible Claude gateway (claude-code / Anthropic-compatible).
        m = model or os.environ.get("CLAUDE_P_MODEL") or "claude-sonnet-4-6"
        key = _key_from_env("ANTHROPIC_API_KEY", "CLAUDE_P_API_KEY", "OPENAI_API_KEY")
        base = os.environ.get("CLAUDE_P_BASE_URL", "http://127.0.0.1:4000/v1")
        return Backend("claude_p", base, m, key, feedback_model=m)

    if backend == "local":
        m = model or os.environ.get("LOCAL_LLM_MODEL") or LOCAL_MODEL
        base = os.environ.get("LOCAL_BASE_URL", LOCAL_BASE_URL)
        return Backend("local", base, m, "ollama", feedback_model=m)

    raise ValueError(
        f"unknown backend {backend!r}; choose from litellm | claude_p | local"
    )


def configure_llm_env(backend: str, model: str | None = None) -> Backend:
    """Set process env so the adapter's OpenAI-compatible LLM client targets
    `backend`. Returns the resolved Backend for logging."""
    b = resolve_backend(backend, model=model)
    os.environ.update(b.env())
    return b
