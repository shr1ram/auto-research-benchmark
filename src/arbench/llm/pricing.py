"""Cost estimation from token counts.

The team LiteLLM proxy is (effectively) free to us, so a raw bill isn't
meaningful. For comparability across systems and write-ups we report an
**OpenRouter-equivalent** cost: what the same token usage would cost at public
OpenRouter list prices for the model actually used. This makes baseline numbers
portable ("AIDE on RAOP ≈ $X of LLM at OpenRouter rates").

Prices are USD per 1M tokens, (prompt, completion). Update as list prices change;
unknown models fall back to a conservative default and are flagged.
"""
from __future__ import annotations

from dataclasses import dataclass

# USD per 1,000,000 tokens: (prompt, completion). OpenRouter list prices.
# Keyed by a normalised model name (lowercased, provider-stripped where obvious).
_PRICES: dict[str, tuple[float, float]] = {
    # The models we actually serve via the LiteLLM proxy, priced at their
    # OpenRouter-equivalent public rates (Moonshot Kimi, DeepSeek).
    "kimi-k2.6": (0.60, 2.50),
    "kimi-k2": (0.60, 2.50),
    "deepseek-v4-flash": (0.27, 1.10),
    "deepseek-chat": (0.27, 1.10),
    # Common references people compare against.
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "o4-mini": (1.10, 4.40),
    "claude-sonnet-4-6": (3.00, 15.00),
    "qwen3.6-64k:27b-q4_k_m": (0.0, 0.0),   # local Ollama: no marginal $
}

# Used when a model isn't in the table — flagged in the result so it's visible.
_DEFAULT = (1.00, 3.00)


def _normalise(model: str) -> str:
    m = (model or "").strip().lower()
    # strip an "openrouter/" or "provider/" prefix if present
    if "/" in m:
        m = m.split("/", 1)[1]
    return m


@dataclass
class CostEstimate:
    model: str
    prompt_tokens: int
    completion_tokens: int
    prompt_usd: float
    completion_usd: float
    total_usd: float
    priced: bool   # False if we fell back to the default table entry

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_usd": round(self.prompt_usd, 6),
            "completion_usd": round(self.completion_usd, 6),
            "total_usd": round(self.total_usd, 6),
            "priced": self.priced,
            "basis": "openrouter-equivalent",
        }


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> CostEstimate:
    key = _normalise(model)
    priced = key in _PRICES
    p_rate, c_rate = _PRICES.get(key, _DEFAULT)
    p_usd = prompt_tokens / 1_000_000 * p_rate
    c_usd = completion_tokens / 1_000_000 * c_rate
    return CostEstimate(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_usd=p_usd,
        completion_usd=c_usd,
        total_usd=p_usd + c_usd,
        priced=priced,
    )
