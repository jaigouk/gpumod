"""Sampler configuration presets for chat-completion models.

Originally built for Qwen 3.5 (THINKING_CODING / NON_THINKING per
https://huggingface.co/Qwen/Qwen3.5-35B-A3B) and extended for Gemma 4
(GEMMA_CODING per https://huggingface.co/google/gemma-4-12B-it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SamplerConfig:
    """Immutable sampler configuration for LLM generation."""

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float = 1.0

    def to_dict(self, *, exclude_defaults: bool = False) -> dict[str, Any]:
        """Convert to dict for API calls.

        Args:
            exclude_defaults: If True, omit repetition_penalty when 1.0
        """
        d: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
        }
        if not (exclude_defaults and self.repetition_penalty == 1.0):
            d["repetition_penalty"] = self.repetition_penalty
        return d


# Qwen-recommended settings for thinking mode with precise coding tasks
THINKING_CODING = SamplerConfig(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
)

# Qwen-recommended settings for non-thinking (instruct) mode
NON_THINKING = SamplerConfig(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
)

# Google-recommended settings for Gemma 4 (all use cases, including coding).
# Per the google/gemma-4-12B-it model card the same sampler is recommended
# for instruction-following, coding, and reasoning. Differs from Qwen's
# THINKING_CODING (temp 0.6, top_k 20) — Gemma's thinking-mode is tuned for
# broader exploration during CoT and converges in the final answer.
#
# gpumod-eods: repetition_penalty=1.05 (NOT Google's default 1.0) — breaks
# degeneration loops at temp=1.0 without measurably suppressing legitimate
# code repetition. Observed in gpumod-h6gs run 2 iter_06 where the 12B Q8
# repeated "If __ename__ import requests" 6+ times until token budget was
# exhausted. 1.05 is the lowest non-trivial value; standard llama.cpp
# guidance for catching runaway repetition without quality cost.
GEMMA_CODING = SamplerConfig(
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.05,
)

# Registry of available configs
_CONFIGS: dict[str, SamplerConfig] = {
    "thinking_coding": THINKING_CODING,
    "non_thinking": NON_THINKING,
    "gemma_coding": GEMMA_CODING,
}


def get_config(name: str) -> SamplerConfig:
    """Get a sampler config by name.

    Args:
        name: Config name ("thinking_coding" or "non_thinking")

    Returns:
        The corresponding SamplerConfig

    Raises:
        ValueError: If name is not recognized
    """
    if name not in _CONFIGS:
        valid = ", ".join(_CONFIGS.keys())
        msg = f"Unknown config: {name}. Valid options: {valid}"
        raise ValueError(msg)
    return _CONFIGS[name]
