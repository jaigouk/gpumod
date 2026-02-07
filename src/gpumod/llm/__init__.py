"""LLM backend abstraction for gpumod AI planning.

Provides a provider-agnostic interface for interacting with LLM APIs
(OpenAI, Anthropic, Ollama) with structured output validation and
security mitigations per SEC-L1 through SEC-L5.
"""

from __future__ import annotations

from gpumod.llm.base import LLMBackend, LLMResponseError
from gpumod.llm.factory import get_backend

__all__ = [
    "LLMBackend",
    "LLMResponseError",
    "get_backend",
]
