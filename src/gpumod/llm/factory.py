"""Factory function for creating LLM backend instances."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from gpumod.llm.anthropic_backend import AnthropicBackend
from gpumod.llm.ollama_backend import OllamaBackend
from gpumod.llm.openai_backend import OpenAIBackend

if TYPE_CHECKING:
    from gpumod.llm.base import LLMBackend

_BackendFactory = Callable[[Any], "LLMBackend"]

_BACKENDS: dict[str, _BackendFactory] = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "ollama": OllamaBackend,
}


def get_backend(settings: Any) -> LLMBackend:
    """Create an LLM backend instance based on the configured provider.

    Parameters
    ----------
    settings:
        A :class:`~gpumod.config.GpumodSettings` instance (or any object
        with compatible attributes).

    Returns
    -------
    LLMBackend
        The configured backend instance.

    Raises
    ------
    ValueError
        If ``settings.llm_backend`` is not a supported provider name.
    """
    backend_name: str = settings.llm_backend
    factory = _BACKENDS.get(backend_name)
    if factory is None:
        supported = ", ".join(sorted(_BACKENDS))
        msg = f"Unsupported LLM backend: {backend_name!r} (supported: {supported})"
        raise ValueError(msg)
    return factory(settings)
