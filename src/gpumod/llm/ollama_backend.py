"""Ollama local API backend via httpx."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx

from gpumod.llm.base import (
    LLM_TIMEOUT,
    LLM_TOTAL_TIMEOUT,
    LLMBackend,
    LLMResponseError,
    safe_json_loads,
    validate_content_type,
)

if TYPE_CHECKING:
    from pydantic import BaseModel


_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaBackend(LLMBackend):
    """Ollama local API backend.

    Sends requests to ``{base_url}/api/generate`` using httpx.
    No API key is required for Ollama.

    The base URL can be customized via ``settings.llm_base_url`` to
    point to a remote Ollama instance.
    """

    def __init__(self, settings: Any) -> None:
        self._model: str = settings.llm_model
        self.base_url: str = settings.llm_base_url or _DEFAULT_BASE_URL
        self._client: httpx.AsyncClient | None = None

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Send a generation request to the Ollama API.

        Parameters
        ----------
        prompt:
            The user prompt text.
        system:
            Optional system prompt.
        response_schema:
            Optional Pydantic model (currently unused, reserved for
            structured output support).

        Returns
        -------
        dict[str, Any]
            The parsed JSON content from the LLM response.

        Raises
        ------
        LLMResponseError
            On HTTP errors, timeouts, connection failures, or
            unparseable responses.
        """

        async def _do_generate() -> dict[str, Any]:
            payload: dict[str, Any] = {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
            if system is not None:
                payload["system"] = system

            url = f"{self.base_url}/api/generate"
            headers: dict[str, str] = {"content-type": "application/json"}

            try:
                client = self._client or httpx.AsyncClient(timeout=LLM_TIMEOUT)
                should_close = self._client is None
                try:
                    response = await client.post(url, json=payload, headers=headers)
                finally:
                    if should_close:
                        await client.aclose()

                if response.status_code >= 400:
                    msg = f"HTTP {response.status_code} error from Ollama API"
                    raise LLMResponseError(msg)

                validate_content_type(response)

                data = response.json()
                content_str = data["response"]

            except LLMResponseError:
                raise
            except httpx.TimeoutException:
                msg = "Timeout connecting to Ollama API"
                raise LLMResponseError(msg) from None
            except httpx.ConnectError:
                msg = "Connection error: could not reach Ollama API"
                raise LLMResponseError(msg) from None
            except (httpx.HTTPError, KeyError) as exc:
                msg = f"Failed to communicate with Ollama API: {type(exc).__name__}"
                raise LLMResponseError(msg) from None

            return safe_json_loads(content_str)

        try:
            return await asyncio.wait_for(_do_generate(), timeout=LLM_TOTAL_TIMEOUT)
        except TimeoutError:
            msg = "Total request lifecycle timeout exceeded"
            raise LLMResponseError(msg) from None
