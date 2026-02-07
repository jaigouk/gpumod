"""Anthropic Messages API backend via httpx."""

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


_DEFAULT_BASE_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicBackend(LLMBackend):
    """Anthropic Messages API backend.

    Sends requests to ``{base_url}/v1/messages`` using httpx.
    Authenticates via the ``x-api-key`` header with the configured API key.

    The API key is read from ``settings.llm_api_key`` (:class:`SecretStr`)
    and is **never** included in error messages (SEC-L3).
    """

    def __init__(self, settings: Any) -> None:
        self._model: str = settings.llm_model
        self.base_url: str = settings.llm_base_url or _DEFAULT_BASE_URL
        self._api_key: str | None = (
            settings.llm_api_key.get_secret_value() if settings.llm_api_key is not None else None
        )
        self._client: httpx.AsyncClient | None = None

    def _get_headers(self) -> dict[str, str]:
        """Build request headers with Anthropic auth."""
        headers: dict[str, str] = {
            "content-type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        if self._api_key is not None:
            headers["x-api-key"] = self._api_key
        return headers

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Send a message request to the Anthropic API.

        Parameters
        ----------
        prompt:
            The user message content.
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
            messages: list[dict[str, str]] = [
                {"role": "user", "content": prompt},
            ]

            payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": 4096,
            }
            if system is not None:
                payload["system"] = system

            url = f"{self.base_url}/v1/messages"
            headers = self._get_headers()

            try:
                client = self._client or httpx.AsyncClient(timeout=LLM_TIMEOUT)
                should_close = self._client is None
                try:
                    response = await client.post(url, json=payload, headers=headers)
                finally:
                    if should_close:
                        await client.aclose()

                if response.status_code >= 400:
                    msg = f"HTTP {response.status_code} error from Anthropic API"
                    raise LLMResponseError(msg)

                validate_content_type(response)

                data = response.json()
                content_str = data["content"][0]["text"]

            except LLMResponseError:
                raise
            except httpx.TimeoutException:
                msg = "Timeout connecting to Anthropic API"
                raise LLMResponseError(msg) from None
            except httpx.ConnectError:
                msg = "Connection error: could not reach Anthropic API"
                raise LLMResponseError(msg) from None
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                msg = f"Failed to communicate with Anthropic API: {type(exc).__name__}"
                raise LLMResponseError(msg) from None

            return safe_json_loads(content_str)

        try:
            return await asyncio.wait_for(_do_generate(), timeout=LLM_TOTAL_TIMEOUT)
        except TimeoutError:
            msg = "Total request lifecycle timeout exceeded"
            raise LLMResponseError(msg) from None
