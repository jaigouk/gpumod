"""LlamaCppClient - adapter for llama.cpp OpenAI-compatible API.

Provides a concrete LLMClient implementation that talks to llama.cpp's
OpenAI-compatible API endpoint.

Usage:
    from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient

    client = LlamaCppClient(base_url="http://localhost:8080")
    response = await client.generate("Write a function", temperature=0.6)
"""

from __future__ import annotations

from typing import Any

import httpx


class LlamaCppClient:
    """Client for llama.cpp OpenAI-compatible API.

    Parameters
    ----------
    base_url:
        Base URL of the llama.cpp server (e.g., "http://localhost:8080").
    timeout:
        Request timeout in seconds. Default 120s for long generations.
    """

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self.last_timing: dict[str, Any] | None = None

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt.

        Parameters
        ----------
        prompt:
            The user prompt to send to the model.
        **kwargs:
            Sampler parameters (temperature, top_p, top_k, etc.)

        Returns
        -------
        str:
            The model's text response.

        Raises
        ------
        ConnectionError:
            If the server is not reachable.
        TimeoutError:
            If the request times out.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

        url = f"{self.base_url}/v1/chat/completions"

        request_body: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }

        try:
            response = await self._client.post(url, json=request_body)
        except httpx.ConnectError as e:
            msg = f"Cannot connect to llama.cpp server at {self.base_url}. Is it running?"
            raise ConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request to {self.base_url} timed out after {self.timeout}s"
            raise TimeoutError(msg) from e

        # Parse response
        data: dict[str, Any] = response.json()

        # Extract timing from response body (preferred) or headers (fallback)
        self._extract_timing(data, response.headers)

        choices: list[dict[str, Any]] = data.get("choices", [])
        if not choices:
            return ""

        message: dict[str, Any] = choices[0].get("message", {})
        content: str = message.get("content", "")
        return content

    def _extract_timing(
        self,
        data: dict[str, Any],
        headers: httpx.Headers | dict[str, str],
    ) -> None:
        """Extract timing from response body ``timings``/``usage`` fields.

        Falls back to the ``X-Llama-Timings`` header when the body
        fields are absent (older llama.cpp builds).
        """
        body_timings: dict[str, Any] | None = data.get("timings")
        usage: dict[str, Any] | None = data.get("usage")

        if body_timings:
            # MTP (Multi-Token Prediction) variants emit `draft_n` and
            # `draft_n_accepted` in the timings block. Surface them as None
            # for non-MTP runs so callers can distinguish absence from zero.
            self.last_timing = {
                "prompt_tokens": body_timings.get("prompt_n", 0),
                "generated_tokens": body_timings.get("predicted_n", 0),
                "prompt_ms": body_timings.get("prompt_ms", 0.0),
                "generation_ms": body_timings.get("predicted_ms", 0.0),
                "draft_n": body_timings.get("draft_n"),
                "draft_n_accepted": body_timings.get("draft_n_accepted"),
            }
            return

        if usage:
            self.last_timing = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "generated_tokens": usage.get("completion_tokens", 0),
                "prompt_ms": 0.0,
                "generation_ms": 0.0,
            }
            return

        # Legacy header fallback
        timing_header = headers.get("X-Llama-Timings")
        if not timing_header:
            self.last_timing = None
            return

        timing: dict[str, Any] = {}
        for part in timing_header.split(";"):
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "prompt_n":
                    timing["prompt_tokens"] = int(value)
                elif key == "predicted_n":
                    timing["generated_tokens"] = int(value)
                elif key == "prompt_ms":
                    timing["prompt_ms"] = float(value)
                elif key == "predicted_ms":
                    timing["generation_ms"] = float(value)

        self.last_timing = timing or None

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
