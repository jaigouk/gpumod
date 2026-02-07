"""Abstract base class for LLM backends."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from pydantic import BaseModel


class LLMResponseError(Exception):
    """Raised when an LLM returns invalid or unparseable output.

    This exception is also used to wrap transport errors (HTTP failures,
    timeouts, connection errors) so callers have a single error type to
    handle.  Sensitive data such as API keys is **never** included in
    the message.
    """


class LLMBackend(ABC):
    """Abstract interface for LLM provider backends.

    All concrete backends must implement the :meth:`generate` method,
    which sends a prompt to the LLM and returns the parsed JSON response
    as a Python dict.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Send a prompt to the LLM and return parsed JSON response.

        Parameters
        ----------
        prompt:
            The user prompt text.
        system:
            Optional system prompt for the LLM.
        response_schema:
            Optional Pydantic model class for response validation.

        Returns
        -------
        dict[str, Any]
            The parsed JSON response from the LLM.

        Raises
        ------
        LLMResponseError
            If the LLM response is invalid, unparseable, or if a
            transport error occurs.
        """


def safe_json_loads(
    data: str,
    *,
    max_size: int = 1_048_576,
    max_depth: int = 50,
) -> dict[str, Any]:
    """Parse JSON with size and depth limits.

    Parameters
    ----------
    data:
        The JSON string to parse.
    max_size:
        Maximum allowed size in bytes. Default 1MB.
    max_depth:
        Maximum allowed nesting depth. Default 50.

    Returns
    -------
    dict[str, Any]
        The parsed JSON object.

    Raises
    ------
    LLMResponseError
        If the data exceeds size or depth limits, or is not valid JSON.
    """
    if len(data.encode("utf-8")) > max_size:
        msg = f"JSON payload exceeds maximum size of {max_size} bytes"
        raise LLMResponseError(msg)

    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, TypeError) as exc:
        msg = "Failed to parse LLM response as JSON"
        raise LLMResponseError(msg) from exc

    def _check_depth(obj: object, current_depth: int = 0) -> None:
        if current_depth > max_depth:
            msg = f"JSON nesting depth exceeds maximum of {max_depth}"
            raise LLMResponseError(msg)
        if isinstance(obj, dict):
            for v in obj.values():
                _check_depth(v, current_depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                _check_depth(item, current_depth + 1)

    _check_depth(parsed)

    if not isinstance(parsed, dict):
        msg = "Expected JSON object at top level"
        raise LLMResponseError(msg)

    return parsed


# Explicit per-phase timeouts (SEC-N1)
LLM_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=30.0,
    write=10.0,
    pool=5.0,
)

# Total lifecycle timeout for a complete LLM request (SEC-N1)
LLM_TOTAL_TIMEOUT: float = 90.0


def validate_content_type(response: httpx.Response) -> None:
    """Validate that the HTTP response has a JSON content type (SEC-N2).

    Parameters
    ----------
    response:
        The httpx Response to validate.

    Raises
    ------
    LLMResponseError
        If the Content-Type is not application/json.
    """
    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        msg = f"Expected application/json Content-Type, got {content_type!r}"
        raise LLMResponseError(msg)
