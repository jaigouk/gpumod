"""Tests for LlamaCppClient - llama.cpp OpenAI-compatible API adapter.

TDD Phase: RED - Write failing tests first.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# LlamaCppClient initialization tests
# ---------------------------------------------------------------------------


class TestLlamaCppClientInit:
    """Tests for LlamaCppClient initialization."""

    def test_creates_with_base_url(self) -> None:
        """Client accepts base_url for llama.cpp server."""
        from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient

        client = LlamaCppClient(base_url="http://localhost:8080")
        assert client.base_url == "http://localhost:8080"

    def test_default_timeout_is_120_seconds(self) -> None:
        """Default timeout is 120s for long generations."""
        from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient

        client = LlamaCppClient(base_url="http://localhost:8080")
        assert client.timeout == 120.0

    def test_accepts_custom_timeout(self) -> None:
        """Client accepts custom timeout."""
        from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient

        client = LlamaCppClient(base_url="http://localhost:8080", timeout=60.0)
        assert client.timeout == 60.0

    def test_strips_trailing_slash_from_base_url(self) -> None:
        """Base URL is normalized to remove trailing slash."""
        from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient

        client = LlamaCppClient(base_url="http://localhost:8080/")
        assert client.base_url == "http://localhost:8080"


# ---------------------------------------------------------------------------
# LlamaCppClient.generate() tests
# ---------------------------------------------------------------------------


class TestLlamaCppClientGenerate:
    """Tests for LlamaCppClient.generate() method."""

    @pytest.mark.asyncio
    async def test_generate_returns_string(self) -> None:
        """Generate returns the model's text response."""
        from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient

        client = LlamaCppClient(base_url="http://localhost:8080")

        # Mock the httpx client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "def hello(): pass"}}]
        }
        mock_response.headers = {}

        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=mock_response)

        result = await client.generate("Write a hello function")
        assert result == "def hello(): pass"

    @pytest.mark.asyncio
    async def test_generate_passes_sampler_kwargs(self) -> None:
        """Generate passes sampler kwargs to API."""
        from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient

        client = LlamaCppClient(base_url="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_response.headers = {}

        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=mock_response)

        await client.generate(
            "prompt",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )

        # Verify the API was called with correct parameters
        call_args = client._client.post.call_args
        fallback = call_args.args[1] if len(call_args.args) > 1 else {}
        request_body = call_args.kwargs.get("json", fallback)
        assert request_body.get("temperature") == 0.6
        assert request_body.get("top_p") == 0.95

    @pytest.mark.asyncio
    async def test_generate_uses_chat_completions_endpoint(self) -> None:
        """Generate calls /v1/chat/completions endpoint."""
        from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient

        client = LlamaCppClient(base_url="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_response.headers = {}

        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=mock_response)

        await client.generate("prompt")

        call_args = client._client.post.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
        assert "/v1/chat/completions" in url

    @pytest.mark.asyncio
    async def test_generate_formats_prompt_as_user_message(self) -> None:
        """Generate wraps prompt in chat message format."""
        from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient

        client = LlamaCppClient(base_url="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_response.headers = {}

        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=mock_response)

        await client.generate("Write code")

        call_args = client._client.post.call_args
        request_body = call_args.kwargs.get("json", {})
        messages = request_body.get("messages", [])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Write code"


# ---------------------------------------------------------------------------
# LlamaCppClient error handling tests
# ---------------------------------------------------------------------------


class TestLlamaCppClientErrors:
    """Tests for LlamaCppClient error handling."""

    @pytest.mark.asyncio
    async def test_raises_connection_error_when_server_not_running(self) -> None:
        """Raises ConnectionError with helpful message when server unreachable."""
        import httpx

        from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient

        client = LlamaCppClient(base_url="http://localhost:9999")

        client._client = AsyncMock()
        client._client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        with pytest.raises(ConnectionError) as exc_info:
            await client.generate("prompt")

        assert "localhost:9999" in str(exc_info.value)
        error_msg = str(exc_info.value).lower()
        assert "not running" in error_msg or "connect" in error_msg

    @pytest.mark.asyncio
    async def test_raises_timeout_error_on_timeout(self) -> None:
        """Raises TimeoutError when generation times out."""
        import httpx

        from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient

        client = LlamaCppClient(base_url="http://localhost:8080", timeout=1.0)

        client._client = AsyncMock()
        client._client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        with pytest.raises(TimeoutError):
            await client.generate("prompt")

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_empty_response(self) -> None:
        """Returns empty string when model returns no content."""
        from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient

        client = LlamaCppClient(base_url="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": ""}}]
        }
        mock_response.headers = {}

        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=mock_response)

        result = await client.generate("prompt")
        assert result == ""


# ---------------------------------------------------------------------------
# LlamaCppClient timing extraction tests
# ---------------------------------------------------------------------------


class TestLlamaCppClientTiming:
    """Tests for timing extraction from llama.cpp responses."""

    @pytest.mark.asyncio
    async def test_extracts_timing_from_headers(self) -> None:
        """Extracts timing info from X-Llama-Timings header."""
        from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient

        client = LlamaCppClient(base_url="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_response.headers = {
            "X-Llama-Timings": "prompt_n=100;prompt_ms=50.5;predicted_n=50;predicted_ms=250.0"
        }

        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=mock_response)

        await client.generate("prompt")

        assert client.last_timing is not None
        assert client.last_timing.get("prompt_tokens") == 100
        assert client.last_timing.get("generated_tokens") == 50

    @pytest.mark.asyncio
    async def test_timing_is_none_when_header_missing(self) -> None:
        """Timing is None when X-Llama-Timings header not present."""
        from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient

        client = LlamaCppClient(base_url="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_response.headers = {}

        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=mock_response)

        await client.generate("prompt")

        assert client.last_timing is None


# ---------------------------------------------------------------------------
# LlamaCppClient protocol compliance tests
# ---------------------------------------------------------------------------


class TestLlamaCppClientProtocol:
    """Tests that LlamaCppClient implements LLMClient protocol."""

    def test_implements_llm_client_protocol(self) -> None:
        """LlamaCppClient satisfies LLMClient protocol."""
        from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient
        from gpumod.benchmarks.qwen35.runner import LLMClient  # noqa: TC001

        client = LlamaCppClient(base_url="http://localhost:8080")

        # Protocol check - client should be usable where LLMClient is expected
        def accepts_llm_client(c: LLMClient) -> None:
            pass

        # This should not raise - if it does, protocol is not satisfied
        accepts_llm_client(client)  # type: ignore[arg-type]
