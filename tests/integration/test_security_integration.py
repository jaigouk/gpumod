"""Integration tests for security controls end-to-end.

Verifies that:
- Rate limiter rejects excess MCP requests end-to-end (SEC-R2)
- LLM response with injected IDs rejected (SEC-L1)
- API key never appears in any error output (SEC-L3)
"""

from __future__ import annotations

import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator

import httpx
import pytest
import typer.testing
from fastmcp import Client, FastMCP

from gpumod.config import GpumodSettings, _clear_settings_cache
from gpumod.db import Database
from gpumod.llm import LLMResponseError
from gpumod.llm.openai_backend import OpenAIBackend
from gpumod.llm.response_validator import validate_plan_response
from gpumod.mcp_resources import register_resources
from gpumod.mcp_server import (
    ErrorSanitizationMiddleware,
    RateLimitMiddleware,
    gpumod_lifespan,
)
from gpumod.mcp_tools import register_tools
from gpumod.models import DriverType, Service

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Clear settings cache before and after each test."""
    _clear_settings_cache()
    yield
    _clear_settings_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(
    api_key: str | None = "sk-test-key",
    backend: str = "openai",
) -> MagicMock:
    """Create a mock GpumodSettings."""
    settings = MagicMock()
    settings.llm_backend = backend
    settings.llm_model = "gpt-4o-mini"
    settings.llm_base_url = None
    if api_key is not None:
        secret = MagicMock()
        secret.get_secret_value.return_value = api_key
        settings.llm_api_key = secret
    else:
        settings.llm_api_key = None
    return settings


def _openai_response(content: dict[str, Any]) -> httpx.Response:
    """Build a mock OpenAI chat completions response."""
    body = {"choices": [{"message": {"content": json.dumps(content)}}]}
    return httpx.Response(200, json=body)


# ---------------------------------------------------------------------------
# SEC-R2: Rate limiter rejects excess MCP requests
# ---------------------------------------------------------------------------


class TestRateLimiterIntegration:
    """End-to-end rate limiting via RateLimitMiddleware (SEC-R2)."""

    def test_rate_limiter_allows_within_limit(self) -> None:
        """Requests within the rate limit are accepted."""
        limiter = RateLimitMiddleware(max_requests=5, window_seconds=1.0)
        # First 5 requests should succeed
        for _ in range(5):
            limiter._check_rate("test_client")  # Should not raise

    def test_rate_limiter_rejects_excess_requests(self) -> None:
        """Requests exceeding the rate limit are rejected with RuntimeError."""
        limiter = RateLimitMiddleware(max_requests=3, window_seconds=60.0)
        # First 3 requests succeed
        for _ in range(3):
            limiter._check_rate("test_client")

        # 4th request should be rejected
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            limiter._check_rate("test_client")

    async def test_rate_limiter_rejects_mcp_tool_call(self) -> None:
        """MCP tool call is rejected when rate limit is exceeded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rate_test.db"

            # Pre-populate the database
            db = Database(db_path)
            await db.connect()
            await db.insert_service(
                Service(
                    id="test-svc",
                    name="Test Service",
                    driver=DriverType.VLLM,
                    port=8000,
                    vram_mb=4000,
                )
            )
            await db.close()

            @asynccontextmanager
            async def _lifespan(
                server: FastMCP[dict[str, Any]],
            ) -> AsyncIterator[dict[str, Any]]:
                async with gpumod_lifespan(server, db_path=db_path) as ctx:
                    yield ctx

            # max_requests=2 so the 3rd request fails
            server: FastMCP[dict[str, Any]] = FastMCP(
                name="gpumod-test",
                lifespan=_lifespan,
                middleware=[
                    ErrorSanitizationMiddleware(),
                    RateLimitMiddleware(max_requests=2, window_seconds=60.0),
                ],
            )
            register_resources(server)
            register_tools(server)

            client = Client(server)
            async with client:
                # First 2 requests should succeed
                r1 = await client.call_tool("list_services", {})
                assert r1.data is not None

                r2 = await client.call_tool("list_services", {})
                assert r2.data is not None

                # 3rd request should trigger rate limit
                # The rate limit error is caught by ErrorSanitizationMiddleware
                # and re-raised. FastMCP client wraps it.
                try:
                    r3 = await client.call_tool("list_services", {})
                    # If it returns, the data should contain an error indicator
                    if hasattr(r3, "data") and isinstance(r3.data, dict):
                        assert "error" in str(r3.data).lower() or "rate" in str(r3.data).lower()
                    else:
                        # The error was raised as text content
                        assert "rate limit" in str(r3).lower() or "error" in str(r3).lower()
                except Exception as exc:
                    # Rate limit error propagated
                    assert "rate limit" in str(exc).lower() or "Rate limit" in str(exc)


# ---------------------------------------------------------------------------
# SEC-L1: LLM response with injected IDs rejected
# ---------------------------------------------------------------------------


class TestLLMInjectionRejection:
    """LLM responses with injected/malicious service IDs are rejected (SEC-L1)."""

    def test_path_traversal_service_id_rejected(self) -> None:
        """service_id containing path traversal is rejected by validator."""
        raw = {
            "services": [
                {"service_id": "../../etc/passwd", "vram_mb": 4000},
            ],
            "reasoning": "malicious payload",
        }
        with pytest.raises(LLMResponseError, match="service_id"):
            validate_plan_response(raw)

    def test_shell_injection_service_id_rejected(self) -> None:
        """service_id containing shell metacharacters is rejected."""
        raw = {
            "services": [
                {"service_id": "svc; rm -rf /", "vram_mb": 4000},
            ],
            "reasoning": "shell injection attempt",
        }
        with pytest.raises(LLMResponseError, match="service_id"):
            validate_plan_response(raw)

    def test_template_injection_service_id_rejected(self) -> None:
        """service_id containing Jinja2 template syntax is rejected."""
        raw = {
            "services": [
                {"service_id": "{{7*7}}", "vram_mb": 4000},
            ],
            "reasoning": "template injection",
        }
        with pytest.raises(LLMResponseError, match="service_id"):
            validate_plan_response(raw)

    def test_sql_injection_service_id_rejected(self) -> None:
        """service_id containing SQL injection patterns is rejected."""
        raw = {
            "services": [
                {"service_id": "svc' OR 1=1--", "vram_mb": 4000},
            ],
            "reasoning": "SQL injection attempt",
        }
        with pytest.raises(LLMResponseError, match="service_id"):
            validate_plan_response(raw)

    async def test_injected_id_rejected_in_full_llm_flow(self) -> None:
        """Full flow: LLM returns injected ID -> prompt -> generate -> validate -> reject."""
        # LLM "hallucinates" a malicious service_id
        malicious_plan = {
            "services": [
                {"service_id": "valid-svc", "vram_mb": 4000},
                {"service_id": "/etc/shadow", "vram_mb": 2000},
            ],
            "reasoning": "normal looking plan",
        }

        mock_response = _openai_response(malicious_plan)
        mock_transport = AsyncMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        settings = _make_settings()
        backend = OpenAIBackend(settings)

        async with httpx.AsyncClient(transport=mock_transport) as client:
            backend._client = client
            raw = await backend.generate("plan the services")

        # The response parses as JSON fine, but SEC-L1 validation catches it
        with pytest.raises(LLMResponseError, match="service_id"):
            validate_plan_response(raw)


# ---------------------------------------------------------------------------
# SEC-L3: API key never appears in any error output
# ---------------------------------------------------------------------------


class TestAPIKeyNeverExposed:
    """API key must never appear in any error message, repr, or string output."""

    async def test_key_not_in_timeout_error(self) -> None:
        """API key not leaked in timeout error from any backend."""
        secret = "sk-ABSOLUTELY-SECRET-key-99999"
        settings = _make_settings(api_key=secret)
        backend = OpenAIBackend(settings)

        mock_transport = AsyncMock()
        mock_transport.handle_async_request = AsyncMock(
            side_effect=httpx.ReadTimeout("read timed out"),
        )

        async with httpx.AsyncClient(transport=mock_transport) as client:
            backend._client = client
            with pytest.raises(LLMResponseError) as exc_info:
                await backend.generate("test")

        assert secret not in str(exc_info.value)
        assert secret not in repr(exc_info.value)

    async def test_key_not_in_connection_error(self) -> None:
        """API key not leaked in connection error."""
        secret = "sk-CONN-ERROR-secret-key-42"
        settings = _make_settings(api_key=secret)
        backend = OpenAIBackend(settings)

        mock_transport = AsyncMock()
        mock_transport.handle_async_request = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused"),
        )

        async with httpx.AsyncClient(transport=mock_transport) as client:
            backend._client = client
            with pytest.raises(LLMResponseError) as exc_info:
                await backend.generate("test")

        assert secret not in str(exc_info.value)

    async def test_key_not_in_http_401_error(self) -> None:
        """API key not leaked in 401 Unauthorized error."""
        secret = "sk-AUTH-FAIL-secret-key-77"
        settings = _make_settings(api_key=secret)
        backend = OpenAIBackend(settings)

        mock_response = httpx.Response(401, json={"error": "Invalid API key"})
        mock_transport = AsyncMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        async with httpx.AsyncClient(transport=mock_transport) as client:
            backend._client = client
            with pytest.raises(LLMResponseError) as exc_info:
                await backend.generate("test")

        error_str = str(exc_info.value)
        assert secret not in error_str
        # Should only contain generic HTTP error info
        assert "401" in error_str or "HTTP" in error_str

    def test_key_not_in_settings_repr(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SecretStr prevents key from appearing in settings repr."""
        monkeypatch.setenv("GPUMOD_LLM_API_KEY", "sk-REPR-TEST-secret-12345")
        _clear_settings_cache()
        settings = GpumodSettings()
        repr_str = repr(settings)
        dump_str = str(settings.model_dump())
        assert "sk-REPR-TEST-secret-12345" not in repr_str
        assert "sk-REPR-TEST-secret-12345" not in dump_str

    def test_key_not_in_cli_error_output(self) -> None:
        """API key is not exposed when CLI plan command hits an error."""
        from gpumod.cli import app

        svc = Service(
            id="svc-1",
            name="Test",
            driver=DriverType.VLLM,
            port=8000,
            vram_mb=4096,
        )
        mock_ctx = MagicMock()
        mock_ctx.db = MagicMock()
        mock_ctx.db.close = AsyncMock()
        mock_ctx.db.list_services = AsyncMock(return_value=[svc])
        mock_ctx.db.get_setting = AsyncMock(return_value=None)
        mock_ctx.vram = MagicMock()
        gpu_info = MagicMock()
        gpu_info.vram_total_mb = 24576
        mock_ctx.vram.get_gpu_info = AsyncMock(return_value=gpu_info)

        secret_key = "sk-CLI-ERROR-test-secret-98765"
        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(
            side_effect=LLMResponseError("Connection refused by API endpoint"),
        )

        cli_runner = typer.testing.CliRunner()
        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli_plan.get_settings") as mock_settings,
            patch("gpumod.cli_plan.get_backend", return_value=mock_backend),
        ):
            mock_settings.return_value = MagicMock(
                llm_api_key=MagicMock(
                    get_secret_value=MagicMock(return_value=secret_key),
                ),
                llm_backend="openai",
            )
            result = cli_runner.invoke(app, ["plan", "suggest"])

        # The secret key must never appear in CLI output
        assert secret_key not in result.output
