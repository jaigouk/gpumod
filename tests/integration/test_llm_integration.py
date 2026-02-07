"""Integration tests for LLM backend end-to-end flow.

Verifies that:
- Full flow: generate prompt -> mock HTTP response -> validate output -> return plan
- Invalid LLM response -> LLMResponseError
- API key missing -> clear error (without exposing key in message)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from gpumod.llm import LLMResponseError
from gpumod.llm.openai_backend import OpenAIBackend
from gpumod.llm.prompts import PLANNING_SYSTEM_PROMPT, build_planning_prompt
from gpumod.llm.response_validator import validate_plan_response

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(
    backend: str = "openai",
    api_key: str | None = "sk-integration-test-key",
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
) -> MagicMock:
    """Create a mock GpumodSettings for LLM backend construction."""
    settings = MagicMock()
    settings.llm_backend = backend
    settings.llm_model = model
    settings.llm_base_url = base_url
    if api_key is not None:
        secret = MagicMock()
        secret.get_secret_value.return_value = api_key
        settings.llm_api_key = secret
    else:
        settings.llm_api_key = None
    return settings


def _openai_response(content: dict[str, Any]) -> httpx.Response:
    """Build a mock OpenAI chat completions response."""
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(content),
                }
            }
        ]
    }
    return httpx.Response(200, json=body)


def _valid_plan_dict(
    service_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Return a valid plan response dict with given service IDs."""
    if service_ids is None:
        service_ids = ["vllm-chat", "fastapi-app"]
    return {
        "services": [
            {"service_id": sid, "vram_mb": 4000 + i * 2000} for i, sid in enumerate(service_ids)
        ],
        "reasoning": "Optimal allocation based on model sizes and GPU capacity.",
    }


# ---------------------------------------------------------------------------
# Full flow integration tests
# ---------------------------------------------------------------------------


class TestLLMFullFlowIntegration:
    """End-to-end: build prompt -> call LLM -> validate response -> return plan."""

    async def test_full_plan_flow_openai(self) -> None:
        """Complete flow: prompt -> OpenAI mock -> validate -> PlanSuggestion."""
        # 1. Build the planning prompt from service data
        services = [
            {"id": "vllm-chat", "name": "vLLM Chat", "vram_mb": 8000},
            {"id": "fastapi-app", "name": "FastAPI App", "vram_mb": 1000},
        ]
        prompt = build_planning_prompt(
            services=services,
            gpu_total_mb=24576,
            current_mode="chat",
        )
        assert "vllm-chat" in prompt
        assert "24576" in prompt

        # 2. Mock the LLM HTTP call
        plan_data = _valid_plan_dict(["vllm-chat", "fastapi-app"])
        mock_response = _openai_response(plan_data)
        mock_transport = AsyncMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        # 3. Call the backend
        settings = _make_settings()
        backend = OpenAIBackend(settings)

        async with httpx.AsyncClient(transport=mock_transport) as client:
            backend._client = client
            raw = await backend.generate(
                prompt=prompt,
                system=PLANNING_SYSTEM_PROMPT,
            )

        # 4. Validate the response (SEC-L1)
        plan = validate_plan_response(raw)
        assert len(plan.services) == 2
        assert plan.services[0].service_id == "vllm-chat"
        assert plan.services[1].service_id == "fastapi-app"
        assert plan.services[0].vram_mb > 0
        assert plan.services[1].vram_mb > 0
        assert "allocation" in plan.reasoning.lower() or "optimal" in plan.reasoning.lower()

    async def test_full_plan_flow_anthropic(self) -> None:
        """Complete flow with Anthropic backend."""
        from gpumod.llm.anthropic_backend import AnthropicBackend

        services = [
            {"id": "llama-code", "name": "Llama Code", "vram_mb": 12000},
        ]
        prompt = build_planning_prompt(
            services=services,
            gpu_total_mb=24576,
            budget_mb=16000,
        )
        assert "llama-code" in prompt
        assert "16000" in prompt

        plan_data = _valid_plan_dict(["llama-code"])
        body = {"content": [{"type": "text", "text": json.dumps(plan_data)}]}
        mock_response = httpx.Response(200, json=body)
        mock_transport = AsyncMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        settings = _make_settings(backend="anthropic", api_key="sk-ant-test")
        backend = AnthropicBackend(settings)

        async with httpx.AsyncClient(transport=mock_transport) as client:
            backend._client = client
            raw = await backend.generate(
                prompt=prompt,
                system=PLANNING_SYSTEM_PROMPT,
            )

        plan = validate_plan_response(raw)
        assert len(plan.services) == 1
        assert plan.services[0].service_id == "llama-code"

    async def test_full_plan_flow_ollama(self) -> None:
        """Complete flow with Ollama backend (no API key needed)."""
        from gpumod.llm.ollama_backend import OllamaBackend

        services = [
            {"id": "vllm-embed", "name": "vLLM Embed", "vram_mb": 4000},
        ]
        prompt = build_planning_prompt(
            services=services,
            gpu_total_mb=24576,
        )

        plan_data = _valid_plan_dict(["vllm-embed"])
        body = {"response": json.dumps(plan_data)}
        mock_response = httpx.Response(200, json=body)
        mock_transport = AsyncMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        settings = _make_settings(backend="ollama", api_key=None, model="llama3.1")
        backend = OllamaBackend(settings)

        async with httpx.AsyncClient(transport=mock_transport) as client:
            backend._client = client
            raw = await backend.generate(
                prompt=prompt,
                system=PLANNING_SYSTEM_PROMPT,
            )

        plan = validate_plan_response(raw)
        assert len(plan.services) == 1
        assert plan.services[0].service_id == "vllm-embed"


# ---------------------------------------------------------------------------
# Invalid LLM response tests
# ---------------------------------------------------------------------------


class TestLLMInvalidResponseIntegration:
    """Invalid LLM responses are caught end-to-end as LLMResponseError."""

    async def test_invalid_json_from_llm_raises_response_error(self) -> None:
        """LLM returns non-JSON content -> LLMResponseError."""
        body = {"choices": [{"message": {"content": "This is not JSON at all."}}]}
        mock_response = httpx.Response(200, json=body)
        mock_transport = AsyncMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        settings = _make_settings()
        backend = OpenAIBackend(settings)

        async with httpx.AsyncClient(transport=mock_transport) as client:
            backend._client = client
            with pytest.raises(LLMResponseError, match="parse"):
                await backend.generate("test prompt")

    async def test_valid_json_but_bad_schema_raises_response_error(self) -> None:
        """LLM returns valid JSON but wrong schema -> LLMResponseError at validation."""
        # Return JSON that doesn't match PlanSuggestion schema
        bad_response = {"answer": "I think you should use more VRAM."}
        mock_response = _openai_response(bad_response)
        mock_transport = AsyncMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        settings = _make_settings()
        backend = OpenAIBackend(settings)

        async with httpx.AsyncClient(transport=mock_transport) as client:
            backend._client = client
            raw = await backend.generate("test prompt")

        # The backend parses JSON fine, but validation should fail
        with pytest.raises(LLMResponseError, match="Invalid"):
            validate_plan_response(raw)

    async def test_http_500_error_raises_response_error(self) -> None:
        """HTTP 500 from LLM API -> LLMResponseError end-to-end."""
        mock_response = httpx.Response(500, json={"error": "Internal server error"})
        mock_transport = AsyncMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        settings = _make_settings()
        backend = OpenAIBackend(settings)

        async with httpx.AsyncClient(transport=mock_transport) as client:
            backend._client = client
            with pytest.raises(LLMResponseError, match="HTTP.*500"):
                await backend.generate("test prompt")


# ---------------------------------------------------------------------------
# API key missing -> clear error
# ---------------------------------------------------------------------------


class TestLLMAPIKeyMissing:
    """Missing API key produces clear error without key exposure (SEC-L3)."""

    def test_check_api_key_raises_for_openai_without_key(self) -> None:
        """OpenAI backend with no API key: cli_plan._check_api_key raises ValueError."""
        from gpumod.cli_plan import _check_api_key

        settings = MagicMock()
        settings.llm_backend = "openai"
        settings.llm_api_key = None

        with pytest.raises(ValueError, match="API key"):
            _check_api_key(settings)

    def test_check_api_key_ok_for_ollama_without_key(self) -> None:
        """Ollama backend with no API key: no error raised."""
        from gpumod.cli_plan import _check_api_key

        settings = MagicMock()
        settings.llm_backend = "ollama"
        settings.llm_api_key = None

        # Should not raise
        _check_api_key(settings)

    async def test_api_key_never_in_error_output_end_to_end(self) -> None:
        """Across all error types, the secret API key never appears in messages."""
        secret_key = "sk-SUPER-SECRET-key-12345-abcdef"
        settings = _make_settings(api_key=secret_key)
        backend = OpenAIBackend(settings)

        error_scenarios: list[Exception] = [
            httpx.ReadTimeout("timed out"),
            httpx.ConnectError("refused"),
        ]

        for exc in error_scenarios:
            mock_transport = AsyncMock()
            mock_transport.handle_async_request = AsyncMock(side_effect=exc)

            async with httpx.AsyncClient(transport=mock_transport) as client:
                backend._client = client
                with pytest.raises(LLMResponseError) as exc_info:
                    await backend.generate("test prompt")

            error_msg = str(exc_info.value)
            assert secret_key not in error_msg, f"Secret key leaked in error: {error_msg}"

        # Also check HTTP error
        mock_response = httpx.Response(401, json={"error": "Unauthorized"})
        mock_transport = AsyncMock()
        mock_transport.handle_async_request = AsyncMock(return_value=mock_response)

        async with httpx.AsyncClient(transport=mock_transport) as client:
            backend._client = client
            with pytest.raises(LLMResponseError) as exc_info:
                await backend.generate("test prompt")

        assert secret_key not in str(exc_info.value)
