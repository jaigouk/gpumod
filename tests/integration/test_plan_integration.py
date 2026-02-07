"""Integration tests for the plan CLI command.

Verifies that:
- `gpumod plan suggest` with mocked LLM displays a plan table
- Plan simulation verification: suggested services checked against GPU capacity
- `--dry-run` shows prompt without API call
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest
import typer.testing

from gpumod.cli import app
from gpumod.config import _clear_settings_cache
from gpumod.models import (
    DriverType,
    Service,
    SimulationResult,
    SleepMode,
)

pytestmark = pytest.mark.integration

runner = typer.testing.CliRunner()


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


def _make_service(
    *,
    id: str = "vllm-chat",
    name: str = "vLLM Chat",
    driver: DriverType = DriverType.VLLM,
    port: int | None = 8000,
    vram_mb: int = 8000,
    sleep_mode: SleepMode = SleepMode.NONE,
    model_id: str | None = None,
) -> Service:
    return Service(
        id=id,
        name=name,
        driver=driver,
        port=port,
        vram_mb=vram_mb,
        sleep_mode=sleep_mode,
        model_id=model_id,
    )


def _make_plan_response(
    services: list[dict[str, Any]] | None = None,
    reasoning: str = "Balanced allocation for optimal GPU utilization.",
) -> dict[str, Any]:
    """Create a raw LLM plan response dict."""
    if services is None:
        services = [
            {"service_id": "vllm-chat", "vram_mb": 8000},
            {"service_id": "fastapi-app", "vram_mb": 1000},
        ]
    return {"services": services, "reasoning": reasoning}


def _make_sim_result(
    *,
    fits: bool = True,
    gpu_total_mb: int = 24576,
    proposed_usage_mb: int = 9000,
    headroom_mb: int = 15576,
    services: list[Service] | None = None,
) -> SimulationResult:
    return SimulationResult(
        fits=fits,
        gpu_total_mb=gpu_total_mb,
        current_usage_mb=0,
        proposed_usage_mb=proposed_usage_mb,
        headroom_mb=headroom_mb,
        services=services or [],
        alternatives=[],
    )


def _make_mock_context(
    services: list[Service] | None = None,
    current_mode: str | None = None,
    gpu_total_mb: int = 24576,
) -> MagicMock:
    """Build a mock AppContext with db, simulation, and vram."""
    ctx = MagicMock()
    ctx.db = MagicMock()
    ctx.db.close = AsyncMock()
    ctx.db.list_services = AsyncMock(return_value=services or [])
    ctx.db.get_setting = AsyncMock(return_value=current_mode)
    ctx.vram = MagicMock()
    gpu_info = MagicMock()
    gpu_info.vram_total_mb = gpu_total_mb
    ctx.vram.get_gpu_info = AsyncMock(return_value=gpu_info)
    ctx.simulation = MagicMock()
    ctx.simulation.simulate_services = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# Plan suggest with mocked LLM -> displays plan table
# ---------------------------------------------------------------------------


class TestPlanSuggestIntegration:
    """Full integration: config -> CLI -> LLM backend -> plan display."""

    def test_plan_suggest_displays_plan_table(self) -> None:
        """gpumod plan suggest with mocked LLM displays plan as Rich table."""
        svc1 = _make_service(id="vllm-chat", vram_mb=8000)
        svc2 = _make_service(
            id="fastapi-app",
            name="FastAPI App",
            driver=DriverType.FASTAPI,
            port=9000,
            vram_mb=1000,
        )
        mock_ctx = _make_mock_context(services=[svc1, svc2])
        sim_result = _make_sim_result(
            fits=True,
            proposed_usage_mb=9000,
            headroom_mb=15576,
            services=[svc1, svc2],
        )
        mock_ctx.simulation.simulate_services = AsyncMock(return_value=sim_result)

        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(return_value=_make_plan_response())

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli_plan.get_settings") as mock_settings,
            patch("gpumod.cli_plan.get_backend", return_value=mock_backend),
        ):
            mock_settings.return_value = MagicMock(
                llm_api_key=MagicMock(get_secret_value=MagicMock(return_value="sk-test")),
                llm_backend="openai",
            )
            result = runner.invoke(app, ["plan", "suggest"])

        assert result.exit_code == 0
        output = result.output
        # Should display the plan table with service IDs and VRAM
        assert "vllm-chat" in output
        assert "fastapi-app" in output
        assert "8000" in output or "8,000" in output
        assert "1000" in output or "1,000" in output
        # Should show the reasoning from the LLM
        assert "Balanced allocation" in output or "allocation" in output.lower()
        # Should show Fits status
        assert "Fits" in output or "fits" in output.lower()
        # Should show advisory commands (SEC-L4)
        assert "gpumod" in output.lower()

    def test_plan_suggest_json_output(self) -> None:
        """--json outputs valid JSON with plan, simulation, and advisory commands."""
        svc1 = _make_service(id="vllm-chat", vram_mb=8000)
        mock_ctx = _make_mock_context(services=[svc1])
        sim_result = _make_sim_result(
            fits=True,
            proposed_usage_mb=8000,
            services=[svc1],
        )
        mock_ctx.simulation.simulate_services = AsyncMock(return_value=sim_result)

        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(
            return_value=_make_plan_response(
                services=[{"service_id": "vllm-chat", "vram_mb": 8000}],
            )
        )

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli_plan.get_settings") as mock_settings,
            patch("gpumod.cli_plan.get_backend", return_value=mock_backend),
        ):
            mock_settings.return_value = MagicMock(
                llm_api_key=MagicMock(get_secret_value=MagicMock(return_value="sk-test")),
                llm_backend="openai",
            )
            result = runner.invoke(app, ["plan", "suggest", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "plan" in parsed
        assert "simulation" in parsed
        assert "advisory_commands" in parsed
        assert parsed["plan"]["services"][0]["service_id"] == "vllm-chat"
        assert parsed["simulation"]["fits"] is True


# ---------------------------------------------------------------------------
# Plan simulation verification
# ---------------------------------------------------------------------------


class TestPlanSimulationVerification:
    """Plan suggested services are verified against GPU capacity via simulation."""

    def test_plan_that_exceeds_gpu_shows_does_not_fit(self) -> None:
        """If LLM suggests too much VRAM, simulation shows 'does not fit'."""
        svc1 = _make_service(id="vllm-chat", vram_mb=20000)
        svc2 = _make_service(
            id="llama-code",
            name="Llama Code",
            driver=DriverType.LLAMACPP,
            port=8002,
            vram_mb=12000,
        )
        mock_ctx = _make_mock_context(services=[svc1, svc2])
        sim_result = _make_sim_result(
            fits=False,
            proposed_usage_mb=32000,
            headroom_mb=-7424,
            services=[svc1, svc2],
        )
        mock_ctx.simulation.simulate_services = AsyncMock(return_value=sim_result)

        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(
            return_value=_make_plan_response(
                services=[
                    {"service_id": "vllm-chat", "vram_mb": 20000},
                    {"service_id": "llama-code", "vram_mb": 12000},
                ],
            )
        )

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli_plan.get_settings") as mock_settings,
            patch("gpumod.cli_plan.get_backend", return_value=mock_backend),
        ):
            mock_settings.return_value = MagicMock(
                llm_api_key=MagicMock(get_secret_value=MagicMock(return_value="sk-test")),
                llm_backend="openai",
            )
            result = runner.invoke(app, ["plan", "suggest"])

        assert result.exit_code == 0
        output = result.output.lower()
        assert "does not fit" in output or "not fit" in output

    def test_simulation_called_with_llm_service_ids(self) -> None:
        """Simulation is called with the service IDs from the LLM response."""
        svc1 = _make_service(id="vllm-chat", vram_mb=8000)
        svc2 = _make_service(
            id="fastapi-app",
            name="FastAPI App",
            driver=DriverType.FASTAPI,
            port=9000,
            vram_mb=1000,
        )
        mock_ctx = _make_mock_context(services=[svc1, svc2])
        sim_result = _make_sim_result(
            fits=True,
            proposed_usage_mb=9000,
            services=[svc1, svc2],
        )
        mock_ctx.simulation.simulate_services = AsyncMock(return_value=sim_result)

        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(
            return_value=_make_plan_response(
                services=[
                    {"service_id": "vllm-chat", "vram_mb": 8000},
                    {"service_id": "fastapi-app", "vram_mb": 1000},
                ],
            )
        )

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli_plan.get_settings") as mock_settings,
            patch("gpumod.cli_plan.get_backend", return_value=mock_backend),
        ):
            mock_settings.return_value = MagicMock(
                llm_api_key=MagicMock(get_secret_value=MagicMock(return_value="sk-test")),
                llm_backend="openai",
            )
            result = runner.invoke(app, ["plan", "suggest"])

        assert result.exit_code == 0
        # Verify simulate_services was called with the LLM's service IDs
        mock_ctx.simulation.simulate_services.assert_awaited_once_with(
            ["vllm-chat", "fastapi-app"],
        )


# ---------------------------------------------------------------------------
# Dry run tests
# ---------------------------------------------------------------------------


class TestPlanDryRun:
    """--dry-run shows prompt without calling the LLM API."""

    def test_dry_run_shows_prompts_no_api_call(self) -> None:
        """--dry-run displays system and user prompts without calling LLM."""
        svc1 = _make_service(id="vllm-chat", vram_mb=8000)
        svc2 = _make_service(
            id="fastapi-app",
            name="FastAPI App",
            driver=DriverType.FASTAPI,
            port=9000,
            vram_mb=1000,
        )
        mock_ctx = _make_mock_context(services=[svc1, svc2])

        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock()

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli_plan.get_settings") as mock_settings,
            patch("gpumod.cli_plan.get_backend", return_value=mock_backend),
        ):
            mock_settings.return_value = MagicMock(
                llm_api_key=None,
                llm_backend="openai",
            )
            result = runner.invoke(app, ["plan", "suggest", "--dry-run"])

        assert result.exit_code == 0
        output = result.output
        # Should show system prompt
        assert "System prompt" in output or "system" in output.lower()
        # Should show user prompt with service data
        assert "SERVICE DATA" in output or "vllm-chat" in output
        # Should NOT have called the LLM
        mock_backend.generate.assert_not_awaited()

    def test_dry_run_does_not_require_api_key(self) -> None:
        """--dry-run works even without an API key configured."""
        svc1 = _make_service(id="vllm-chat", vram_mb=8000)
        mock_ctx = _make_mock_context(services=[svc1])

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli_plan.get_settings") as mock_settings,
        ):
            mock_settings.return_value = MagicMock(
                llm_api_key=None,
                llm_backend="openai",
            )
            result = runner.invoke(app, ["plan", "suggest", "--dry-run"])

        assert result.exit_code == 0
        output = result.output.lower()
        # Should NOT show an API key error
        assert "api key" not in output
        # Should show prompt content
        assert "vllm-chat" in output
