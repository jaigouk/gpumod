"""Tests for gpumod.cli_plan -- Plan CLI commands (suggest)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import typer.testing

from gpumod.cli import app
from gpumod.llm.base import LLMResponseError
from gpumod.models import (
    DriverType,
    Service,
    SimulationResult,
    SleepMode,
)

runner = typer.testing.CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(
    *,
    id: str = "svc-1",
    name: str = "Test Service",
    driver: DriverType = DriverType.VLLM,
    port: int | None = 8000,
    vram_mb: int = 4096,
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
    reasoning: str = "Balanced allocation across services.",
) -> dict[str, Any]:
    """Create a raw LLM plan response dict."""
    if services is None:
        services = [
            {"service_id": "svc-1", "vram_mb": 4096},
            {"service_id": "svc-2", "vram_mb": 8192},
        ]
    return {
        "services": services,
        "reasoning": reasoning,
    }


def _make_simulation_result(
    *,
    fits: bool = True,
    gpu_total_mb: int = 24576,
    current_usage_mb: int = 0,
    proposed_usage_mb: int = 12288,
    headroom_mb: int = 12288,
    services: list[Service] | None = None,
) -> SimulationResult:
    return SimulationResult(
        fits=fits,
        gpu_total_mb=gpu_total_mb,
        current_usage_mb=current_usage_mb,
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
# TestPlanSuggest -- basic flow
# ---------------------------------------------------------------------------


class TestPlanSuggest:
    """Tests for `gpumod plan suggest` command."""

    def test_plan_suggest_basic(self) -> None:
        """Calls LLM and displays plan as Rich table."""
        svc1 = _make_service(id="svc-1", vram_mb=4096)
        svc2 = _make_service(id="svc-2", name="Service 2", vram_mb=8192)
        mock_ctx = _make_mock_context(services=[svc1, svc2])
        sim_result = _make_simulation_result(
            fits=True,
            proposed_usage_mb=12288,
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
        # Should display plan with service allocations
        assert "svc-1" in output
        assert "svc-2" in output
        assert "4096" in output or "4,096" in output
        assert "8192" in output or "8,192" in output
        # Should show reasoning
        assert "Balanced allocation" in output or "reasoning" in output.lower()
        # Advisory: should show suggested commands (SEC-L4)
        assert "gpumod" in output.lower()

    def test_plan_suggest_with_mode(self) -> None:
        """--mode flag filters services for the specified mode."""
        svc1 = _make_service(id="svc-1", vram_mb=4096)
        mock_ctx = _make_mock_context(services=[svc1])
        mock_ctx.db.get_mode = AsyncMock(return_value=MagicMock(id="dev-mode", name="Dev Mode"))
        mock_ctx.db.get_mode_services = AsyncMock(return_value=[svc1])
        sim_result = _make_simulation_result(
            fits=True,
            proposed_usage_mb=4096,
            services=[svc1],
        )
        mock_ctx.simulation.simulate_services = AsyncMock(return_value=sim_result)

        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(
            return_value=_make_plan_response(
                services=[{"service_id": "svc-1", "vram_mb": 4096}],
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
            result = runner.invoke(app, ["plan", "suggest", "--mode", "dev-mode"])

        assert result.exit_code == 0
        # Should have used mode services
        mock_ctx.db.get_mode_services.assert_awaited_once_with("dev-mode")

    def test_plan_suggest_with_budget(self) -> None:
        """--budget flag passes budget_mb to prompt builder."""
        svc1 = _make_service(id="svc-1", vram_mb=4096)
        mock_ctx = _make_mock_context(services=[svc1])
        sim_result = _make_simulation_result(
            fits=True,
            proposed_usage_mb=4096,
            services=[svc1],
        )
        mock_ctx.simulation.simulate_services = AsyncMock(return_value=sim_result)

        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(
            return_value=_make_plan_response(
                services=[{"service_id": "svc-1", "vram_mb": 4096}],
            )
        )

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli_plan.get_settings") as mock_settings,
            patch("gpumod.cli_plan.get_backend", return_value=mock_backend),
            patch("gpumod.cli_plan.build_planning_prompt", wraps=None) as mock_prompt,
        ):
            mock_settings.return_value = MagicMock(
                llm_api_key=MagicMock(get_secret_value=MagicMock(return_value="sk-test")),
                llm_backend="openai",
            )
            # Make mock_prompt return something the backend can use
            mock_prompt.return_value = "test prompt"
            result = runner.invoke(app, ["plan", "suggest", "--budget", "16000"])

        assert result.exit_code == 0
        # Verify budget was passed to prompt builder
        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args
        assert call_kwargs.kwargs.get("budget_mb") == 16000 or (
            len(call_kwargs.args) >= 4 and call_kwargs.args[3] == 16000
        )

    def test_plan_suggest_json(self) -> None:
        """--json flag outputs valid JSON."""
        svc1 = _make_service(id="svc-1", vram_mb=4096)
        svc2 = _make_service(id="svc-2", name="Service 2", vram_mb=8192)
        mock_ctx = _make_mock_context(services=[svc1, svc2])
        sim_result = _make_simulation_result(
            fits=True,
            proposed_usage_mb=12288,
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
            result = runner.invoke(app, ["plan", "suggest", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "plan" in parsed
        assert "simulation" in parsed
        assert "services" in parsed["plan"]
        assert "reasoning" in parsed["plan"]
        assert isinstance(parsed["plan"]["services"], list)

    def test_plan_suggest_dry_run(self) -> None:
        """--dry-run shows prompt without calling LLM API."""
        svc1 = _make_service(id="svc-1", vram_mb=4096)
        mock_ctx = _make_mock_context(services=[svc1])

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
        # Should show the prompt content
        assert "SERVICE DATA" in result.output or "service" in result.output.lower()
        assert "svc-1" in result.output
        # LLM should NOT have been called
        mock_backend.generate.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestPlanSuggestErrors -- error handling
# ---------------------------------------------------------------------------


class TestPlanSuggestErrors:
    """Tests for error handling in plan suggest."""

    def test_no_api_key_configured(self) -> None:
        """Shows clear error message when no API key is configured."""
        svc1 = _make_service(id="svc-1", vram_mb=4096)
        mock_ctx = _make_mock_context(services=[svc1])

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli_plan.get_settings") as mock_settings,
        ):
            mock_settings.return_value = MagicMock(
                llm_api_key=None,
                llm_backend="openai",
            )
            result = runner.invoke(app, ["plan", "suggest"])

        assert result.exit_code == 0  # error_handler catches it
        output = result.output.lower()
        assert "api key" in output or "api_key" in output or "not configured" in output

    def test_no_api_key_ollama_ok(self) -> None:
        """Ollama backend does not require an API key."""
        svc1 = _make_service(id="svc-1", vram_mb=4096)
        mock_ctx = _make_mock_context(services=[svc1])
        sim_result = _make_simulation_result(
            fits=True,
            proposed_usage_mb=4096,
            services=[svc1],
        )
        mock_ctx.simulation.simulate_services = AsyncMock(return_value=sim_result)

        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(
            return_value=_make_plan_response(
                services=[{"service_id": "svc-1", "vram_mb": 4096}],
            )
        )

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli_plan.get_settings") as mock_settings,
            patch("gpumod.cli_plan.get_backend", return_value=mock_backend),
        ):
            mock_settings.return_value = MagicMock(
                llm_api_key=None,
                llm_backend="ollama",
            )
            result = runner.invoke(app, ["plan", "suggest"])

        assert result.exit_code == 0
        # Should succeed -- not an API key error
        output = result.output.lower()
        assert "api key" not in output

    def test_llm_unavailable(self) -> None:
        """LLMResponseError handled with clear error message."""
        svc1 = _make_service(id="svc-1", vram_mb=4096)
        mock_ctx = _make_mock_context(services=[svc1])

        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(side_effect=LLMResponseError("Connection refused"))

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

        assert result.exit_code == 0  # error_handler catches it
        output = result.output.lower()
        assert "error" in output

    def test_invalid_llm_response_rejected(self) -> None:
        """Invalid LLM response (SEC-L1) shows clear error."""
        svc1 = _make_service(id="svc-1", vram_mb=4096)
        mock_ctx = _make_mock_context(services=[svc1])

        # Return invalid response (bad service_id format)
        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(
            return_value={
                "services": [{"service_id": "../etc/passwd", "vram_mb": 4096}],
                "reasoning": "malicious",
            }
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

        assert result.exit_code == 0  # error_handler catches it
        output = result.output.lower()
        assert "error" in output or "invalid" in output

    def test_llm_response_unknown_service_id(self) -> None:
        """LLM returns service_id not in DB -- rejected with clear error."""
        svc1 = _make_service(id="svc-1", vram_mb=4096)
        mock_ctx = _make_mock_context(services=[svc1])
        # Simulate services will raise because unknown-svc not in DB
        mock_ctx.simulation.simulate_services = AsyncMock(
            side_effect=ValueError("Service not found: 'unknown-svc'")
        )

        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(
            return_value=_make_plan_response(
                services=[
                    {"service_id": "svc-1", "vram_mb": 4096},
                    {"service_id": "unknown-svc", "vram_mb": 8192},
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

        assert result.exit_code == 0  # error_handler catches it
        output = result.output.lower()
        assert "error" in output or "not found" in output


# ---------------------------------------------------------------------------
# TestPlanAdvisoryOnly -- SEC-L4 advisory mode
# ---------------------------------------------------------------------------


class TestPlanAdvisoryOnly:
    """Plan is advisory only -- shows suggested commands, never auto-executes (SEC-L4)."""

    def test_advisory_shows_commands(self) -> None:
        """Plan output includes suggested CLI commands for manual execution."""
        svc1 = _make_service(id="svc-1", vram_mb=4096)
        svc2 = _make_service(id="svc-2", name="Service 2", vram_mb=8192)
        mock_ctx = _make_mock_context(services=[svc1, svc2])
        sim_result = _make_simulation_result(
            fits=True,
            proposed_usage_mb=12288,
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
        # Should contain advisory language
        assert "advisory" in output.lower() or "suggested" in output.lower()
        # Should contain gpumod commands the user can run manually
        assert "gpumod" in output.lower()

    def test_plan_never_auto_executes(self) -> None:
        """Plan never calls lifecycle start/stop -- purely advisory."""
        svc1 = _make_service(id="svc-1", vram_mb=4096)
        mock_ctx = _make_mock_context(services=[svc1])
        sim_result = _make_simulation_result(
            fits=True,
            proposed_usage_mb=4096,
            services=[svc1],
        )
        mock_ctx.simulation.simulate_services = AsyncMock(return_value=sim_result)

        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(
            return_value=_make_plan_response(
                services=[{"service_id": "svc-1", "vram_mb": 4096}],
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
        # Lifecycle methods must NOT have been called
        mock_ctx.lifecycle.start.assert_not_called()
        mock_ctx.lifecycle.stop.assert_not_called()


# ---------------------------------------------------------------------------
# TestPlanSuggestNoServices
# ---------------------------------------------------------------------------


class TestPlanSuggestNoServices:
    """Tests for plan suggest with no services in DB."""

    def test_no_services_in_db(self) -> None:
        """Shows clear message when no services are registered."""
        mock_ctx = _make_mock_context(services=[])

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli_plan.get_settings") as mock_settings,
        ):
            mock_settings.return_value = MagicMock(
                llm_api_key=MagicMock(get_secret_value=MagicMock(return_value="sk-test")),
                llm_backend="openai",
            )
            result = runner.invoke(app, ["plan", "suggest"])

        assert result.exit_code == 0
        output = result.output.lower()
        assert "no services" in output

    def test_mode_not_found(self) -> None:
        """Shows error when specified mode doesn't exist."""
        mock_ctx = _make_mock_context(services=[])
        mock_ctx.db.get_mode = AsyncMock(return_value=None)

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli_plan.get_settings") as mock_settings,
        ):
            mock_settings.return_value = MagicMock(
                llm_api_key=MagicMock(get_secret_value=MagicMock(return_value="sk-test")),
                llm_backend="openai",
            )
            result = runner.invoke(app, ["plan", "suggest", "--mode", "nonexistent"])

        assert result.exit_code == 0  # error_handler catches it
        output = result.output.lower()
        assert "not found" in output or "error" in output


# ---------------------------------------------------------------------------
# TestPlanSuggestSimulation
# ---------------------------------------------------------------------------


class TestPlanSuggestSimulation:
    """Tests for simulation verification of plan."""

    def test_plan_fits_gpu(self) -> None:
        """Plan shows 'Fits' when simulation says plan fits GPU."""
        svc1 = _make_service(id="svc-1", vram_mb=4096)
        mock_ctx = _make_mock_context(services=[svc1])
        sim_result = _make_simulation_result(
            fits=True,
            proposed_usage_mb=4096,
            headroom_mb=20480,
            services=[svc1],
        )
        mock_ctx.simulation.simulate_services = AsyncMock(return_value=sim_result)

        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(
            return_value=_make_plan_response(
                services=[{"service_id": "svc-1", "vram_mb": 4096}],
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
        assert "fits" in output

    def test_plan_exceeds_gpu(self) -> None:
        """Plan shows warning when simulation says plan exceeds GPU."""
        svc1 = _make_service(id="svc-1", vram_mb=20000)
        svc2 = _make_service(id="svc-2", name="Service 2", vram_mb=10000)
        mock_ctx = _make_mock_context(services=[svc1, svc2])
        sim_result = _make_simulation_result(
            fits=False,
            proposed_usage_mb=30000,
            gpu_total_mb=24576,
            headroom_mb=-5424,
            services=[svc1, svc2],
        )
        mock_ctx.simulation.simulate_services = AsyncMock(return_value=sim_result)

        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(
            return_value=_make_plan_response(
                services=[
                    {"service_id": "svc-1", "vram_mb": 20000},
                    {"service_id": "svc-2", "vram_mb": 10000},
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
        assert "does not fit" in output or "not fit" in output or "warning" in output
