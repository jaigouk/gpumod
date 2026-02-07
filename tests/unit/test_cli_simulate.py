"""Tests for gpumod.cli_simulate -- Simulate CLI commands (mode, services)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import typer.testing

from gpumod.cli import app
from gpumod.models import (
    DriverType,
    Service,
    SimulationAlternative,
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


def _make_simulation_result(
    *,
    fits: bool = True,
    gpu_total_mb: int = 24576,
    current_usage_mb: int = 0,
    proposed_usage_mb: int = 12000,
    headroom_mb: int = 12576,
    services: list[Service] | None = None,
    alternatives: list[SimulationAlternative] | None = None,
) -> SimulationResult:
    return SimulationResult(
        fits=fits,
        gpu_total_mb=gpu_total_mb,
        current_usage_mb=current_usage_mb,
        proposed_usage_mb=proposed_usage_mb,
        headroom_mb=headroom_mb,
        services=services or [],
        alternatives=alternatives or [],
    )


def _make_alternative(
    *,
    id: str = "alt-1",
    strategy: str = "service_removal",
    description: str = "Remove svc-big (16000 MB)",
    affected_services: list[str] | None = None,
    vram_saved_mb: int = 16000,
    projected_total_mb: int = 12000,
    trade_offs: list[str] | None = None,
) -> SimulationAlternative:
    return SimulationAlternative(
        id=id,
        strategy=strategy,
        description=description,
        affected_services=affected_services or ["svc-big"],
        vram_saved_mb=vram_saved_mb,
        projected_total_mb=projected_total_mb,
        trade_offs=trade_offs or ["Service svc-big will be unavailable"],
    )


def _make_mock_context() -> MagicMock:
    ctx = MagicMock()
    ctx.db = MagicMock()
    ctx.db.close = AsyncMock()
    ctx.registry = MagicMock()
    ctx.vram = MagicMock()
    ctx.model_registry = MagicMock()
    ctx.simulation = MagicMock()
    ctx.simulation.simulate_mode = AsyncMock()
    ctx.simulation.simulate_services = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# TestSimulateMode
# ---------------------------------------------------------------------------


class TestSimulateMode:
    """Tests for `gpumod simulate mode <mode_id>` command."""

    def test_simulate_mode_fits(self) -> None:
        """Shows 'Fits' message with VRAM summary when simulation fits."""
        svc1 = _make_service(id="svc-a", vram_mb=8000)
        svc2 = _make_service(id="svc-b", vram_mb=4000)
        result = _make_simulation_result(
            fits=True,
            proposed_usage_mb=12000,
            gpu_total_mb=24576,
            headroom_mb=12576,
            services=[svc1, svc2],
        )
        mock_ctx = _make_mock_context()
        mock_ctx.simulation.simulate_mode = AsyncMock(return_value=result)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            cli_result = runner.invoke(app, ["simulate", "mode", "dev-mode"])

        assert cli_result.exit_code == 0
        output = cli_result.output.lower()
        assert "fits" in output
        assert "12000" in cli_result.output or "12,000" in cli_result.output
        assert "24576" in cli_result.output or "24,576" in cli_result.output

    def test_simulate_mode_exceeds(self) -> None:
        """Shows 'Does not fit' + alternatives table when VRAM exceeds."""
        svc1 = _make_service(id="svc-a", vram_mb=16000)
        svc2 = _make_service(id="svc-b", vram_mb=12000)
        alt = _make_alternative(
            id="remove-svc-a",
            strategy="service_removal",
            description="Remove svc-a (16000 MB)",
            affected_services=["svc-a"],
            vram_saved_mb=16000,
            projected_total_mb=12000,
        )
        result = _make_simulation_result(
            fits=False,
            proposed_usage_mb=28000,
            gpu_total_mb=24576,
            headroom_mb=-3424,
            services=[svc1, svc2],
            alternatives=[alt],
        )
        mock_ctx = _make_mock_context()
        mock_ctx.simulation.simulate_mode = AsyncMock(return_value=result)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            cli_result = runner.invoke(app, ["simulate", "mode", "heavy-mode"])

        assert cli_result.exit_code == 0
        output = cli_result.output.lower()
        assert "does not fit" in output or "not fit" in output
        # Should show alternatives
        assert "remove" in output or "alternative" in output

    def test_simulate_mode_with_add(self) -> None:
        """--add flag passes add_service_ids to engine."""
        result = _make_simulation_result(fits=True)
        mock_ctx = _make_mock_context()
        mock_ctx.simulation.simulate_mode = AsyncMock(return_value=result)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            cli_result = runner.invoke(app, ["simulate", "mode", "dev-mode", "--add", "svc-extra"])

        assert cli_result.exit_code == 0
        mock_ctx.simulation.simulate_mode.assert_awaited_once()
        call_kwargs = mock_ctx.simulation.simulate_mode.call_args
        # Check that add was passed correctly
        assert call_kwargs.kwargs.get("add") == ["svc-extra"] or (
            call_kwargs.args and "svc-extra" in str(call_kwargs)
        )

    def test_simulate_mode_with_remove(self) -> None:
        """--remove flag passes remove_service_ids to engine."""
        result = _make_simulation_result(fits=True)
        mock_ctx = _make_mock_context()
        mock_ctx.simulation.simulate_mode = AsyncMock(return_value=result)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            cli_result = runner.invoke(
                app, ["simulate", "mode", "dev-mode", "--remove", "svc-old"]
            )

        assert cli_result.exit_code == 0
        mock_ctx.simulation.simulate_mode.assert_awaited_once()
        call_kwargs = mock_ctx.simulation.simulate_mode.call_args
        assert call_kwargs.kwargs.get("remove") == ["svc-old"] or (
            call_kwargs.args and "svc-old" in str(call_kwargs)
        )

    def test_simulate_mode_json(self) -> None:
        """--json outputs valid JSON SimulationResult."""
        svc1 = _make_service(id="svc-a", vram_mb=8000)
        result = _make_simulation_result(
            fits=True,
            proposed_usage_mb=8000,
            gpu_total_mb=24576,
            headroom_mb=16576,
            services=[svc1],
        )
        mock_ctx = _make_mock_context()
        mock_ctx.simulation.simulate_mode = AsyncMock(return_value=result)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            cli_result = runner.invoke(app, ["simulate", "mode", "dev-mode", "--json"])

        assert cli_result.exit_code == 0
        parsed = json.loads(cli_result.output)
        assert parsed["fits"] is True
        assert parsed["proposed_usage_mb"] == 8000
        assert parsed["gpu_total_mb"] == 24576

    def test_simulate_mode_visual(self) -> None:
        """--visual calls ComparisonView rendering."""
        svc1 = _make_service(id="svc-a", vram_mb=8000)
        result = _make_simulation_result(
            fits=True,
            proposed_usage_mb=8000,
            gpu_total_mb=24576,
            headroom_mb=16576,
            services=[svc1],
        )
        mock_ctx = _make_mock_context()
        mock_ctx.simulation.simulate_mode = AsyncMock(return_value=result)

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli_simulate.ComparisonView") as mock_cv,
        ):
            mock_cv_instance = MagicMock()
            mock_cv_instance.render.return_value = "Current: (0 MB)\nProposed: (8000 MB)"
            mock_cv.return_value = mock_cv_instance

            cli_result = runner.invoke(app, ["simulate", "mode", "dev-mode", "--visual"])

        assert cli_result.exit_code == 0
        mock_cv_instance.render.assert_called_once()


# ---------------------------------------------------------------------------
# TestSimulateServices
# ---------------------------------------------------------------------------


class TestSimulateServices:
    """Tests for `gpumod simulate services <service_ids>` command."""

    def test_simulate_services_fits(self) -> None:
        """Shows fit result for ad-hoc service list."""
        svc1 = _make_service(id="svc-a", vram_mb=8000)
        svc2 = _make_service(id="svc-b", vram_mb=4000)
        result = _make_simulation_result(
            fits=True,
            proposed_usage_mb=12000,
            gpu_total_mb=24576,
            headroom_mb=12576,
            services=[svc1, svc2],
        )
        mock_ctx = _make_mock_context()
        mock_ctx.simulation.simulate_services = AsyncMock(return_value=result)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            cli_result = runner.invoke(app, ["simulate", "services", "svc-a,svc-b"])

        assert cli_result.exit_code == 0
        output = cli_result.output.lower()
        assert "fits" in output

    def test_simulate_services_exceeds(self) -> None:
        """Shows overflow + alternatives."""
        svc1 = _make_service(id="svc-a", vram_mb=16000)
        svc2 = _make_service(id="svc-b", vram_mb=12000)
        alt = _make_alternative()
        result = _make_simulation_result(
            fits=False,
            proposed_usage_mb=28000,
            gpu_total_mb=24576,
            headroom_mb=-3424,
            services=[svc1, svc2],
            alternatives=[alt],
        )
        mock_ctx = _make_mock_context()
        mock_ctx.simulation.simulate_services = AsyncMock(return_value=result)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            cli_result = runner.invoke(app, ["simulate", "services", "svc-a,svc-b"])

        assert cli_result.exit_code == 0
        output = cli_result.output.lower()
        assert "does not fit" in output or "not fit" in output

    def test_simulate_services_json(self) -> None:
        """--json output for services subcommand."""
        svc1 = _make_service(id="svc-a", vram_mb=8000)
        result = _make_simulation_result(
            fits=True,
            proposed_usage_mb=8000,
            services=[svc1],
        )
        mock_ctx = _make_mock_context()
        mock_ctx.simulation.simulate_services = AsyncMock(return_value=result)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            cli_result = runner.invoke(app, ["simulate", "services", "svc-a", "--json"])

        assert cli_result.exit_code == 0
        parsed = json.loads(cli_result.output)
        assert parsed["fits"] is True
        assert parsed["proposed_usage_mb"] == 8000

    def test_simulate_services_unknown(self) -> None:
        """Shows error for unknown service ID."""
        mock_ctx = _make_mock_context()
        mock_ctx.simulation.simulate_services = AsyncMock(
            side_effect=ValueError("Service not found: 'unknown-svc'")
        )

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            cli_result = runner.invoke(app, ["simulate", "services", "unknown-svc"])

        assert cli_result.exit_code == 0  # error_handler catches it
        assert "error" in cli_result.output.lower() or "not found" in cli_result.output.lower()


# ---------------------------------------------------------------------------
# TestSimulateContext
# ---------------------------------------------------------------------------


class TestSimulateContext:
    """Tests for --context override parsing."""

    def test_context_override_parsed(self) -> None:
        """--context svc=8192 parsed correctly and passed to engine."""
        result = _make_simulation_result(fits=True)
        mock_ctx = _make_mock_context()
        mock_ctx.simulation.simulate_mode = AsyncMock(return_value=result)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            cli_result = runner.invoke(
                app,
                ["simulate", "mode", "dev-mode", "--context", "svc-a=8192"],
            )

        assert cli_result.exit_code == 0
        mock_ctx.simulation.simulate_mode.assert_awaited_once()
        call_kwargs = mock_ctx.simulation.simulate_mode.call_args
        context_overrides = call_kwargs.kwargs.get("context_overrides")
        assert context_overrides == {"svc-a": 8192}

    def test_context_override_invalid_format(self) -> None:
        """Bad format shows error."""
        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            cli_result = runner.invoke(
                app,
                ["simulate", "mode", "dev-mode", "--context", "badformat"],
            )

        assert cli_result.exit_code == 0  # error_handler catches it
        assert "error" in cli_result.output.lower() or "invalid" in cli_result.output.lower()
