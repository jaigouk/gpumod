"""Tests for gpumod.cli_service — Service CLI commands (list, status, start, stop)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import typer.testing

from gpumod.cli import app
from gpumod.models import DriverType, Service, ServiceState, ServiceStatus

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
) -> Service:
    return Service(id=id, name=name, driver=driver, port=port, vram_mb=vram_mb)


def _make_status(
    *,
    state: ServiceState = ServiceState.RUNNING,
    vram_mb: int | None = 4096,
    uptime_seconds: int | None = 120,
    health_ok: bool | None = True,
) -> ServiceStatus:
    return ServiceStatus(
        state=state,
        vram_mb=vram_mb,
        uptime_seconds=uptime_seconds,
        health_ok=health_ok,
    )


def _make_mock_context(
    **overrides: object,
) -> MagicMock:
    ctx = MagicMock()
    ctx.registry = MagicMock()
    ctx.lifecycle = MagicMock()
    ctx.manager = MagicMock()
    ctx.db = MagicMock()

    # Set up async mocks with defaults
    ctx.registry.list_all = AsyncMock(return_value=[])
    ctx.registry.get = AsyncMock()
    ctx.lifecycle.start = AsyncMock()
    ctx.lifecycle.stop = AsyncMock()
    ctx.db.close = AsyncMock()

    # Mock get_driver to return a driver mock with an async status method
    mock_driver = MagicMock()
    mock_driver.status = AsyncMock(return_value=_make_status())
    ctx.registry.get_driver = MagicMock(return_value=mock_driver)

    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


# ---------------------------------------------------------------------------
# service list tests
# ---------------------------------------------------------------------------


class TestServiceList:
    """Tests for `gpumod service list` command."""

    def test_service_list_shows_all_services(self) -> None:
        svc1 = _make_service(id="svc-1", name="vLLM Service", port=8000, vram_mb=4096)
        svc2 = _make_service(
            id="svc-2",
            name="LlamaCpp Service",
            driver=DriverType.LLAMACPP,
            port=8001,
            vram_mb=2048,
        )
        mock_ctx = _make_mock_context()
        mock_ctx.registry.list_all = AsyncMock(return_value=[svc1, svc2])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["service", "list"])

        assert result.exit_code == 0
        assert "svc-1" in result.output
        assert "svc-2" in result.output
        assert "vLLM Service" in result.output
        assert "LlamaCpp Service" in result.output

    def test_service_list_json_flag_outputs_json(self) -> None:
        svc1 = _make_service(id="svc-1", name="vLLM Service")
        mock_ctx = _make_mock_context()
        mock_ctx.registry.list_all = AsyncMock(return_value=[svc1])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["service", "list", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "svc-1"

    def test_service_list_empty_shows_message(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.registry.list_all = AsyncMock(return_value=[])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["service", "list"])

        assert result.exit_code == 0
        assert "no services" in result.output.lower()

    def test_service_list_rich_table_has_columns(self) -> None:
        svc1 = _make_service(id="svc-1", name="vLLM Service")
        mock_ctx = _make_mock_context()
        mock_ctx.registry.list_all = AsyncMock(return_value=[svc1])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["service", "list"])

        assert result.exit_code == 0
        output = result.output
        assert "ID" in output
        assert "Name" in output
        assert "Driver" in output
        assert "Port" in output
        assert "VRAM" in output
        assert "State" in output


# ---------------------------------------------------------------------------
# service status tests
# ---------------------------------------------------------------------------


class TestServiceStatus:
    """Tests for `gpumod service status <service_id>` command."""

    def test_service_status_shows_service_info(self) -> None:
        svc = _make_service(id="svc-1", name="vLLM Service", port=8000, vram_mb=4096)
        status = _make_status(state=ServiceState.RUNNING, vram_mb=4096, uptime_seconds=300)
        mock_ctx = _make_mock_context()
        mock_ctx.registry.get = AsyncMock(return_value=svc)
        mock_driver = MagicMock()
        mock_driver.status = AsyncMock(return_value=status)
        mock_ctx.registry.get_driver = MagicMock(return_value=mock_driver)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["service", "status", "svc-1"])

        assert result.exit_code == 0
        assert "svc-1" in result.output
        assert "running" in result.output.lower()

    def test_service_status_not_found_shows_error(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.registry.get = AsyncMock(side_effect=KeyError("Service not found: 'missing'"))

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["service", "status", "missing"])

        assert result.exit_code == 0  # error_handler catches it
        assert "error" in result.output.lower() or "not found" in result.output.lower()

    def test_service_status_json_flag(self) -> None:
        svc = _make_service(id="svc-1", name="vLLM Service", port=8000, vram_mb=4096)
        status = _make_status(state=ServiceState.RUNNING, vram_mb=4096, uptime_seconds=300)
        mock_ctx = _make_mock_context()
        mock_ctx.registry.get = AsyncMock(return_value=svc)
        mock_driver = MagicMock()
        mock_driver.status = AsyncMock(return_value=status)
        mock_ctx.registry.get_driver = MagicMock(return_value=mock_driver)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["service", "status", "svc-1", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["id"] == "svc-1"
        assert parsed["status"]["state"] == "running"


# ---------------------------------------------------------------------------
# service start tests
# ---------------------------------------------------------------------------


class TestServiceStart:
    """Tests for `gpumod service start <service_id>` command."""

    def test_service_start_calls_lifecycle_start(self) -> None:
        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["service", "start", "svc-1"])

        assert result.exit_code == 0
        mock_ctx.lifecycle.start.assert_awaited_once_with("svc-1")
        assert "svc-1" in result.output
        # Should show success-like message
        assert "start" in result.output.lower()

    def test_service_start_not_found_shows_error(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.lifecycle.start = AsyncMock(side_effect=KeyError("Service not found: 'missing'"))

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["service", "start", "missing"])

        assert result.exit_code == 0  # error_handler catches it
        assert "error" in result.output.lower() or "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# service stop tests
# ---------------------------------------------------------------------------


class TestServiceStop:
    """Tests for `gpumod service stop <service_id>` command."""

    def test_service_stop_calls_lifecycle_stop(self) -> None:
        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["service", "stop", "svc-1"])

        assert result.exit_code == 0
        mock_ctx.lifecycle.stop.assert_awaited_once_with("svc-1")
        assert "svc-1" in result.output
        assert "stop" in result.output.lower()

    def test_service_stop_not_found_shows_error(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.lifecycle.stop = AsyncMock(side_effect=KeyError("Service not found: 'missing'"))

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["service", "stop", "missing"])

        assert result.exit_code == 0  # error_handler catches it
        assert "error" in result.output.lower() or "not found" in result.output.lower()
