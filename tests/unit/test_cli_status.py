"""Tests for gpumod.cli — status and init top-level commands."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import typer.testing

from gpumod.cli import app
from gpumod.models import (
    DriverType,
    GPUInfo,
    Service,
    ServiceInfo,
    ServiceState,
    ServiceStatus,
    SystemStatus,
    VRAMUsage,
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
) -> Service:
    return Service(id=id, name=name, driver=driver, port=port, vram_mb=vram_mb)


def _make_service_info(
    *,
    id: str = "svc-1",
    name: str = "Test Service",
    driver: DriverType = DriverType.VLLM,
    port: int | None = 8000,
    vram_mb: int = 4096,
    state: ServiceState = ServiceState.RUNNING,
    status_vram_mb: int | None = None,
) -> ServiceInfo:
    svc = _make_service(id=id, name=name, driver=driver, port=port, vram_mb=vram_mb)
    resolved_vram = status_vram_mb if status_vram_mb is not None else vram_mb
    status = ServiceStatus(state=state, vram_mb=resolved_vram)
    return ServiceInfo(service=svc, status=status)


def _make_gpu_info(
    *,
    name: str = "NVIDIA RTX 4090",
    vram_total_mb: int = 24576,
) -> GPUInfo:
    return GPUInfo(name=name, vram_total_mb=vram_total_mb)


def _make_vram_usage(
    *,
    total_mb: int = 24576,
    used_mb: int = 8192,
    free_mb: int = 16384,
) -> VRAMUsage:
    return VRAMUsage(total_mb=total_mb, used_mb=used_mb, free_mb=free_mb)


def _make_full_system_status(
    *,
    gpu: GPUInfo | None = None,
    vram: VRAMUsage | None = None,
    current_mode: str | None = "inference",
    services: list[ServiceInfo] | None = None,
) -> SystemStatus:
    if gpu is None:
        gpu = _make_gpu_info()
    if vram is None:
        vram = _make_vram_usage()
    if services is None:
        services = [
            _make_service_info(
                id="vllm-7b",
                name="vLLM 7B",
                vram_mb=4096,
                state=ServiceState.RUNNING,
            ),
            _make_service_info(
                id="llama-13b",
                name="LlamaCpp 13B",
                driver=DriverType.LLAMACPP,
                vram_mb=8192,
                state=ServiceState.SLEEPING,
            ),
        ]
    return SystemStatus(gpu=gpu, vram=vram, current_mode=current_mode, services=services)


def _make_mock_context() -> MagicMock:
    """Create a mock AppContext with all backend services mocked."""
    ctx = MagicMock()
    ctx.manager = MagicMock()
    ctx.manager.get_status = AsyncMock(return_value=_make_full_system_status())
    ctx.db = MagicMock()
    ctx.db.connect = AsyncMock()
    ctx.db.close = AsyncMock()
    ctx.db.insert_service = AsyncMock()
    ctx.preset_loader = MagicMock()
    ctx.preset_loader.discover_presets = MagicMock(return_value=[])
    ctx.preset_loader.to_service = MagicMock()
    return ctx


# ---------------------------------------------------------------------------
# status command tests
# ---------------------------------------------------------------------------


class TestStatusCommand:
    """Tests for `gpumod status` top-level command."""

    def test_status_shows_system_overview(self) -> None:
        """Status command with full SystemStatus shows GPU and service overview."""
        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        # Should show GPU name
        assert "RTX 4090" in result.output
        # Should show service names
        assert "vLLM 7B" in result.output
        assert "LlamaCpp 13B" in result.output

    def test_status_shows_gpu_info(self) -> None:
        """Status command shows GPU name and VRAM information."""
        gpu = _make_gpu_info(name="NVIDIA A100", vram_total_mb=81920)
        system_status = _make_full_system_status(gpu=gpu)
        mock_ctx = _make_mock_context()
        mock_ctx.manager.get_status = AsyncMock(return_value=system_status)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "A100" in result.output
        assert "81920" in result.output

    def test_status_shows_current_mode(self) -> None:
        """Status command shows the current mode name."""
        system_status = _make_full_system_status(current_mode="inference")
        mock_ctx = _make_mock_context()
        mock_ctx.manager.get_status = AsyncMock(return_value=system_status)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "inference" in result.output.lower()

    def test_status_shows_service_list(self) -> None:
        """Status command shows service names and their states."""
        services = [
            _make_service_info(id="svc-a", name="Alpha", state=ServiceState.RUNNING),
            _make_service_info(id="svc-b", name="Beta", state=ServiceState.STOPPED),
        ]
        system_status = _make_full_system_status(services=services)
        mock_ctx = _make_mock_context()
        mock_ctx.manager.get_status = AsyncMock(return_value=system_status)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Alpha" in result.output
        assert "Beta" in result.output
        assert "running" in result.output.lower()
        assert "stopped" in result.output.lower()

    def test_status_visual_flag_renders_vram_bar(self) -> None:
        """Status command with --visual flag uses the visualization module."""
        mock_ctx = _make_mock_context()

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli.StatusPanel") as mock_panel_cls,
        ):
            mock_panel_instance = MagicMock()
            mock_panel_instance.render = MagicMock(return_value=MagicMock())
            mock_panel_cls.return_value = mock_panel_instance

            result = runner.invoke(app, ["status", "--visual"])

        assert result.exit_code == 0
        mock_panel_instance.render.assert_called_once()

    def test_status_json_flag_outputs_json(self) -> None:
        """Status command with --json outputs valid JSON."""
        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["status", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)
        assert "gpu" in parsed
        assert "services" in parsed
        assert parsed["gpu"]["name"] == "NVIDIA RTX 4090"

    def test_status_no_gpu_shows_warning(self) -> None:
        """Status command with gpu=None shows a warning message."""
        system_status = SystemStatus(gpu=None, vram=None, current_mode=None, services=[])
        mock_ctx = _make_mock_context()
        mock_ctx.manager.get_status = AsyncMock(return_value=system_status)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "no gpu" in result.output.lower()


# ---------------------------------------------------------------------------
# init command tests
# ---------------------------------------------------------------------------


class TestInitCommand:
    """Tests for `gpumod init` top-level command."""

    def test_init_creates_database(self) -> None:
        """Init command connects to the database."""
        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        # create_context was called (which connects the db)

    def test_init_discovers_presets(self) -> None:
        """Init command calls discover_presets on the preset loader."""
        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        mock_ctx.preset_loader.discover_presets.assert_called_once()

    def test_init_loads_presets_into_db(self) -> None:
        """Init command inserts each preset as a service into the database."""
        from gpumod.models import PresetConfig

        preset1 = PresetConfig(
            id="preset-1", name="Preset One", driver=DriverType.VLLM, vram_mb=4096
        )
        preset2 = PresetConfig(
            id="preset-2", name="Preset Two", driver=DriverType.LLAMACPP, vram_mb=2048
        )
        svc1 = _make_service(id="preset-1", name="Preset One")
        svc2 = _make_service(id="preset-2", name="Preset Two")

        mock_ctx = _make_mock_context()
        mock_ctx.preset_loader.discover_presets = MagicMock(return_value=[preset1, preset2])
        mock_ctx.preset_loader.to_service = MagicMock(side_effect=[svc1, svc2])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert mock_ctx.db.insert_service.await_count == 2

    def test_init_shows_summary(self) -> None:
        """Init command shows the number of presets found."""
        from gpumod.models import PresetConfig

        preset1 = PresetConfig(
            id="preset-1", name="Preset One", driver=DriverType.VLLM, vram_mb=4096
        )
        preset2 = PresetConfig(
            id="preset-2", name="Preset Two", driver=DriverType.LLAMACPP, vram_mb=2048
        )
        svc1 = _make_service(id="preset-1", name="Preset One")
        svc2 = _make_service(id="preset-2", name="Preset Two")

        mock_ctx = _make_mock_context()
        mock_ctx.preset_loader.discover_presets = MagicMock(return_value=[preset1, preset2])
        mock_ctx.preset_loader.to_service = MagicMock(side_effect=[svc1, svc2])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "2" in result.output

    def test_init_custom_db_path(self) -> None:
        """Init command with --db-path option passes custom path to create_context."""
        mock_ctx = _make_mock_context()
        mock_create = AsyncMock(return_value=mock_ctx)

        with patch("gpumod.cli.create_context", new=mock_create):
            result = runner.invoke(app, ["init", "--db-path", "/tmp/custom.db"])

        assert result.exit_code == 0
        # Verify create_context was called with the custom db_path
        call_kwargs = mock_create.call_args
        assert call_kwargs is not None
        # The db_path should be a Path object matching our input
        from pathlib import Path

        called_path = call_kwargs[1].get("db_path") or call_kwargs[0][0]
        assert str(called_path) == "/tmp/custom.db" or called_path == Path("/tmp/custom.db")

    def test_init_idempotent(self) -> None:
        """Init command handles IntegrityError on duplicate insert gracefully."""
        import sqlite3

        from gpumod.models import PresetConfig

        preset1 = PresetConfig(
            id="preset-1", name="Preset One", driver=DriverType.VLLM, vram_mb=4096
        )
        svc1 = _make_service(id="preset-1", name="Preset One")

        mock_ctx = _make_mock_context()
        mock_ctx.preset_loader.discover_presets = MagicMock(return_value=[preset1])
        mock_ctx.preset_loader.to_service = MagicMock(return_value=svc1)
        # Simulate IntegrityError on insert (duplicate key)
        mock_ctx.db.insert_service = AsyncMock(
            side_effect=sqlite3.IntegrityError("UNIQUE constraint failed"),
        )

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["init"])

        # Should not crash — handles duplicate gracefully
        assert result.exit_code == 0
        # Should mention skipping or already existing
        output_lower = result.output.lower()
        assert "skip" in output_lower or "already" in output_lower or "1" in result.output

    def test_init_closes_db(self) -> None:
        """Init command closes the database connection on completion."""
        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        mock_ctx.db.close.assert_awaited_once()
