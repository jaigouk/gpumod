"""Tests for gpumod.services.unit_installer — auto-install systemd unit files."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpumod.models import Service
from gpumod.services.unit_installer import UnitFileInstaller


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_service(
    service_id: str = "vllm-chat",
    unit_name: str = "vllm-chat.service",
    driver: str = "vllm",
    port: int = 7071,
    vram_mb: int = 7000,
) -> Service:
    return Service(
        id=service_id,
        name=f"Test {service_id}",
        driver=driver,
        port=port,
        vram_mb=vram_mb,
        unit_name=unit_name,
        extra_config={"unit_vars": {"gpu_mem_util": 0.35}},
    )


def _make_installer(tmp_path: Path) -> tuple[UnitFileInstaller, MagicMock, AsyncMock]:
    """Create a UnitFileInstaller with mocked DB and template engine."""
    mock_db = AsyncMock()
    mock_db.get_setting = AsyncMock(return_value=None)

    mock_engine = MagicMock()
    mock_engine.render_service_unit = MagicMock(
        return_value="[Unit]\nDescription=Test\n[Service]\nExecStart=/usr/bin/test\n"
    )

    installer = UnitFileInstaller(
        db=mock_db,
        template_engine=mock_engine,
        unit_dir=tmp_path,
    )
    return installer, mock_engine, mock_db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUnitFileInstaller:
    """Tests for UnitFileInstaller.ensure_unit_file."""

    @pytest.mark.asyncio
    async def test_installs_missing_unit_file(self, tmp_path: Path) -> None:
        """When unit file does not exist, it should be created."""
        installer, mock_engine, _ = _make_installer(tmp_path)
        service = _make_service()

        await installer.ensure_unit_file(service)

        unit_path = tmp_path / "vllm-chat.service"
        assert unit_path.exists()
        content = unit_path.read_text()
        assert "[Unit]" in content
        mock_engine.render_service_unit.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_existing_unit_file(self, tmp_path: Path) -> None:
        """When unit file already exists, it should not be overwritten."""
        installer, mock_engine, _ = _make_installer(tmp_path)
        service = _make_service()

        # Pre-create the file
        unit_path = tmp_path / "vllm-chat.service"
        unit_path.write_text("existing content")

        await installer.ensure_unit_file(service)

        assert unit_path.read_text() == "existing content"
        mock_engine.render_service_unit.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_service_without_unit_name(self, tmp_path: Path) -> None:
        """Services without a unit_name should be skipped."""
        installer, mock_engine, _ = _make_installer(tmp_path)
        service = _make_service(unit_name="")
        service.unit_name = None

        await installer.ensure_unit_file(service)

        mock_engine.render_service_unit.assert_not_called()

    @pytest.mark.asyncio
    async def test_marks_daemon_reload_needed(self, tmp_path: Path) -> None:
        """After installing a unit file, daemon_reload_needed should be True."""
        installer, _, _ = _make_installer(tmp_path)
        service = _make_service()

        assert not installer._daemon_reload_needed
        await installer.ensure_unit_file(service)
        assert installer._daemon_reload_needed

    @pytest.mark.asyncio
    async def test_no_reload_when_file_exists(self, tmp_path: Path) -> None:
        """No reload flag when unit file already exists."""
        installer, _, _ = _make_installer(tmp_path)
        service = _make_service()

        (tmp_path / "vllm-chat.service").write_text("existing")

        await installer.ensure_unit_file(service)
        assert not installer._daemon_reload_needed

    @pytest.mark.asyncio
    async def test_passes_unit_vars_to_template(self, tmp_path: Path) -> None:
        """unit_vars from service.extra_config should be passed to the template engine."""
        installer, mock_engine, _ = _make_installer(tmp_path)
        service = _make_service()

        await installer.ensure_unit_file(service)

        call_kwargs = mock_engine.render_service_unit.call_args
        assert call_kwargs[1].get("unit_vars") == {"gpu_mem_util": 0.35} or \
            call_kwargs[0][2] == {"gpu_mem_util": 0.35}

    @pytest.mark.asyncio
    async def test_handles_template_render_error(self, tmp_path: Path) -> None:
        """Template render errors should be logged, not raised."""
        installer, mock_engine, _ = _make_installer(tmp_path)
        mock_engine.render_service_unit.side_effect = ValueError("bad template")
        service = _make_service()

        # Should not raise
        await installer.ensure_unit_file(service)

        # File should not be created
        assert not (tmp_path / "vllm-chat.service").exists()

    @pytest.mark.asyncio
    async def test_installs_multiple_services(self, tmp_path: Path) -> None:
        """Multiple services should each get their own unit file."""
        installer, _, _ = _make_installer(tmp_path)

        svc1 = _make_service("svc-a", "svc-a.service")
        svc2 = _make_service("svc-b", "svc-b.service")

        await installer.ensure_unit_file(svc1)
        await installer.ensure_unit_file(svc2)

        assert (tmp_path / "svc-a.service").exists()
        assert (tmp_path / "svc-b.service").exists()


class TestDaemonReload:
    """Tests for daemon_reload_if_needed."""

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_reloads_when_needed(self, mock_exec: MagicMock, tmp_path: Path) -> None:
        """daemon_reload_if_needed should call systemctl when units were installed."""
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_exec.return_value = proc

        installer, _, _ = _make_installer(tmp_path)
        installer._daemon_reload_needed = True

        await installer.daemon_reload_if_needed()

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert args == ("systemctl", "--user", "daemon-reload")
        assert not installer._daemon_reload_needed

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_skips_reload_when_not_needed(self, mock_exec: MagicMock, tmp_path: Path) -> None:
        """daemon_reload_if_needed should not call systemctl when no units were installed."""
        installer, _, _ = _make_installer(tmp_path)
        installer._daemon_reload_needed = False

        await installer.daemon_reload_if_needed()

        mock_exec.assert_not_called()


class TestLifecycleIntegration:
    """Tests that LifecycleManager calls UnitFileInstaller before starting services."""

    @pytest.mark.asyncio
    async def test_lifecycle_calls_ensure_unit_file(self) -> None:
        """LifecycleManager.start should call ensure_unit_file for each service."""
        from gpumod.models import ServiceState, ServiceStatus
        from gpumod.services.lifecycle import LifecycleManager

        # Mock registry
        service = _make_service()
        mock_registry = AsyncMock()
        mock_registry.get = AsyncMock(return_value=service)

        mock_driver = AsyncMock()
        mock_driver.supports_sleep = False
        mock_driver.status = AsyncMock(
            return_value=ServiceStatus(state=ServiceState.STOPPED)
        )
        mock_driver.start = AsyncMock()
        mock_driver.health_check = AsyncMock(return_value=True)
        mock_registry.get_driver = MagicMock(return_value=mock_driver)

        # Mock unit installer
        mock_installer = AsyncMock()
        mock_installer.ensure_unit_file = AsyncMock()
        mock_installer.daemon_reload_if_needed = AsyncMock()

        lifecycle = LifecycleManager(mock_registry, unit_installer=mock_installer)
        await lifecycle.start("vllm-chat")

        mock_installer.ensure_unit_file.assert_called_once_with(service)
        mock_installer.daemon_reload_if_needed.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifecycle_works_without_installer(self) -> None:
        """LifecycleManager.start should work when no installer is provided."""
        from gpumod.models import ServiceState, ServiceStatus
        from gpumod.services.lifecycle import LifecycleManager

        service = _make_service()
        mock_registry = AsyncMock()
        mock_registry.get = AsyncMock(return_value=service)

        mock_driver = AsyncMock()
        mock_driver.supports_sleep = False
        mock_driver.status = AsyncMock(
            return_value=ServiceStatus(state=ServiceState.STOPPED)
        )
        mock_driver.start = AsyncMock()
        mock_driver.health_check = AsyncMock(return_value=True)
        mock_registry.get_driver = MagicMock(return_value=mock_driver)

        lifecycle = LifecycleManager(mock_registry)
        await lifecycle.start("vllm-chat")

        # Should not raise — just skips the installer step
        mock_driver.start.assert_called_once()
