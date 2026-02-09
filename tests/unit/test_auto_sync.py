"""Tests for auto-sync on CLI/MCP startup — AC1-AC5 of gpumod-9h8."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import yaml
from typer.testing import CliRunner

from gpumod.cli import app

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_preset(directory: Path, filename: str, data: dict) -> Path:
    """Write a YAML preset file into a directory, return file path."""
    fp = directory / filename
    fp.write_text(yaml.safe_dump(data), encoding="utf-8")
    return fp


def _minimal_preset(
    id: str = "svc-test",
    name: str = "Test",
    driver: str = "vllm",
    port: int = 8000,
    vram_mb: int = 2000,
    **overrides: object,
) -> dict:
    """Return a minimal valid preset dict."""
    base = {
        "id": id,
        "name": name,
        "driver": driver,
        "port": port,
        "vram_mb": vram_mb,
    }
    base.update(overrides)
    return base


def _make_mock_context() -> MagicMock:
    """Create a mock AppContext with all backend services mocked."""
    ctx = MagicMock()
    ctx.manager = MagicMock()
    ctx.manager.get_status = AsyncMock()
    ctx.db = MagicMock()
    ctx.db.connect = AsyncMock()
    ctx.db.close = AsyncMock()
    ctx.preset_loader = MagicMock()
    ctx.preset_loader.discover_presets = MagicMock(return_value=[])
    ctx.mode_loader = MagicMock()
    ctx.mode_loader.discover_modes = MagicMock(return_value=[])
    return ctx


# ---------------------------------------------------------------------------
# AC1: cli_context() calls sync_presets() before yielding
# ---------------------------------------------------------------------------


class TestCliContextAutoSync:
    """Tests for auto-sync in cli_context() — AC1."""

    async def test_cli_context_calls_sync_presets(self) -> None:
        """cli_context() should call sync_presets() before yielding."""
        from gpumod.cli import cli_context

        mock_ctx = _make_mock_context()

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli.sync_presets", new=AsyncMock()) as mock_sync_presets,
            patch("gpumod.cli.sync_modes", new=AsyncMock()) as mock_sync_modes,
        ):
            async with cli_context():
                pass

            mock_sync_presets.assert_awaited_once()
            mock_sync_modes.assert_awaited_once()

    async def test_cli_context_reflects_yaml_changes(self, tmp_path: Path) -> None:
        """Editing a preset YAML should be reflected when cli_context() yields."""
        from gpumod.cli import cli_context

        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        preset = _minimal_preset(id="svc", name="Original", vram_mb=1000)
        _write_preset(preset_dir, "svc.yaml", preset)

        # First: create context and sync
        with (
            patch("gpumod.cli._builtin_presets_dir", return_value=preset_dir),
            patch("gpumod.cli._builtin_modes_dir", return_value=tmp_path / "modes"),
            patch("gpumod.cli._default_db_path", return_value=tmp_path / "test.db"),
        ):
            async with cli_context() as ctx:
                svc = await ctx.db.get_service("svc")
                assert svc is not None
                assert svc.vram_mb == 1000

            # Modify the preset YAML
            updated = _minimal_preset(id="svc", name="Updated", vram_mb=5000)
            _write_preset(preset_dir, "svc.yaml", updated)

            # Second context should auto-sync and reflect changes
            async with cli_context() as ctx:
                svc = await ctx.db.get_service("svc")
                assert svc is not None
                assert svc.vram_mb == 5000  # Updated value

    async def test_cli_context_syncs_modes(self, tmp_path: Path) -> None:
        """cli_context() should also sync modes after presets."""
        from gpumod.cli import cli_context

        mock_ctx = _make_mock_context()

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli.sync_presets", new=AsyncMock()),
            patch("gpumod.cli.sync_modes", new=AsyncMock()) as mock_sync_modes,
        ):
            async with cli_context():
                pass

            mock_sync_modes.assert_awaited_once()


# ---------------------------------------------------------------------------
# AC3: Sync errors do not crash CLI commands
# ---------------------------------------------------------------------------


class TestAutoSyncErrorHandling:
    """Tests for graceful error handling in auto-sync — AC3."""

    async def test_malformed_yaml_does_not_crash(self, tmp_path: Path) -> None:
        """Malformed YAML should log a warning but not crash the command."""
        from gpumod.cli import cli_context

        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        # Write valid YAML
        _write_preset(preset_dir, "valid.yaml", _minimal_preset(id="valid"))
        # Write malformed YAML
        (preset_dir / "invalid.yaml").write_text("id: test\n  bad indent: here", encoding="utf-8")

        with (
            patch("gpumod.cli._builtin_presets_dir", return_value=preset_dir),
            patch("gpumod.cli._builtin_modes_dir", return_value=tmp_path / "modes"),
            patch("gpumod.cli._default_db_path", return_value=tmp_path / "test.db"),
        ):
            # Should not raise — degraded gracefully
            async with cli_context() as ctx:
                # Valid preset should still be synced
                svc = await ctx.db.get_service("valid")
                assert svc is not None

    async def test_sync_error_logged_as_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Sync errors should be logged at WARNING level."""
        from gpumod.cli import cli_context

        mock_ctx = _make_mock_context()
        sync_error = AsyncMock(side_effect=ValueError("YAML parse error"))

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli.sync_presets", new=sync_error),
            patch("gpumod.cli.sync_modes", new=AsyncMock()),
            caplog.at_level(logging.WARNING),
        ):
            # Should not raise
            async with cli_context():
                pass

            # Should log warning
            assert any("sync" in record.message.lower() or "error" in record.message.lower()
                      for record in caplog.records)


# ---------------------------------------------------------------------------
# AC4: --no-sync flag skips auto-sync
# ---------------------------------------------------------------------------


class TestNoSyncFlag:
    """Tests for --no-sync flag — AC4."""

    async def test_no_sync_flag_skips_sync(self) -> None:
        """cli_context(no_sync=True) should skip sync_presets() and sync_modes()."""
        from gpumod.cli import cli_context

        mock_ctx = _make_mock_context()

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli.sync_presets", new=AsyncMock()) as mock_sync_presets,
            patch("gpumod.cli.sync_modes", new=AsyncMock()) as mock_sync_modes,
        ):
            async with cli_context(no_sync=True):
                pass

            mock_sync_presets.assert_not_awaited()
            mock_sync_modes.assert_not_awaited()

    def test_status_command_has_no_sync_flag(self) -> None:
        """gpumod status should have a --no-sync flag."""
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "--no-sync" in result.output

    def test_status_with_no_sync_skips_sync(self) -> None:
        """gpumod status --no-sync should skip auto-sync."""
        from gpumod.models import SystemStatus

        mock_ctx = _make_mock_context()
        mock_ctx.manager.get_status = AsyncMock(
            return_value=SystemStatus(gpu=None, vram=None, current_mode=None, services=[])
        )

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli.sync_presets", new=AsyncMock()) as mock_sync_presets,
            patch("gpumod.cli.sync_modes", new=AsyncMock()) as mock_sync_modes,
        ):
            result = runner.invoke(app, ["status", "--no-sync"])

            assert result.exit_code == 0
            mock_sync_presets.assert_not_awaited()
            mock_sync_modes.assert_not_awaited()


# ---------------------------------------------------------------------------
# AC5: Idempotent — running sync 10 times produces the same DB state
# ---------------------------------------------------------------------------


class TestAutoSyncIdempotent:
    """Tests for idempotent auto-sync — AC5."""

    async def test_multiple_syncs_produce_same_state(self, tmp_path: Path) -> None:
        """Running sync 10 times should produce the same DB state as once."""
        from gpumod.cli import cli_context

        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        _write_preset(preset_dir, "a.yaml", _minimal_preset(id="a", vram_mb=1000))
        _write_preset(preset_dir, "b.yaml", _minimal_preset(id="b", vram_mb=2000))

        modes_dir = tmp_path / "modes"
        modes_dir.mkdir()

        with (
            patch("gpumod.cli._builtin_presets_dir", return_value=preset_dir),
            patch("gpumod.cli._builtin_modes_dir", return_value=modes_dir),
            patch("gpumod.cli._default_db_path", return_value=tmp_path / "test.db"),
        ):
            # Run 10 syncs
            for _ in range(10):
                async with cli_context():
                    pass

            # Final check
            async with cli_context() as ctx:
                services = await ctx.db.list_services()
                assert len(services) == 2
                assert {s.id for s in services} == {"a", "b"}


# ---------------------------------------------------------------------------
# AC2: MCP lifespan calls sync
# ---------------------------------------------------------------------------


class TestMCPLifespanAutoSync:
    """Tests for auto-sync in MCP lifespan — AC2."""

    async def test_mcp_lifespan_calls_sync(self, tmp_path: Path) -> None:
        """gpumod_lifespan() should call sync_presets() before yielding."""
        from fastmcp import FastMCP

        from gpumod.mcp_server import gpumod_lifespan

        mock_server = MagicMock(spec=FastMCP)

        with (
            patch("gpumod.mcp_server.get_settings") as mock_settings,
            patch("gpumod.mcp_server.sync_presets", new=AsyncMock()) as mock_sync_presets,
            patch("gpumod.mcp_server.sync_modes", new=AsyncMock()) as mock_sync_modes,
        ):
            mock_settings.return_value.db_path = tmp_path / "mcp.db"
            mock_settings.return_value.presets_dir = tmp_path / "presets"

            async with gpumod_lifespan(mock_server, db_path=tmp_path / "mcp.db"):
                pass

            mock_sync_presets.assert_awaited_once()
            mock_sync_modes.assert_awaited_once()

    async def test_mcp_lifespan_logs_sync_result(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MCP lifespan should log sync results at DEBUG level."""
        from unittest.mock import MagicMock

        from fastmcp import FastMCP

        from gpumod.mcp_server import gpumod_lifespan
        from gpumod.templates.presets import PresetSyncResult

        mock_server = MagicMock(spec=FastMCP)
        mock_result = PresetSyncResult(inserted=2, updated=1, unchanged=5, deleted=0)

        with (
            patch("gpumod.mcp_server.get_settings") as mock_settings,
            patch("gpumod.mcp_server.sync_presets", new=AsyncMock(return_value=mock_result)),
            patch("gpumod.mcp_server.sync_modes", new=AsyncMock()),
            caplog.at_level(logging.DEBUG),
        ):
            mock_settings.return_value.db_path = tmp_path / "mcp.db"
            mock_settings.return_value.presets_dir = tmp_path / "presets"

            async with gpumod_lifespan(mock_server, db_path=tmp_path / "mcp.db"):
                pass

            # Should log sync results
            log_text = " ".join(r.message for r in caplog.records)
            assert "inserted" in log_text.lower() or "sync" in log_text.lower()


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------


class TestAutoSyncLogging:
    """Tests for sync result logging."""

    async def test_sync_results_logged_at_debug_level(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Sync results should be logged at DEBUG level."""
        from gpumod.cli import cli_context
        from gpumod.templates.presets import PresetSyncResult

        mock_ctx = _make_mock_context()
        mock_result = PresetSyncResult(inserted=3, updated=0, unchanged=2, deleted=1)

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli.sync_presets", new=AsyncMock(return_value=mock_result)),
            patch("gpumod.cli.sync_modes", new=AsyncMock()),
            caplog.at_level(logging.DEBUG),
        ):
            async with cli_context():
                pass

            # Should log sync results
            log_text = " ".join(r.message for r in caplog.records)
            assert "3" in log_text or "inserted" in log_text.lower()
