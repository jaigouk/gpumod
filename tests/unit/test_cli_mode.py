"""Tests for gpumod.cli_mode -- Mode CLI commands (list, status, switch, create)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import typer.testing

from gpumod.cli import app
from gpumod.models import Mode, ModeResult

runner = typer.testing.CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mode(
    *,
    id: str = "mode-1",
    name: str = "Test Mode",
    description: str | None = "A test mode",
    services: list[str] | None = None,
    total_vram_mb: int | None = 8192,
) -> Mode:
    return Mode(
        id=id,
        name=name,
        description=description,
        services=services or [],
        total_vram_mb=total_vram_mb,
    )


def _make_mode_result(
    *,
    success: bool = True,
    mode_id: str = "mode-1",
    started: list[str] | None = None,
    stopped: list[str] | None = None,
    message: str | None = None,
    errors: list[str] | None = None,
) -> ModeResult:
    return ModeResult(
        success=success,
        mode_id=mode_id,
        started=started or [],
        stopped=stopped or [],
        message=message,
        errors=errors or [],
    )


def _make_mock_context() -> MagicMock:
    ctx = MagicMock()
    ctx.db = MagicMock()
    ctx.db.list_modes = AsyncMock(return_value=[])
    ctx.db.get_mode = AsyncMock(return_value=None)
    ctx.db.get_current_mode = AsyncMock(return_value=None)
    ctx.db.insert_mode = AsyncMock()
    ctx.db.set_mode_services = AsyncMock()
    ctx.db.get_service = AsyncMock(return_value=None)
    ctx.db.close = AsyncMock()
    ctx.manager = MagicMock()
    ctx.manager.switch_mode = AsyncMock()
    ctx.registry = MagicMock()
    ctx.registry.list_all = AsyncMock(return_value=[])
    return ctx


# ---------------------------------------------------------------------------
# mode list tests
# ---------------------------------------------------------------------------


class TestModeList:
    """Tests for `gpumod mode list` command."""

    def test_mode_list_shows_all_modes(self) -> None:
        mode1 = _make_mode(id="mode-1", name="Dev Mode", description="Development")
        mode2 = _make_mode(id="mode-2", name="Prod Mode", description="Production")
        mock_ctx = _make_mock_context()
        mock_ctx.db.list_modes = AsyncMock(return_value=[mode1, mode2])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["mode", "list"])

        assert result.exit_code == 0
        assert "mode-1" in result.output
        assert "mode-2" in result.output
        assert "Dev Mode" in result.output
        assert "Prod Mode" in result.output

    def test_mode_list_json_flag(self) -> None:
        mode1 = _make_mode(id="mode-1", name="Dev Mode", total_vram_mb=4096)
        mode2 = _make_mode(id="mode-2", name="Prod Mode", total_vram_mb=8192)
        mock_ctx = _make_mock_context()
        mock_ctx.db.list_modes = AsyncMock(return_value=[mode1, mode2])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["mode", "list", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["id"] == "mode-1"
        assert parsed[1]["id"] == "mode-2"

    def test_mode_list_empty_shows_message(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.db.list_modes = AsyncMock(return_value=[])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["mode", "list"])

        assert result.exit_code == 0
        assert "no modes" in result.output.lower()


# ---------------------------------------------------------------------------
# mode status tests
# ---------------------------------------------------------------------------


class TestModeStatus:
    """Tests for `gpumod mode status` command."""

    def test_mode_status_shows_current_mode(self) -> None:
        mode = _make_mode(
            id="mode-1",
            name="Dev Mode",
            description="Development mode",
            services=["svc-1", "svc-2"],
        )
        mock_ctx = _make_mock_context()
        mock_ctx.db.get_current_mode = AsyncMock(return_value="mode-1")
        mock_ctx.db.get_mode = AsyncMock(return_value=mode)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["mode", "status"])

        assert result.exit_code == 0
        assert "Dev Mode" in result.output
        assert "mode-1" in result.output

    def test_mode_status_no_active_mode(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.db.get_current_mode = AsyncMock(return_value=None)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["mode", "status"])

        assert result.exit_code == 0
        assert "no active mode" in result.output.lower()

    def test_mode_status_json_flag(self) -> None:
        mode = _make_mode(
            id="mode-1",
            name="Dev Mode",
            description="Development mode",
            services=["svc-1", "svc-2"],
            total_vram_mb=8192,
        )
        mock_ctx = _make_mock_context()
        mock_ctx.db.get_current_mode = AsyncMock(return_value="mode-1")
        mock_ctx.db.get_mode = AsyncMock(return_value=mode)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["mode", "status", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["id"] == "mode-1"
        assert parsed["name"] == "Dev Mode"


# ---------------------------------------------------------------------------
# mode switch tests
# ---------------------------------------------------------------------------


class TestModeSwitch:
    """Tests for `gpumod mode switch <mode_id>` command."""

    def test_mode_switch_calls_switch_mode(self) -> None:
        mode_result = _make_mode_result(
            success=True, mode_id="mode-1", message="Switched to mode-1"
        )
        mock_ctx = _make_mock_context()
        mock_ctx.manager.switch_mode = AsyncMock(return_value=mode_result)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["mode", "switch", "mode-1"])

        assert result.exit_code == 0
        mock_ctx.manager.switch_mode.assert_awaited_once_with("mode-1")

    def test_mode_switch_shows_started_stopped(self) -> None:
        mode_result = _make_mode_result(
            success=True,
            mode_id="mode-1",
            started=["svc-1", "svc-2"],
            stopped=["svc-3"],
        )
        mock_ctx = _make_mock_context()
        mock_ctx.manager.switch_mode = AsyncMock(return_value=mode_result)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["mode", "switch", "mode-1"])

        assert result.exit_code == 0
        assert "svc-1" in result.output
        assert "svc-2" in result.output
        assert "svc-3" in result.output

    def test_mode_switch_failure_shows_errors(self) -> None:
        mode_result = _make_mode_result(
            success=False,
            mode_id="mode-1",
            errors=["Failed to start svc-1", "VRAM exceeded"],
        )
        mock_ctx = _make_mock_context()
        mock_ctx.manager.switch_mode = AsyncMock(return_value=mode_result)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["mode", "switch", "mode-1"])

        assert result.exit_code == 0
        assert "Failed to start svc-1" in result.output
        assert "VRAM exceeded" in result.output

    def test_mode_switch_json_flag(self) -> None:
        mode_result = _make_mode_result(
            success=True,
            mode_id="mode-1",
            started=["svc-1"],
            stopped=["svc-2"],
            message="Switched",
        )
        mock_ctx = _make_mock_context()
        mock_ctx.manager.switch_mode = AsyncMock(return_value=mode_result)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["mode", "switch", "mode-1", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["success"] is True
        assert parsed["mode_id"] == "mode-1"
        assert "svc-1" in parsed["started"]
        assert "svc-2" in parsed["stopped"]


# ---------------------------------------------------------------------------
# mode create tests
# ---------------------------------------------------------------------------


class TestModeCreate:
    """Tests for `gpumod mode create` command."""

    def test_mode_create_inserts_mode(self) -> None:
        mock_ctx = _make_mock_context()
        # Services exist
        mock_svc = MagicMock()
        mock_svc.id = "svc-1"
        mock_ctx.db.get_service = AsyncMock(return_value=mock_svc)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(
                app,
                [
                    "mode",
                    "create",
                    "Dev Mode",
                    "--services",
                    "svc-1",
                    "--description",
                    "Development mode",
                ],
            )

        assert result.exit_code == 0
        mock_ctx.db.insert_mode.assert_awaited_once()
        # Check the inserted mode
        call_args = mock_ctx.db.insert_mode.call_args
        inserted_mode = call_args[0][0]
        assert inserted_mode.name == "Dev Mode"
        assert inserted_mode.description == "Development mode"
        mock_ctx.db.set_mode_services.assert_awaited_once()

    def test_mode_create_validates_service_ids(self) -> None:
        mock_ctx = _make_mock_context()
        # Service does NOT exist
        mock_ctx.db.get_service = AsyncMock(return_value=None)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(
                app,
                [
                    "mode",
                    "create",
                    "Bad Mode",
                    "--services",
                    "nonexistent-svc",
                ],
            )

        assert result.exit_code == 0
        # Should show error about nonexistent service
        assert "nonexistent-svc" in result.output.lower()
        # Should NOT have inserted the mode
        mock_ctx.db.insert_mode.assert_not_awaited()
