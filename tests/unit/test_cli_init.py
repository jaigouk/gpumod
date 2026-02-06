"""Tests for gpumod.cli — CLI foundation (Typer app skeleton)."""

from __future__ import annotations

import json
from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import typer
import typer.testing

if TYPE_CHECKING:
    from pathlib import Path

runner = typer.testing.CliRunner()


# ---------------------------------------------------------------------------
# App structure tests
# ---------------------------------------------------------------------------


class TestAppStructure:
    """Verify the Typer app and its registered subcommand groups."""

    def test_app_is_typer_instance(self) -> None:
        from gpumod.cli import app

        assert isinstance(app, typer.Typer)

    def test_app_has_service_subcommand(self) -> None:
        from gpumod.cli import app

        result = runner.invoke(app, ["service", "--help"])
        assert result.exit_code == 0
        assert "service" in result.output.lower()

    def test_app_has_mode_subcommand(self) -> None:
        from gpumod.cli import app

        result = runner.invoke(app, ["mode", "--help"])
        assert result.exit_code == 0
        assert "mode" in result.output.lower()

    def test_app_has_template_subcommand(self) -> None:
        from gpumod.cli import app

        result = runner.invoke(app, ["template", "--help"])
        assert result.exit_code == 0
        assert "template" in result.output.lower()

    def test_app_has_model_subcommand(self) -> None:
        from gpumod.cli import app

        result = runner.invoke(app, ["model", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower()


# ---------------------------------------------------------------------------
# Placeholder command tests
# ---------------------------------------------------------------------------


class TestPlaceholderCommands:
    """Verify placeholder status and init commands exist."""

    def test_status_command_exists(self) -> None:
        from gpumod.cli import app

        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0

    def test_init_command_exists(self) -> None:
        from gpumod.cli import app

        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# json_output helper tests
# ---------------------------------------------------------------------------


class TestJsonOutput:
    """Verify the json_output helper function."""

    def test_json_output_renders_dict_as_json(self, capsys: object) -> None:
        from gpumod.cli import json_output

        data = {"name": "test", "count": 42}
        result = json_output(data, as_json=True)
        assert result is None

        # json_output prints to stdout when as_json is True
        out = StringIO()
        with patch("sys.stdout", out):
            json_output(data, as_json=True)
        output = out.getvalue()
        parsed = json.loads(output)
        assert parsed == data

    def test_json_output_without_flag_returns_data(self) -> None:
        from gpumod.cli import json_output

        data = {"name": "test", "count": 42}
        result = json_output(data, as_json=False)
        assert result == data


# ---------------------------------------------------------------------------
# error_handler context manager tests
# ---------------------------------------------------------------------------


class TestErrorHandler:
    """Verify the error_handler context manager."""

    def test_error_handler_catches_runtime_error(self) -> None:
        from gpumod.cli import error_handler

        console_mock = MagicMock()
        with error_handler(console=console_mock):
            msg = "something went wrong"
            raise RuntimeError(msg)

        # Should have printed an error via console, not raised
        console_mock.print.assert_called_once()
        call_args = str(console_mock.print.call_args)
        assert "something went wrong" in call_args

    def test_error_handler_lets_system_exit_through(self) -> None:
        from gpumod.cli import error_handler

        console_mock = MagicMock()
        with (
            error_handler(console=console_mock),
            typer.testing.CliRunner().isolated_filesystem(),
        ):
            # SystemExit should propagate, not be caught
            pass

        # Test that SystemExit actually propagates
        import pytest

        with pytest.raises(SystemExit), error_handler(console=console_mock):
            raise SystemExit(1)


# ---------------------------------------------------------------------------
# run_async helper tests
# ---------------------------------------------------------------------------


class TestRunAsync:
    """Verify the run_async helper."""

    def test_run_async_helper_runs_coroutine(self) -> None:
        from gpumod.cli import run_async

        async def _coro() -> int:
            return 42

        result = run_async(_coro())
        assert result == 42


# ---------------------------------------------------------------------------
# create_context tests
# ---------------------------------------------------------------------------


class TestCreateContext:
    """Verify AppContext factory function."""

    def test_create_context_returns_app_context(self, tmp_path: Path) -> None:
        from gpumod.cli import AppContext, create_context, run_async

        db_path = tmp_path / "test.db"

        ctx = run_async(create_context(db_path=db_path))

        assert isinstance(ctx, AppContext)
        assert ctx.db is not None
        assert ctx.registry is not None
        assert ctx.lifecycle is not None
        assert ctx.vram is not None
        assert ctx.sleep is not None
        assert ctx.manager is not None
        assert ctx.model_registry is not None
        assert ctx.template_engine is not None
        assert ctx.preset_loader is not None

        # Clean up — close the database
        run_async(ctx.db.close())


# ---------------------------------------------------------------------------
# __main__ module tests
# ---------------------------------------------------------------------------


class TestMainModule:
    """Verify __main__.py entry point."""

    def test_main_module_invokes_app(self) -> None:
        """Verify that __main__.py imports and invokes app."""
        import importlib

        # Patch app() so it doesn't actually run the CLI
        with patch("gpumod.cli.app") as mock_app:
            # reload __main__ to trigger app()
            import gpumod.__main__

            mock_app.reset_mock()
            importlib.reload(gpumod.__main__)
            mock_app.assert_called_once()
