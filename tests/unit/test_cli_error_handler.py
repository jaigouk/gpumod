"""Tests for gpumod.cli.error_handler exit-code propagation (gpumod-p2gj).

Before the fix, error_handler caught Exception, printed it, then returned
normally — so any CLI command using it exited 0 even when the underlying
operation failed. Automation using ``set -e`` was silently lied to.

After the fix, error_handler raises ``typer.Exit(code=1)`` after printing,
so the shell sees a non-zero exit code while the user still gets the
Rich-formatted error message.
"""

from __future__ import annotations

import typer
import typer.testing

from gpumod.cli import error_handler

runner = typer.testing.CliRunner()


class TestErrorHandlerExitCode:
    """error_handler must propagate caught exceptions as exit code != 0."""

    def test_caught_exception_exits_nonzero(self) -> None:
        app = typer.Typer()

        @app.command()
        def fail() -> None:
            with error_handler():
                raise RuntimeError("boom")

        result = runner.invoke(app, [])
        assert result.exit_code != 0
        assert "boom" in result.output or "boom" in (result.stderr or "")

    def test_no_exception_exits_zero(self) -> None:
        app = typer.Typer()

        @app.command()
        def ok() -> None:
            with error_handler():
                pass

        result = runner.invoke(app, [])
        assert result.exit_code == 0

    def test_typer_exit_propagates_existing_code(self) -> None:
        """typer.Exit(code=2) inside the block must propagate verbatim (not turn into 1)."""
        app = typer.Typer()

        @app.command()
        def explicit() -> None:
            with error_handler():
                raise typer.Exit(code=2)

        result = runner.invoke(app, [])
        assert result.exit_code == 2

    def test_keyboard_interrupt_propagates(self) -> None:
        """KeyboardInterrupt must propagate so the user can still Ctrl-C out."""
        app = typer.Typer()

        @app.command()
        def interrupted() -> None:
            with error_handler():
                raise KeyboardInterrupt

        result = runner.invoke(app, [])
        # CliRunner surfaces KeyboardInterrupt as a non-zero exit code.
        assert result.exit_code != 0
