"""Preset CLI commands -- gpumod preset sync."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

preset_app = typer.Typer(name="preset", help="Manage service presets.")

_console = Console()


def _run_async(coro):  # noqa: ANN001, ANN202
    """Run an async coroutine from synchronous CLI code."""
    import asyncio

    return asyncio.run(coro)


@preset_app.command()
def sync(
    db_path: str = typer.Option(
        None,
        "--db-path",
        help="Path to the SQLite database file.",
    ),
) -> None:
    """Sync YAML preset files into the database.

    Compares each preset YAML file against the DB and inserts new
    services or updates changed ones. Unchanged services are skipped.
    """
    from gpumod.cli import cli_context, error_handler
    from gpumod.templates.presets import sync_presets

    async def _cmd() -> None:
        resolved_path = Path(db_path) if db_path is not None else None
        async with cli_context(db_path=resolved_path) as ctx:
            result = await sync_presets(ctx.db, ctx.preset_loader)

            _console.print(
                f"[bold green]Preset sync:[/bold green] "
                f"{result.inserted} inserted, "
                f"{result.updated} updated, "
                f"{result.unchanged} unchanged, "
                f"{result.deleted} deleted."
            )

    with error_handler(console=_console):
        _run_async(_cmd())
