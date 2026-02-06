"""Mode CLI commands -- gpumod mode list|status|switch|create."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from gpumod.cli import AppContext
    from gpumod.models import Mode, ModeResult

mode_app = typer.Typer(name="mode", help="Manage service modes.")

_console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_context() -> AppContext:
    """Create and return the AppContext via run_async."""
    from gpumod.cli import create_context, run_async

    return run_async(create_context())  # type: ignore[no-any-return]


def _close_db(ctx: AppContext) -> None:
    """Close the database connection."""
    from gpumod.cli import run_async

    run_async(ctx.db.close())


def _mode_to_dict(mode: Mode) -> dict[str, Any]:
    """Convert a Mode to a JSON-serialisable dict."""
    return {
        "id": mode.id,
        "name": mode.name,
        "description": mode.description,
        "services": mode.services,
        "total_vram_mb": mode.total_vram_mb,
    }


def _mode_result_to_dict(result: ModeResult) -> dict[str, Any]:
    """Convert a ModeResult to a JSON-serialisable dict."""
    return {
        "success": result.success,
        "mode_id": result.mode_id,
        "started": result.started,
        "stopped": result.stopped,
        "message": result.message,
        "errors": result.errors,
    }


def _slugify(name: str) -> str:
    """Generate a mode ID from a human-readable name.

    Lowercases, replaces non-alphanumeric chars with hyphens, and strips
    leading/trailing hyphens.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower())
    return slug.strip("-")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@mode_app.command("list")
def list_modes(
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List all registered service modes."""
    from gpumod.cli import error_handler, json_output, run_async

    ctx = _get_context()
    try:
        with error_handler(console=_console):
            modes: list[Mode] = run_async(ctx.db.list_modes())

            if not modes:
                if as_json:
                    json_output([], as_json=True)
                else:
                    _console.print("[dim]No modes defined.[/dim]")
                return

            rows = [_mode_to_dict(m) for m in modes]

            if json_output(rows, as_json=as_json) is None:
                return

            # Rich table output
            table = Table(title="Service Modes")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="bold")
            table.add_column("Description")
            table.add_column("Total VRAM (MB)", justify="right")

            for row in rows:
                table.add_row(
                    row["id"],
                    row["name"],
                    row["description"] or "-",
                    str(row["total_vram_mb"]) if row["total_vram_mb"] is not None else "-",
                )

            _console.print(table)
    finally:
        _close_db(ctx)


@mode_app.command("status")
def mode_status(
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show the currently active service mode."""
    from gpumod.cli import error_handler, json_output, run_async

    ctx = _get_context()
    try:
        with error_handler(console=_console):
            current_mode_id: str | None = run_async(ctx.db.get_current_mode())

            if current_mode_id is None:
                if as_json:
                    json_output({"active": False, "mode": None}, as_json=True)
                else:
                    _console.print("[dim]No active mode.[/dim]")
                return

            mode: Mode | None = run_async(ctx.db.get_mode(current_mode_id))
            if mode is None:
                _console.print(
                    f"[yellow]Current mode '{current_mode_id}' not found in database.[/yellow]"
                )
                return

            data = _mode_to_dict(mode)
            if json_output(data, as_json=as_json) is None:
                return

            # Rich panel output
            lines = [
                f"[bold]Name:[/bold]        {mode.name}",
                f"[bold]Description:[/bold] {mode.description or '-'}",
                f"[bold]Services:[/bold]    {', '.join(mode.services) if mode.services else '-'}",
                f"[bold]Total VRAM:[/bold]  {mode.total_vram_mb} MB"
                if mode.total_vram_mb is not None
                else "-",
            ]

            panel = Panel(
                "\n".join(lines),
                title=f"Active Mode: {mode.id}",
                border_style="green",
            )
            _console.print(panel)
    finally:
        _close_db(ctx)


@mode_app.command("switch")
def switch_mode(
    mode_id: str = typer.Argument(help="Mode ID to switch to."),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Switch to a different service mode."""
    from gpumod.cli import error_handler, json_output, run_async

    ctx = _get_context()
    try:
        with error_handler(console=_console):
            result: ModeResult = run_async(ctx.manager.switch_mode(mode_id))

            data = _mode_result_to_dict(result)
            if json_output(data, as_json=as_json) is None:
                return

            # Rich output
            if result.success:
                _console.print(
                    f"[green]Switched to mode [bold]{result.mode_id}[/bold] successfully.[/green]"
                )
            else:
                _console.print(
                    f"[red]Failed to switch to mode [bold]{result.mode_id}[/bold].[/red]"
                )

            if result.message:
                _console.print(f"  {result.message}")

            if result.started:
                _console.print("[green]Started:[/green]")
                for svc in result.started:
                    _console.print(f"  [green]+[/green] {svc}")

            if result.stopped:
                _console.print("[red]Stopped:[/red]")
                for svc in result.stopped:
                    _console.print(f"  [red]-[/red] {svc}")

            if result.errors:
                _console.print("[yellow]Errors:[/yellow]")
                for err in result.errors:
                    _console.print(f"  [yellow]![/yellow] {err}")
    finally:
        _close_db(ctx)


@mode_app.command("create")
def create_mode(
    name: str = typer.Argument(help="Name for the new mode."),
    services: str = typer.Option("", "--services", help="Comma-separated service IDs."),
    description: str = typer.Option("", "--description", help="Mode description."),
) -> None:
    """Create a new service mode."""
    from gpumod.cli import error_handler, run_async
    from gpumod.models import Mode as ModeModel

    ctx = _get_context()
    try:
        with error_handler(console=_console):
            # Parse service IDs
            service_ids: list[str] = [s.strip() for s in services.split(",") if s.strip()]

            # Validate service IDs exist
            invalid_ids: list[str] = []
            for sid in service_ids:
                svc = run_async(ctx.db.get_service(sid))
                if svc is None:
                    invalid_ids.append(sid)

            if invalid_ids:
                _console.print(f"[red]Error: Services not found: {', '.join(invalid_ids)}[/red]")
                return

            # Generate mode ID from name
            mode_id = _slugify(name)

            mode = ModeModel(
                id=mode_id,
                name=name,
                description=description or None,
                services=service_ids,
            )

            run_async(ctx.db.insert_mode(mode))

            if service_ids:
                run_async(ctx.db.set_mode_services(mode_id, service_ids))

            _console.print(f"[green]Created mode [bold]{mode_id}[/bold] ({name}).[/green]")
            if service_ids:
                _console.print(f"  Services: {', '.join(service_ids)}")
    finally:
        _close_db(ctx)
