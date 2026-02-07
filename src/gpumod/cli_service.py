"""Service CLI commands -- gpumod service list|status|start|stop."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from gpumod.models import Service, ServiceStatus

service_app = typer.Typer(name="service", help="Manage GPU services.")

_console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _service_to_dict(svc: Service, status: ServiceStatus) -> dict[str, Any]:
    """Convert a service + status pair to a JSON-serialisable dict."""
    return {
        "id": svc.id,
        "name": svc.name,
        "driver": str(svc.driver),
        "port": svc.port,
        "vram_mb": svc.vram_mb,
        "status": {
            "state": str(status.state),
            "vram_mb": status.vram_mb,
            "uptime_seconds": status.uptime_seconds,
            "health_ok": status.health_ok,
            "sleep_level": status.sleep_level,
            "last_error": status.last_error,
        },
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@service_app.command("list")
def list_services(
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List all registered GPU services."""
    from gpumod.cli import cli_context, error_handler, json_output, run_async

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                services: list[Service] = await ctx.registry.list_all()

                if not services:
                    if as_json:
                        json_output([], as_json=True)
                    else:
                        _console.print("[dim]No services registered.[/dim]")
                    return

                # Fetch live state for each service
                rows: list[dict[str, Any]] = []
                for svc in services:
                    driver = ctx.registry.get_driver(svc.driver)
                    status: ServiceStatus = await driver.status(svc)
                    rows.append(_service_to_dict(svc, status))

                if json_output(rows, as_json=as_json) is None:
                    return

                # Rich table output
                table = Table(title="GPU Services")
                table.add_column("ID", style="cyan")
                table.add_column("Name", style="bold")
                table.add_column("Driver")
                table.add_column("Port", justify="right")
                table.add_column("VRAM (MB)", justify="right")
                table.add_column("State")

                for row in rows:
                    state_str = row["status"]["state"]
                    state_style = _state_style(state_str)
                    table.add_row(
                        row["id"],
                        row["name"],
                        row["driver"],
                        str(row["port"]) if row["port"] is not None else "-",
                        str(row["vram_mb"]),
                        f"[{state_style}]{state_str}[/{state_style}]",
                    )

                _console.print(table)

    run_async(_cmd())


@service_app.command("status")
def service_status(
    service_id: str = typer.Argument(help="Service ID to check."),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show detailed status of a GPU service."""
    from gpumod.cli import cli_context, error_handler, json_output, run_async

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                svc: Service = await ctx.registry.get(service_id)
                driver = ctx.registry.get_driver(svc.driver)
                status: ServiceStatus = await driver.status(svc)

                data = _service_to_dict(svc, status)
                if json_output(data, as_json=as_json) is None:
                    return

                # Rich panel output
                state_str = str(status.state)
                state_style = _state_style(state_str)
                lines = [
                    f"[bold]Name:[/bold]    {svc.name}",
                    f"[bold]Driver:[/bold]  {svc.driver}",
                    f"[bold]Port:[/bold]    {svc.port or '-'}",
                    f"[bold]VRAM:[/bold]    {svc.vram_mb} MB",
                    f"[bold]State:[/bold]   [{state_style}]{state_str}[/{state_style}]",
                ]
                if status.uptime_seconds is not None:
                    lines.append(f"[bold]Uptime:[/bold]  {status.uptime_seconds}s")
                if status.health_ok is not None:
                    health_str = "OK" if status.health_ok else "FAIL"
                    lines.append(f"[bold]Health:[/bold]  {health_str}")
                if status.last_error is not None:
                    lines.append(f"[bold]Error:[/bold]   {status.last_error}")

                panel = Panel("\n".join(lines), title=f"Service: {svc.id}", border_style="blue")
                _console.print(panel)

    run_async(_cmd())


@service_app.command("start")
def start_service(
    service_id: str = typer.Argument(help="Service ID to start."),
) -> None:
    """Start a GPU service."""
    from gpumod.cli import cli_context, error_handler, run_async

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                await ctx.lifecycle.start(service_id)
                _console.print(
                    f"[green]Started service [bold]{service_id}[/bold] successfully.[/green]"
                )

    run_async(_cmd())


@service_app.command("stop")
def stop_service(
    service_id: str = typer.Argument(help="Service ID to stop."),
) -> None:
    """Stop a GPU service."""
    from gpumod.cli import cli_context, error_handler, run_async

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                await ctx.lifecycle.stop(service_id)
                _console.print(
                    f"[yellow]Stopped service [bold]{service_id}[/bold] successfully.[/yellow]"
                )

    run_async(_cmd())


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------


def _state_style(state: str) -> str:
    """Return a Rich style string for a service state."""
    styles: dict[str, str] = {
        "running": "green",
        "stopped": "dim",
        "starting": "yellow",
        "sleeping": "cyan",
        "unhealthy": "red",
        "stopping": "yellow",
        "failed": "bold red",
        "unknown": "dim",
    }
    return styles.get(state, "white")
