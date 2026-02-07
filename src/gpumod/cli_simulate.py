"""Simulate CLI commands -- gpumod simulate mode|services."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from gpumod.visualization import ComparisonView

if TYPE_CHECKING:
    from gpumod.models import SimulationResult

simulate_app = typer.Typer(name="simulate", help="Simulate VRAM requirements.")

_console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTEXT_RE = re.compile(r"^([a-zA-Z0-9][a-zA-Z0-9_\-]{0,63})=(\d+)$")


def _parse_context_overrides(raw: list[str] | None) -> dict[str, int] | None:
    """Parse --context key=value pairs into a dict.

    Parameters
    ----------
    raw:
        Raw list of "service_id=context_tokens" strings from CLI.

    Returns
    -------
    dict[str, int] | None
        Parsed overrides or None if no input.

    Raises
    ------
    ValueError
        If any entry is malformed.
    """
    if not raw:
        return None

    overrides: dict[str, int] = {}
    for entry in raw:
        match = _CONTEXT_RE.match(entry)
        if not match:
            msg = f"Invalid context override format: {entry!r} (expected 'service_id=tokens')"
            raise ValueError(msg)
        overrides[match.group(1)] = int(match.group(2))
    return overrides or None


def _parse_csv(value: str) -> list[str]:
    """Split a comma-separated string into a list of stripped, non-empty strings."""
    return [s.strip() for s in value.split(",") if s.strip()]


def _render_result(result: SimulationResult, *, console: Console) -> None:
    """Render a SimulationResult as a Rich table to the console.

    Parameters
    ----------
    result:
        The simulation result to display.
    console:
        The Rich console for output.
    """
    # Summary line
    if result.fits:
        console.print(
            f"[bold green]Fits[/bold green]: "
            f"{result.proposed_usage_mb} / {result.gpu_total_mb} MB "
            f"(headroom: {result.headroom_mb} MB)"
        )
    else:
        console.print(
            f"[bold red]Does not fit[/bold red]: "
            f"{result.proposed_usage_mb} / {result.gpu_total_mb} MB "
            f"(over by {-result.headroom_mb} MB)"
        )

    # Services table
    if result.services:
        svc_table = Table(title="Services")
        svc_table.add_column("ID", style="cyan")
        svc_table.add_column("Name", style="bold")
        svc_table.add_column("Driver")
        svc_table.add_column("VRAM (MB)", justify="right")

        for svc in result.services:
            svc_table.add_row(
                svc.id,
                svc.name,
                svc.driver.value,
                str(svc.vram_mb),
            )
        console.print(svc_table)

    # Alternatives table (only when overflowing)
    if result.alternatives:
        console.print("\n[bold yellow]Alternatives:[/bold yellow]")
        alt_table = Table(title="Suggested Alternatives")
        alt_table.add_column("#", justify="right")
        alt_table.add_column("Strategy")
        alt_table.add_column("Description")
        alt_table.add_column("VRAM Saved (MB)", justify="right")
        alt_table.add_column("Projected (MB)", justify="right")

        for idx, alt in enumerate(result.alternatives, 1):
            fits_str = (
                "[green]fits[/green]"
                if alt.projected_total_mb <= result.gpu_total_mb
                else "[red]over[/red]"
            )
            alt_table.add_row(
                str(idx),
                alt.strategy,
                alt.description,
                str(alt.vram_saved_mb),
                f"{alt.projected_total_mb} {fits_str}",
            )
        console.print(alt_table)


def _render_visual(result: SimulationResult, *, console: Console) -> None:
    """Render a visual comparison view of current vs proposed VRAM.

    Parameters
    ----------
    result:
        The simulation result.
    console:
        The Rich console for output.
    """
    cv = ComparisonView()
    proposed_services = [(svc.name, svc.vram_mb, "running") for svc in result.services]
    rendered = cv.render(
        current_services=[],
        proposed_services=proposed_services,
        gpu_total_mb=result.gpu_total_mb,
    )
    console.print(rendered)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@simulate_app.command("mode")
def simulate_mode(
    mode: str = typer.Argument(help="Mode ID to simulate."),
    add: str | None = typer.Option(None, "--add", help="Comma-separated service IDs to add."),
    remove: str | None = typer.Option(
        None, "--remove", help="Comma-separated service IDs to remove."
    ),
    context_override: list[str] | None = typer.Option(  # noqa: B008
        None, "--context", help="Context overrides as service_id=tokens."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
    visual: bool = typer.Option(False, "--visual", help="Show VRAM bar visualization."),
) -> None:
    """Simulate VRAM usage for a mode with optional add/remove."""
    from gpumod.cli import cli_context, error_handler, json_output, run_async

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                # Parse comma-separated lists
                add_ids = _parse_csv(add) if add else None
                remove_ids = _parse_csv(remove) if remove else None
                context_overrides = _parse_context_overrides(context_override)

                result: SimulationResult = await ctx.simulation.simulate_mode(
                    mode,
                    add=add_ids,
                    remove=remove_ids,
                    context_overrides=context_overrides,
                )

                if as_json:
                    json_output(result.model_dump(mode="json"), as_json=True)
                    return

                if visual:
                    _render_visual(result, console=_console)
                    return

                _render_result(result, console=_console)

    run_async(_cmd())


@simulate_app.command("services")
def simulate_services(
    services: str = typer.Argument(help="Comma-separated service IDs to simulate."),
    context_override: list[str] | None = typer.Option(  # noqa: B008
        None, "--context", help="Context overrides as service_id=tokens."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
    visual: bool = typer.Option(False, "--visual", help="Show VRAM bar visualization."),
) -> None:
    """Simulate VRAM usage for an explicit list of services."""
    from gpumod.cli import cli_context, error_handler, json_output, run_async

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                service_ids = _parse_csv(services)
                context_overrides = _parse_context_overrides(context_override)

                result: SimulationResult = await ctx.simulation.simulate_services(
                    service_ids,
                    context_overrides=context_overrides,
                )

                if as_json:
                    json_output(result.model_dump(mode="json"), as_json=True)
                    return

                if visual:
                    _render_visual(result, console=_console)
                    return

                _render_result(result, console=_console)

    run_async(_cmd())
