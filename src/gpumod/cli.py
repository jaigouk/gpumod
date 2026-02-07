"""CLI foundation for gpumod — Typer app skeleton with subcommand groups.

Defines the main Typer app, sub-apps for service/mode/template/model,
the AppContext dataclass for backend dependency injection, and helper utilities
(run_async, json_output, error_handler, cli_context).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from gpumod.cli_mode import mode_app
from gpumod.cli_model import model_app
from gpumod.cli_plan import plan_app
from gpumod.cli_service import service_app
from gpumod.cli_simulate import simulate_app
from gpumod.cli_template import template_app
from gpumod.config import get_settings
from gpumod.db import Database
from gpumod.registry import ModelRegistry
from gpumod.services.lifecycle import LifecycleManager
from gpumod.services.manager import ServiceManager
from gpumod.services.registry import ServiceRegistry
from gpumod.services.sleep import SleepController
from gpumod.services.vram import VRAMTracker
from gpumod.simulation import SimulationEngine
from gpumod.templates.engine import TemplateEngine
from gpumod.templates.presets import PresetLoader
from gpumod.visualization import StatusPanel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine, Generator


# ---------------------------------------------------------------------------
# AppContext — backend dependency container
# ---------------------------------------------------------------------------


@dataclass
class AppContext:
    """Container for all backend service dependencies.

    Provides a single object through which CLI commands can access
    all backend services without coupling to concrete instantiation.
    """

    db: Database
    registry: ServiceRegistry
    lifecycle: LifecycleManager
    vram: VRAMTracker
    sleep: SleepController
    manager: ServiceManager
    model_registry: ModelRegistry
    template_engine: TemplateEngine
    preset_loader: PresetLoader
    simulation: SimulationEngine


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------


def _default_db_path() -> Path:
    """Return the default database path from centralized settings."""
    return get_settings().db_path


def _builtin_presets_dir() -> Path:
    """Return the built-in presets directory from centralized settings."""
    return get_settings().presets_dir


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


async def create_context(db_path: Path | None = None) -> AppContext:
    """Create and wire up all backend dependencies.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Defaults to
        ``~/.config/gpumod/gpumod.db``.

    Returns
    -------
    AppContext
        A fully-initialized context with all backend services.
    """
    resolved_db_path = db_path if db_path is not None else _default_db_path()

    # Ensure parent directory exists.
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)

    db = Database(resolved_db_path)
    await db.connect()

    registry = ServiceRegistry(db)
    lifecycle = LifecycleManager(registry)
    vram = VRAMTracker()
    sleep = SleepController(registry)
    manager = ServiceManager(
        db=db,
        registry=registry,
        lifecycle=lifecycle,
        vram=vram,
        sleep=sleep,
    )
    model_registry = ModelRegistry(db)
    template_engine = TemplateEngine()

    presets_dir = _builtin_presets_dir()
    preset_dirs: list[Path] = []
    if presets_dir.is_dir():
        preset_dirs.append(presets_dir)
    preset_loader = PresetLoader(preset_dirs=preset_dirs)

    simulation = SimulationEngine(db=db, vram=vram, model_registry=model_registry)

    return AppContext(
        db=db,
        registry=registry,
        lifecycle=lifecycle,
        vram=vram,
        sleep=sleep,
        manager=manager,
        model_registry=model_registry,
        template_engine=template_engine,
        preset_loader=preset_loader,
        simulation=simulation,
    )


# ---------------------------------------------------------------------------
# cli_context — async context manager for CLI commands
# ---------------------------------------------------------------------------


@asynccontextmanager
async def cli_context(
    db_path: Path | None = None,
) -> AsyncGenerator[AppContext, None]:
    """Async context manager that creates an AppContext and closes the DB.

    Replaces the repeated pattern of calling create_context() in a
    try/finally with an explicit ctx.db.close() call.

    Parameters
    ----------
    db_path:
        Optional path forwarded to :func:`create_context`.

    Yields
    ------
    AppContext
        A fully-initialized context with all backend services.
    """
    ctx = await create_context(db_path=db_path)
    try:
        yield ctx
    finally:
        await ctx.db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_T = TypeVar("_T")


def run_async(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an async coroutine from synchronous Typer command code.

    Parameters
    ----------
    coro:
        An awaitable coroutine to execute.

    Returns
    -------
    _T
        The return value of the coroutine, preserving the original type.
    """
    return asyncio.run(coro)


def json_output(data: Any, *, as_json: bool) -> Any:  # noqa: ANN401
    """Conditionally print data as JSON or return it for Rich formatting.

    Parameters
    ----------
    data:
        The data to output.
    as_json:
        If True, print as formatted JSON to stdout and return None.
        If False, return data unchanged for Rich table rendering.

    Returns
    -------
    Any | None
        None if printed as JSON, otherwise the original data.
    """
    if as_json:
        print(json.dumps(data, indent=2, default=str))  # noqa: T201
        return None
    return data


@contextmanager
def error_handler(
    console: Console | None = None,
) -> Generator[None, None, None]:
    """Context manager that catches exceptions and prints Rich-formatted errors.

    SystemExit and KeyboardInterrupt are allowed to propagate.

    Parameters
    ----------
    console:
        Optional Rich Console instance for output. Creates a new one if None.
    """
    if console is None:
        console = Console(stderr=True)
    try:
        yield
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="gpumod",
    help="GPU Service Manager for ML workloads.",
    no_args_is_help=True,
)


def run_cli() -> None:
    """Entry point for the ``gpumod`` console script."""
    app()


app.add_typer(service_app, name="service")
app.add_typer(mode_app, name="mode")
app.add_typer(template_app, name="template")
app.add_typer(model_app, name="model")
app.add_typer(simulate_app, name="simulate")
app.add_typer(plan_app, name="plan")


# ---------------------------------------------------------------------------
# State color mapping for Rich table rendering
# ---------------------------------------------------------------------------

_STATE_COLORS: dict[str, str] = {
    "running": "green",
    "starting": "green",
    "sleeping": "yellow",
    "stopping": "yellow",
    "unhealthy": "red",
    "failed": "red",
    "stopped": "dim",
    "unknown": "dim",
}


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------


@app.command()
def status(
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
    visual: bool = typer.Option(False, "--visual", help="Show VRAM bar visualization."),
) -> None:
    """Show system status (GPU, VRAM, services)."""
    console = Console()

    async def _cmd() -> None:
        async with cli_context() as ctx:
            system_status = await ctx.manager.get_status()

            if as_json:
                json_output(
                    system_status.model_dump(mode="json"),
                    as_json=True,
                )
                return

            if visual:
                panel = StatusPanel()
                rendered = panel.render(system_status)
                console.print(rendered)
                return

            if system_status.gpu is None:
                console.print("[bold yellow]Warning:[/bold yellow] No GPU detected")
            else:
                gpu = system_status.gpu
                console.print(
                    f"[bold]GPU:[/bold] {gpu.name}  [dim]VRAM: {gpu.vram_total_mb} MB[/dim]"
                )

            if system_status.current_mode is not None:
                console.print(f"[bold]Mode:[/bold] {system_status.current_mode}")

            if system_status.services:
                table = Table(title="Services")
                table.add_column("Name", style="bold")
                table.add_column("State")
                table.add_column("VRAM (MB)", justify="right")
                table.add_column("Driver")

                for si in system_status.services:
                    state_val = si.status.state.value
                    color = _STATE_COLORS.get(state_val, "dim")
                    vram = (
                        si.status.vram_mb if si.status.vram_mb is not None else si.service.vram_mb
                    )
                    table.add_row(
                        si.service.name,
                        f"[{color}]{state_val}[/{color}]",
                        str(vram),
                        si.service.driver.value,
                    )
                console.print(table)
            else:
                console.print("[dim]No services registered.[/dim]")

    with error_handler(console=console):
        run_async(_cmd())


@app.command()
def tui() -> None:
    """Launch the interactive TUI dashboard."""
    from gpumod.tui import GpumodApp

    async def _run() -> None:
        async with cli_context() as ctx:
            tui_app = GpumodApp(ctx=ctx)
            await tui_app.run_async()

    run_async(_run())


@app.command()
def init(
    db_path: str = typer.Option(
        None,
        "--db-path",
        help="Path to the SQLite database file.",
    ),
    preset_dir: str = typer.Option(
        None,
        "--preset-dir",
        help="Additional directory to search for presets.",
    ),
) -> None:
    """Initialize the gpumod database and configuration."""
    console = Console()

    async def _cmd() -> None:
        resolved_path = Path(db_path) if db_path is not None else None
        async with cli_context(db_path=resolved_path) as ctx:
            presets = ctx.preset_loader.discover_presets()

            loaded = 0
            skipped = 0
            for preset in presets:
                svc = ctx.preset_loader.to_service(preset)
                try:
                    await ctx.db.insert_service(svc)
                    loaded += 1
                except sqlite3.IntegrityError:
                    skipped += 1
                    console.print(f"[dim]Skipped (already exists): {preset.name}[/dim]")

            console.print(
                f"[bold green]Initialized.[/bold green] "
                f"Found {len(presets)} preset(s), "
                f"loaded {loaded}, skipped {skipped}."
            )

    with error_handler(console=console):
        run_async(_cmd())
