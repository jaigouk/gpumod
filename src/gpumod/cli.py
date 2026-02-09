"""CLI foundation for gpumod — Typer app skeleton with subcommand groups.

Defines the main Typer app, sub-apps for service/mode/template/model,
the AppContext dataclass for backend dependency injection, and helper utilities
(run_async, json_output, error_handler, cli_context).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.table import Table

from gpumod.cli_mode import mode_app
from gpumod.cli_model import model_app
from gpumod.cli_plan import plan_app
from gpumod.cli_preset import preset_app
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
from gpumod.services.unit_installer import UnitFileInstaller
from gpumod.services.vram import VRAMTracker
from gpumod.simulation import SimulationEngine
from gpumod.templates.engine import TemplateEngine
from gpumod.templates.modes import ModeLoader, sync_modes
from gpumod.templates.presets import PresetLoader, sync_presets
from gpumod.visualization import StatusPanel

logger = logging.getLogger(__name__)

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
    mode_loader: ModeLoader
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


def _builtin_modes_dir() -> Path:
    """Return the built-in modes directory from centralized settings."""
    return get_settings().modes_dir


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
    unit_installer = UnitFileInstaller(db=db, template_engine=TemplateEngine())
    lifecycle = LifecycleManager(registry, unit_installer=unit_installer)
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

    modes_dir = _builtin_modes_dir()
    mode_dirs: list[Path] = []
    if modes_dir.is_dir():
        mode_dirs.append(modes_dir)
    mode_loader = ModeLoader(mode_dirs=mode_dirs)

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
        mode_loader=mode_loader,
        simulation=simulation,
    )


# ---------------------------------------------------------------------------
# cli_context — async context manager for CLI commands
# ---------------------------------------------------------------------------


@asynccontextmanager
async def cli_context(
    db_path: Path | None = None,
    *,
    no_sync: bool = False,
) -> AsyncGenerator[AppContext, None]:
    """Async context manager that creates an AppContext and closes the DB.

    Replaces the repeated pattern of calling create_context() in a
    try/finally with an explicit ctx.db.close() call.

    By default, syncs presets and modes from YAML files to the database
    before yielding. This ensures the DB always reflects the current
    state of YAML definitions.

    Parameters
    ----------
    db_path:
        Optional path forwarded to :func:`create_context`.
    no_sync:
        If True, skip auto-sync of presets and modes. Useful for
        read-only commands or when sync is not needed.

    Yields
    ------
    AppContext
        A fully-initialized context with all backend services.
    """
    ctx = await create_context(db_path=db_path)
    try:
        if not no_sync:
            try:
                preset_result = await sync_presets(ctx.db, ctx.preset_loader)
                logger.debug(
                    "Preset sync: %d inserted, %d updated, %d unchanged, %d deleted",
                    preset_result.inserted,
                    preset_result.updated,
                    preset_result.unchanged,
                    preset_result.deleted,
                )
            except Exception as exc:
                logger.warning("Preset sync failed: %s", exc)

            try:
                mode_result = await sync_modes(ctx.db, ctx.mode_loader)
                logger.debug(
                    "Mode sync: %d inserted, %d updated, %d unchanged, %d deleted",
                    mode_result.inserted,
                    mode_result.updated,
                    mode_result.unchanged,
                    mode_result.deleted,
                )
            except Exception as exc:
                logger.warning("Mode sync failed: %s", exc)

        yield ctx
    finally:
        await ctx.db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from synchronous Typer command code.

    Parameters
    ----------
    coro:
        An awaitable coroutine to execute.

    Returns
    -------
    T
        The return value of the coroutine, preserving the original type.
    """
    return asyncio.run(coro)


def json_output(data: Any, *, as_json: bool) -> Any:
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
        print(json.dumps(data, indent=2, default=str))
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
app.add_typer(preset_app, name="preset")
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
    no_sync: bool = typer.Option(False, "--no-sync", help="Skip auto-sync of presets and modes."),
) -> None:
    """Show system status (GPU, VRAM, services)."""
    console = Console()

    async def _cmd() -> None:
        async with cli_context(no_sync=no_sync) as ctx:
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
    mode_dir: str = typer.Option(
        None,
        "--mode-dir",
        help="Additional directory to search for mode definitions.",
    ),
) -> None:
    """Initialize the gpumod database and configuration."""
    console = Console()

    async def _cmd() -> None:
        resolved_path = Path(db_path) if db_path is not None else None
        async with cli_context(db_path=resolved_path) as ctx:
            presets = ctx.preset_loader.discover_presets()

            svc_loaded = 0
            svc_skipped = 0
            for preset in presets:
                svc = ctx.preset_loader.to_service(preset)
                try:
                    await ctx.db.insert_service(svc)
                    svc_loaded += 1
                except sqlite3.IntegrityError:
                    svc_skipped += 1
                    console.print(f"[dim]Skipped (already exists): {preset.name}[/dim]")

            console.print(
                f"[bold green]Services:[/bold green] "
                f"Found {len(presets)} preset(s), "
                f"loaded {svc_loaded}, skipped {svc_skipped}."
            )

            modes = ctx.mode_loader.discover_modes()
            mode_loaded = 0
            mode_skipped = 0
            for mode in modes:
                with contextlib.suppress(ValueError):
                    mode.total_vram_mb = ModeLoader.calculate_vram(mode, presets)
                try:
                    await ctx.db.insert_mode(mode)
                    mode_loaded += 1
                except sqlite3.IntegrityError:
                    mode_skipped += 1
                    console.print(f"[dim]Skipped (already exists): {mode.name}[/dim]")

                with contextlib.suppress(sqlite3.IntegrityError):
                    await ctx.db.set_mode_services(
                        mode.id, mode.services, list(range(len(mode.services)))
                    )

            console.print(
                f"[bold green]Modes:[/bold green] "
                f"Found {len(modes)} mode(s), "
                f"loaded {mode_loaded}, skipped {mode_skipped}."
            )

            console.print("[bold green]Initialized.[/bold green]")

    with error_handler(console=console):
        run_async(_cmd())


def _format_sync_summary(
    preset_result: Any,
    mode_result: Any,
) -> list[str]:
    """Build summary parts from sync results."""
    parts = []
    if preset_result.inserted > 0:
        parts.append(f"{preset_result.inserted} inserted")
    if preset_result.updated > 0:
        parts.append(f"{preset_result.updated} updated")
    if preset_result.deleted > 0:
        parts.append(f"{preset_result.deleted} deleted")
    if mode_result.inserted > 0:
        parts.append(f"{mode_result.inserted} modes inserted")
    if mode_result.updated > 0:
        parts.append(f"{mode_result.updated} modes updated")
    if mode_result.deleted > 0:
        parts.append(f"{mode_result.deleted} modes deleted")
    return parts


@app.command()
def watch(
    timeout: float = typer.Option(
        None,
        "--timeout",
        help="Stop watching after N seconds (for testing).",
    ),
    debounce: int = typer.Option(
        500,
        "--debounce",
        help="Debounce window in milliseconds.",
    ),
    no_sync: bool = typer.Option(
        False,
        "--no-sync",
        help="Skip initial sync before starting watcher.",
    ),
) -> None:
    """Watch preset/mode directories for changes and auto-sync.

    Monitors YAML files for changes and automatically syncs to the database.
    Useful for rapid iteration during development.
    """
    from gpumod.watcher import run_watcher

    console = Console()

    async def _run_watch() -> None:
        async with cli_context(no_sync=no_sync) as ctx:
            console.print("[bold]gpumod watch[/bold] starting...")
            console.print(f"  Preset dirs: {[str(d) for d in ctx.preset_loader.preset_dirs]}")
            console.print(f"  Mode dirs: {[str(d) for d in ctx.mode_loader.mode_dirs]}")
            console.print(f"  Debounce: {debounce}ms\n")
            console.print("[dim]Watching for changes. Press Ctrl+C to stop.[/dim]")

            async def do_sync() -> None:
                """Sync presets and modes, print summary."""
                try:
                    preset_result = await sync_presets(ctx.db, ctx.preset_loader)
                    mode_result = await sync_modes(ctx.db, ctx.mode_loader)
                    parts = _format_sync_summary(preset_result, mode_result)
                    if parts:
                        console.print(f"[green]Synced:[/green] {', '.join(parts)}")
                    else:
                        console.print("[dim]No changes detected.[/dim]")
                except Exception as exc:
                    console.print(f"[red]Sync error:[/red] {exc}")
                    logger.exception("Sync failed")

            try:
                await run_watcher(
                    preset_dirs=ctx.preset_loader.preset_dirs,
                    mode_dirs=ctx.mode_loader.mode_dirs,
                    sync_fn=do_sync,
                    debounce_ms=debounce,
                    watch_timeout=timeout,
                )
            except KeyboardInterrupt:
                console.print("\n[dim]Watcher stopped.[/dim]")

    with error_handler(console=console):
        run_async(_run_watch())
