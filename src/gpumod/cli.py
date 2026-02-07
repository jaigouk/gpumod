"""CLI foundation for gpumod — Typer app skeleton with subcommand groups.

Defines the main Typer app, sub-apps for service/mode/template/model,
the AppContext dataclass for backend dependency injection, and helper utilities
(run_async, json_output, error_handler).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.table import Table

from gpumod.cli_mode import mode_app
from gpumod.cli_model import model_app
from gpumod.cli_service import service_app
from gpumod.cli_simulate import simulate_app
from gpumod.cli_template import template_app
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
    from collections.abc import Generator


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

_DEFAULT_DB_PATH = Path.home() / ".config" / "gpumod" / "gpumod.db"
_BUILTIN_PRESETS_DIR = Path(__file__).parent.parent.parent / "presets"


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
    resolved_db_path = db_path if db_path is not None else _DEFAULT_DB_PATH

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

    preset_dirs: list[Path] = []
    if _BUILTIN_PRESETS_DIR.is_dir():
        preset_dirs.append(_BUILTIN_PRESETS_DIR)
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
# Helpers
# ---------------------------------------------------------------------------


def run_async(coro: Any) -> Any:  # noqa: ANN401
    """Run an async coroutine from synchronous Typer command code.

    Parameters
    ----------
    coro:
        An awaitable coroutine to execute.

    Returns
    -------
    Any
        The return value of the coroutine.
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

app.add_typer(service_app, name="service")
app.add_typer(mode_app, name="mode")
app.add_typer(template_app, name="template")
app.add_typer(model_app, name="model")
app.add_typer(simulate_app, name="simulate")


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
    with error_handler(console=console):
        ctx = run_async(create_context())
        try:
            system_status = run_async(ctx.manager.get_status())

            # JSON output mode
            if as_json:
                json_output(
                    system_status.model_dump(mode="json"),
                    as_json=True,
                )
                return

            # Visual mode — use StatusPanel from visualization module
            if visual:
                panel = StatusPanel()
                rendered = panel.render(system_status)
                console.print(rendered)
                return

            # Default Rich table mode
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
        finally:
            run_async(ctx.db.close())


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
    with error_handler(console=console):
        resolved_path = Path(db_path) if db_path is not None else None
        ctx = run_async(create_context(db_path=resolved_path))
        try:
            # Discover presets
            presets = ctx.preset_loader.discover_presets()

            # Load each preset into the database
            loaded = 0
            skipped = 0
            for preset in presets:
                svc = ctx.preset_loader.to_service(preset)
                try:
                    run_async(ctx.db.insert_service(svc))
                    loaded += 1
                except sqlite3.IntegrityError:
                    skipped += 1
                    console.print(f"[dim]Skipped (already exists): {preset.name}[/dim]")

            # Summary
            console.print(
                f"[bold green]Initialized.[/bold green] "
                f"Found {len(presets)} preset(s), "
                f"loaded {loaded}, skipped {skipped}."
            )
        finally:
            run_async(ctx.db.close())
