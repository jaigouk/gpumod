"""Model CLI commands -- gpumod model list|info|register|remove."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from gpumod.models import ModelInfo

model_app = typer.Typer(name="model", help="Manage ML models.")

_console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_to_dict(model: ModelInfo) -> dict[str, Any]:
    """Convert a ModelInfo to a JSON-serialisable dict."""
    return {
        "id": model.id,
        "source": str(model.source),
        "parameters_b": model.parameters_b,
        "architecture": model.architecture,
        "base_vram_mb": model.base_vram_mb,
        "kv_cache_per_1k_tokens_mb": model.kv_cache_per_1k_tokens_mb,
        "quantizations": model.quantizations,
        "capabilities": model.capabilities,
        "fetched_at": model.fetched_at,
        "notes": model.notes,
    }


def _format_parameters(params: float | None) -> str:
    """Format parameter count nicely, e.g. 7.0 -> '7.0B'."""
    if params is None:
        return "-"
    return f"{params}B"


def _build_register_kwargs(
    *,
    file_path: str | None,
    vram: int | None,
    params: float | None,
    architecture: str | None,
    quant: str | None,
) -> dict[str, Any]:
    """Build kwargs dict for ModelRegistry.register()."""
    kwargs: dict[str, Any] = {}
    if file_path is not None:
        kwargs["file_path"] = file_path
    if vram is not None:
        kwargs["base_vram_mb"] = vram
    if params is not None:
        kwargs["parameters_b"] = params
    if architecture is not None:
        kwargs["architecture"] = architecture
    if quant is not None:
        kwargs["quant"] = quant
    return kwargs


def _print_registered(registered: ModelInfo) -> None:
    """Print registration result to console."""
    _console.print(
        f"[green]Registered model [bold]{registered.id}[/bold] successfully.[/green]"
    )
    _console.print(f"  Source: {registered.source}")
    if registered.parameters_b is not None:
        _console.print(f"  Parameters: {_format_parameters(registered.parameters_b)}")
    if registered.architecture is not None:
        _console.print(f"  Architecture: {registered.architecture}")
    if registered.base_vram_mb is not None:
        _console.print(f"  Base VRAM: {registered.base_vram_mb} MB")
    if registered.quantizations:
        _console.print(f"  Quantizations: {', '.join(registered.quantizations)}")
    if registered.notes:
        _console.print(f"  Notes: {registered.notes}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@model_app.command("list")
def list_models(
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List all registered ML models."""
    from gpumod.cli import cli_context, error_handler, json_output, run_async

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                models: list[ModelInfo] = await ctx.model_registry.list_models()

                if not models:
                    if as_json:
                        json_output([], as_json=True)
                    else:
                        _console.print("[dim]No models registered.[/dim]")
                    return

                rows = [_model_to_dict(m) for m in models]

                if json_output(rows, as_json=as_json) is None:
                    return

                # Rich table output
                table = Table(title="ML Models")
                table.add_column("ID", style="cyan")
                table.add_column("Source")
                table.add_column("Parameters")
                table.add_column("Architecture")
                table.add_column("Base VRAM (MB)", justify="right")
                table.add_column("KV/1K (MB)", justify="right")

                for row in rows:
                    table.add_row(
                        row["id"],
                        row["source"],
                        _format_parameters(row["parameters_b"]),
                        row["architecture"] or "-",
                        str(row["base_vram_mb"]) if row["base_vram_mb"] is not None else "-",
                        str(row["kv_cache_per_1k_tokens_mb"])
                        if row["kv_cache_per_1k_tokens_mb"] is not None
                        else "-",
                    )

                _console.print(table)

    run_async(_cmd())


@model_app.command("info")
def model_info(
    model_id: str = typer.Argument(help="Model ID to show."),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
    context_size: int = typer.Option(
        4096, "--context-size", help="Context size for VRAM estimation."
    ),
) -> None:
    """Show detailed information about a registered model."""
    from gpumod.cli import cli_context, error_handler, json_output, run_async

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                model: ModelInfo | None = await ctx.model_registry.get(model_id)

                if model is None:
                    msg = f"Model not found: {model_id!r}"
                    raise KeyError(msg)

                # Get VRAM estimate
                vram_estimate: int | None = None
                with contextlib.suppress(ValueError):
                    vram_estimate = await ctx.model_registry.estimate_vram(
                        model_id, context_size=context_size
                    )

                data = _model_to_dict(model)
                if vram_estimate is not None:
                    data["estimated_vram_mb"] = vram_estimate
                    data["context_size"] = context_size

                if json_output(data, as_json=as_json) is None:
                    return

                # Rich panel output
                lines = [
                    f"[bold]Source:[/bold]       {model.source}",
                    f"[bold]Parameters:[/bold]   {_format_parameters(model.parameters_b)}",
                    f"[bold]Architecture:[/bold] {model.architecture or '-'}",
                    f"[bold]Base VRAM:[/bold]    {model.base_vram_mb or '-'} MB",
                    f"[bold]KV/1K:[/bold]        {model.kv_cache_per_1k_tokens_mb or '-'} MB",
                ]

                if model.quantizations:
                    lines.append(f"[bold]Quantizations:[/bold] {', '.join(model.quantizations)}")
                if model.capabilities:
                    lines.append(f"[bold]Capabilities:[/bold]  {', '.join(model.capabilities)}")
                if model.fetched_at:
                    lines.append(f"[bold]Fetched at:[/bold]   {model.fetched_at}")
                if model.notes:
                    lines.append(f"[bold]Notes:[/bold]        {model.notes}")

                if vram_estimate is not None:
                    lines.append("")
                    lines.append(
                        f"[bold]Estimated VRAM ({context_size} ctx):[/bold] {vram_estimate} MB"
                    )

                panel = Panel(
                    "\n".join(lines),
                    title=f"Model: {model.id}",
                    border_style="blue",
                )
                _console.print(panel)

    run_async(_cmd())


@model_app.command("register")
def register_model(
    model_id: str = typer.Argument(help="Model ID to register."),
    source: str = typer.Option(
        "huggingface", "--source", help="Model source (huggingface, gguf, local)."
    ),
    file_path: str | None = typer.Option(None, "--file-path", help="File path for GGUF models."),
    vram: int | None = typer.Option(None, "--vram", help="Base VRAM (MB) for local models."),
    params: float | None = typer.Option(
        None, "--params", help="Parameter count (billions) for local models."
    ),
    architecture: str | None = typer.Option(
        None, "--architecture", help="Architecture name for local models."
    ),
    quant: str | None = typer.Option(
        None, "--quant", help="GGUF quantization to select (e.g. Q4_K_M, Q4_K_XL)."
    ),
) -> None:
    """Register a new ML model in the registry."""
    from gpumod.cli import cli_context, error_handler, run_async
    from gpumod.models import ModelSource

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                model_source = ModelSource(source)
                kwargs = _build_register_kwargs(
                    file_path=file_path,
                    vram=vram,
                    params=params,
                    architecture=architecture,
                    quant=quant,
                )
                registered: ModelInfo = await ctx.model_registry.register(
                    model_id, model_source, **kwargs
                )
                _print_registered(registered)

    run_async(_cmd())


@model_app.command("remove")
def remove_model(
    model_id: str = typer.Argument(help="Model ID to remove."),
) -> None:
    """Remove a model from the registry."""
    from gpumod.cli import cli_context, error_handler, run_async

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                await ctx.model_registry.remove(model_id)
                _console.print(
                    f"[yellow]Removed model [bold]{model_id}[/bold] from registry.[/yellow]"
                )

    run_async(_cmd())
