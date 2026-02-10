"""CLI commands for model discovery.

Provides the `gpumod discover` command for automated model discovery,
GGUF selection, and preset generation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, IntPrompt
from rich.table import Table

from gpumod.discovery.gguf_metadata import GGUFFile, GGUFMetadataFetcher
from gpumod.discovery.preset_generator import PresetGenerator, PresetRequest
from gpumod.discovery.system_info import (
    NvidiaSmiUnavailableError,
    SystemInfo,
    SystemInfoCollector,
)
from gpumod.discovery.unsloth_lister import (
    HuggingFaceAPIError,
    UnslothModel,
    UnslothModelLister,
)

logger = logging.getLogger(__name__)


def _format_vram(vram_mb: int) -> str:
    """Format VRAM with appropriate units (MB or GB).

    Args:
        vram_mb: VRAM in megabytes.

    Returns:
        Formatted string like "1.5 GB" or "512 MB".
    """
    if vram_mb >= 1024:
        return f"{vram_mb / 1024:.1f} GB"
    return f"{vram_mb} MB"


_DISCOVER_HELP = """Discover and configure models from HuggingFace.

[bold]Examples:[/bold]
  gpumod discover                              # List unsloth models
  gpumod discover --search deepseek            # Search by name
  gpumod discover -s kimi -a moonshotai        # Search in specific org
  gpumod discover --author bartowski -s llama  # Combine author + search
  gpumod discover --task code                  # Filter by task type
"""

discover_app = typer.Typer(
    name="discover",
    help=_DISCOVER_HELP,
    no_args_is_help=False,
    rich_markup_mode="rich",
)


@discover_app.callback(invoke_without_command=True)
def discover(  # noqa: C901, PLR0913, PLR0915
    ctx: typer.Context,
    search: str = typer.Option(
        None,
        "--search",
        "-s",
        help="Search models by name (e.g., 'deepseek', 'kimi', 'qwen')",
    ),
    author: str = typer.Option(
        "unsloth",
        "--author",
        "-a",
        help="HuggingFace organization (default: unsloth)",
    ),
    task: str = typer.Option(
        None,
        "--task",
        help="Filter by task: code, chat, embed, reasoning",
    ),
    vram: int = typer.Option(
        None,
        "--vram",
        help="VRAM budget in MB (default: detected available)",
    ),
    context: int = typer.Option(
        8192,
        "--context",
        help="Context size (default: 8192)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview without writing files",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Output JSON, no interaction",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Bypass HuggingFace cache",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Debug output",
    ),
) -> None:
    """Discover GGUF models from HuggingFace.

    Searches for models by name, filters by VRAM, and generates
    ready-to-use presets for gpumod.

    Examples:
        gpumod discover --search deepseek
        gpumod discover --search kimi --author moonshotai
        gpumod discover --task code --context 8192
        gpumod discover --author bartowski --search llama
    """
    if ctx.invoked_subcommand is not None:
        return

    # Use quiet console for JSON mode (suppresses output)
    console = Console(quiet=as_json)

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    async def _discover() -> None:  # noqa: C901, PLR0912, PLR0915
        # Step 1: Collect system info
        console.print("\n[bold]System Info[/bold]")
        console.print("─" * 40)

        try:
            collector = SystemInfoCollector()
            if not as_json:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task("Detecting GPU...", total=None)
                    system_info = await collector.get_system_info()
            else:
                system_info = await collector.get_system_info()
        except NvidiaSmiUnavailableError as exc:
            if as_json:
                print(json.dumps({"error": str(exc)}))
            else:
                console.print(f"[bold red]Error:[/bold red] {exc}")
                console.print("nvidia-smi is required for GPU detection.")
            raise typer.Exit(1) from exc

        # Display system info
        _print_system_info(console, system_info)

        # Determine VRAM budget
        vram_budget = vram if vram is not None else system_info.gpu_available_mb
        if vram is not None and vram > system_info.gpu_total_mb:
            console.print(
                f"[yellow]Warning:[/yellow] --vram {vram} exceeds GPU capacity "
                f"({system_info.gpu_total_mb} MB)"
            )

        console.print(f"\n[bold]VRAM Budget:[/bold] {vram_budget} MB")

        # Step 2: Fetch models from HuggingFace
        console.print("\n[bold]Discovering Models[/bold]")
        console.print("─" * 40)

        # Build search description
        search_desc = []
        if search:
            search_desc.append(f"search='{search}'")
        if author:
            search_desc.append(f"author={author}")
        search_info = ", ".join(search_desc) if search_desc else "all GGUF models"

        try:
            lister = UnslothModelLister(author=author or None)
            if not as_json:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task(f"Fetching models ({search_info})...", total=None)
                    models = await lister.list_models(
                        task=task, search=search, force_refresh=no_cache
                    )
            else:
                models = await lister.list_models(task=task, search=search, force_refresh=no_cache)
        except HuggingFaceAPIError as exc:
            if as_json:
                print(json.dumps({"error": str(exc)}))
            else:
                console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(1) from exc

        if not models:
            if as_json:
                print(json.dumps([]))
            else:
                console.print("[yellow]No models found matching criteria.[/yellow]")
                if search:
                    console.print("Try a different --search term or --author")
            raise typer.Exit(0)

        source = author or "HuggingFace"
        console.print(f"Found {len(models)} GGUF models from {source}")

        # Step 3: For each model, get GGUF files and filter by VRAM
        console.print("\n[bold]Analyzing Quantizations[/bold]")
        console.print("─" * 40)

        fetcher = GGUFMetadataFetcher()
        compatible_models: list[tuple[UnslothModel, list[GGUFFile]]] = []

        if not as_json:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task_id = progress.add_task("Checking models...", total=len(models))

                for model in models[:20]:  # Limit to first 20 for performance
                    try:
                        gguf_files = await fetcher.list_gguf_files(model.repo_id)
                        # Filter to files that fit VRAM budget
                        fitting = [f for f in gguf_files if f.estimated_vram_mb <= vram_budget]
                        if fitting:
                            compatible_models.append((model, fitting))
                    except Exception as exc:
                        logger.debug("Failed to fetch %s: %s", model.repo_id, exc)

                    progress.advance(task_id)
        else:
            for model in models[:20]:  # Limit to first 20 for performance
                try:
                    gguf_files = await fetcher.list_gguf_files(model.repo_id)
                    # Filter to files that fit VRAM budget
                    fitting = [f for f in gguf_files if f.estimated_vram_mb <= vram_budget]
                    if fitting:
                        compatible_models.append((model, fitting))
                except Exception as exc:
                    logger.debug("Failed to fetch %s: %s", model.repo_id, exc)

        if not compatible_models:
            if as_json:
                print(json.dumps([]))
            else:
                console.print(
                    f"[yellow]No models fit within {vram_budget} MB VRAM budget.[/yellow]"
                )
                console.print("Try increasing --vram or reducing --context")
            raise typer.Exit(0)

        # JSON output mode
        if as_json:
            output = [
                {
                    "model": m.repo_id,
                    "name": m.name,
                    "files": [
                        {
                            "filename": f.filename,
                            "quant": f.quant_type,
                            "vram_mb": f.estimated_vram_mb,
                        }
                        for f in files
                    ],
                }
                for m, files in compatible_models
            ]
            print(json.dumps(output, indent=2))
            raise typer.Exit(0)

        # Step 4: Display compatible models
        console.print(f"\n[bold]Compatible Models ({len(compatible_models)})[/bold]")
        console.print("─" * 60)

        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Model", min_width=30)
        table.add_column("Quant", width=12)
        table.add_column("VRAM", justify="right", width=10)

        idx = 1
        flat_choices: list[tuple[UnslothModel, GGUFFile]] = []
        for model, files in compatible_models:
            # Show best (smallest) quantization that fits
            best = files[0]  # Already sorted by size
            table.add_row(
                str(idx),
                model.name[:40],
                best.quant_type or "unknown",
                _format_vram(best.estimated_vram_mb),
            )
            flat_choices.append((model, best))
            idx += 1

        console.print(table)

        # Step 5: Interactive selection
        console.print()
        try:
            choice = IntPrompt.ask(
                "Select model",
                default=1,
                show_default=True,
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled.[/dim]")
            raise typer.Exit(0) from None

        if choice < 1 or choice > len(flat_choices):
            console.print("[red]Invalid selection.[/red]")
            raise typer.Exit(1)

        selected_model, selected_gguf = flat_choices[choice - 1]

        # Step 6: Generate preset
        console.print("\n[bold]Generating Preset[/bold]")
        console.print("─" * 40)

        # Check if model name suggests MoE
        is_moe = any(
            kw in selected_model.name.lower() for kw in ("moe", "mixture", "expert", "a3b")
        )

        request = PresetRequest(
            repo_id=selected_model.repo_id,
            gguf_file=selected_gguf,
            system_info=system_info,
            ctx_size=context,
            is_moe=is_moe,
        )

        generator = PresetGenerator()
        yaml_str = generator.generate(request)

        # Show preview
        console.print(Panel(yaml_str, title="Preset Preview", border_style="dim"))

        if dry_run:
            console.print("[dim]--dry-run: not writing file[/dim]")
            raise typer.Exit(0)

        # Step 7: Write preset file
        # Derive filename from service ID
        import yaml

        parsed = yaml.safe_load(yaml_str)
        service_id = parsed["id"]
        preset_path = Path(f"presets/llm/{service_id}.yaml")

        if preset_path.exists() and not Confirm.ask(  # noqa: ASYNC240
            f"Overwrite existing {preset_path}?", default=False
        ):
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

        # Ensure directory exists
        preset_path.parent.mkdir(parents=True, exist_ok=True)

        preset_path.write_text(yaml_str)  # noqa: ASYNC240
        console.print(f"\n[green]✓[/green] Preset written to [bold]{preset_path}[/bold]")
        repo = selected_model.repo_id
        gguf = selected_gguf.filename
        console.print(
            f"\nNext steps:\n"
            f"  1. Download the model: huggingface-cli download {repo} {gguf}\n"
            f"  2. Move to ~/models/\n"
            f"  3. Add service: gpumod service add --preset {service_id}\n"
        )

    from gpumod.cli import error_handler, run_async

    with error_handler(console=console):
        run_async(_discover())


def _print_system_info(console: Console, info: SystemInfo) -> None:
    """Print system info in a formatted way."""
    console.print(f"[bold]GPU:[/bold] {info.gpu_name}")
    console.print(
        f"[bold]VRAM:[/bold] {info.gpu_used_mb} / {info.gpu_total_mb} MB "
        f"([green]{info.gpu_available_mb} MB available[/green])"
    )
    console.print(f"[bold]RAM:[/bold] {info.ram_available_mb} / {info.ram_total_mb} MB")
    if info.swap_available_mb > 0:
        console.print(f"[bold]Swap:[/bold] {info.swap_available_mb} MB available")
    if info.current_mode:
        console.print(f"[bold]Mode:[/bold] {info.current_mode}")
    if info.running_services:
        console.print(f"[bold]Running:[/bold] {', '.join(info.running_services)}")
