"""Template CLI commands -- gpumod template list|show|generate|install."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

template_app = typer.Typer(name="template", help="Manage service templates.")

_console = Console()

# Default systemd unit install directory.
_SYSTEMD_UNIT_DIR = Path("/etc/systemd/system")

# Pattern for safe unit names: alphanumeric, hyphens, underscores only.
_SAFE_UNIT_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_install_path(target: Path, base_dir: Path) -> None:
    """Validate that the install path is safe (no path traversal).

    Args:
        target: The resolved target file path.
        base_dir: The directory files must reside in.

    Raises:
        ValueError: If the path is outside the allowed directory.
    """
    resolved_target = target.resolve()
    resolved_base = base_dir.resolve()
    is_under_base = str(resolved_target).startswith(str(resolved_base) + "/")
    is_direct_child = resolved_target.parent == resolved_base
    if not is_under_base and not is_direct_child:
        msg = f"Unsafe install path: {target} is outside {base_dir}"
        raise ValueError(msg)


def _get_unit_name(service: Any) -> str:
    """Derive the unit file name for a service.

    Uses service.unit_name if set, otherwise defaults to
    ``gpumod-{service.id}``.

    Args:
        service: The service object.

    Returns:
        The unit name (without .service suffix).

    Raises:
        ValueError: If the derived unit name contains unsafe characters.
    """
    unit_name: str = service.unit_name if service.unit_name else f"gpumod-{service.id}"
    if not _SAFE_UNIT_NAME_RE.match(unit_name):
        msg = f"Unsafe unit name: {unit_name!r}"
        raise ValueError(msg)
    return unit_name


async def _build_settings(ctx: Any) -> dict[str, str]:  # noqa: ANN401
    """Build a settings dict from DB settings for template rendering."""
    settings: dict[str, str] = {}
    for key in ("user", "cuda_devices", "working_dir"):
        val: str | None = await ctx.db.get_setting(key)
        if val is not None:
            settings[key] = val
    return settings


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@template_app.command("list")
def list_templates(
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List available Jinja2 service templates."""
    from gpumod.cli import cli_context, error_handler, json_output, run_async

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                templates = ctx.template_engine.available_templates()

                if not templates:
                    if as_json:
                        json_output([], as_json=True)
                    else:
                        _console.print("[dim]No templates available.[/dim]")
                    return

                if json_output(templates, as_json=as_json) is None:
                    return

                table = Table(title="Available Templates")
                table.add_column("Template Name", style="cyan")

                for name in templates:
                    table.add_row(name)

                _console.print(table)

    run_async(_cmd())


@template_app.command("show")
def show_template(
    template_name: str = typer.Argument(help="Name of the template file to render."),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show a rendered template with sample context."""
    from gpumod.cli import cli_context, error_handler, json_output, run_async

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                rendered = ctx.template_engine.render(template_name, {})

                if as_json:
                    json_output(
                        {"template_name": template_name, "rendered": rendered},
                        as_json=True,
                    )
                    return

                _console.print(
                    Panel(
                        Syntax(rendered, "ini", theme="monokai"),
                        title=f"Template: {template_name}",
                        border_style="blue",
                    )
                )

    run_async(_cmd())


@template_app.command("generate")
def generate_template(
    service_id: str = typer.Argument(help="Service ID to generate a unit file for."),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
    output: str | None = typer.Option(None, "--output", "-o", help="Write rendered unit to file."),
) -> None:
    """Generate a systemd unit file for a registered service."""
    from gpumod.cli import cli_context, error_handler, json_output, run_async

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                service = await ctx.db.get_service(service_id)
                if service is None:
                    msg = f"Service not found: {service_id!r}"
                    raise KeyError(msg)

                settings = await _build_settings(ctx)
                unit_vars = service.extra_config.get("unit_vars")
                rendered = ctx.template_engine.render_service_unit(
                    service, settings, unit_vars=unit_vars
                )

                if output is not None:
                    out_path = Path(output)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(rendered)
                    _console.print(f"[green]Wrote unit file to [bold]{out_path}[/bold][/green]")
                    return

                if as_json:
                    json_output(
                        {"service_id": service_id, "rendered": rendered},
                        as_json=True,
                    )
                    return

                _console.print(
                    Panel(
                        Syntax(rendered, "ini", theme="monokai"),
                        title=f"Unit file for service: {service_id}",
                        border_style="blue",
                    )
                )

    run_async(_cmd())


@template_app.command("install")
def install_template(
    service_id: str = typer.Argument(help="Service ID to install a unit file for."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Install a systemd unit file for a registered service.

    Renders the unit and writes it to the systemd directory.
    Requires --yes for confirmation unless previewing.
    """
    from gpumod.cli import cli_context, error_handler, run_async

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                service = await ctx.db.get_service(service_id)
                if service is None:
                    msg = f"Service not found: {service_id!r}"
                    raise KeyError(msg)

                settings = await _build_settings(ctx)
                unit_vars = service.extra_config.get("unit_vars")
                rendered = ctx.template_engine.render_service_unit(
                    service, settings, unit_vars=unit_vars
                )

                unit_name = _get_unit_name(service)
                target_path = _SYSTEMD_UNIT_DIR / f"{unit_name}.service"

                if not yes:
                    _console.print(
                        Panel(
                            Syntax(rendered, "ini", theme="monokai"),
                            title=f"Preview: {target_path}",
                            border_style="yellow",
                        )
                    )
                    _console.print("[yellow]Pass --yes to confirm installation.[/yellow]")
                    return

                # Validate target path is safe
                _validate_install_path(target_path, _SYSTEMD_UNIT_DIR)

                # Write the unit file
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(rendered)
                _console.print(f"[green]Installed unit file: [bold]{target_path}[/bold][/green]")

    run_async(_cmd())
