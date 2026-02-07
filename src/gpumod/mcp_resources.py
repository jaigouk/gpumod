"""MCP resources for browsing gpumod modes, services, and models.

Provides both static resources (help, config) and dynamic resource templates
for detailed views. All output is Markdown-formatted with no internal paths
exposed (SEC-E2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp import Context  # noqa: TCH002 — runtime import needed for FastMCP DI

from gpumod.validation import validate_mode_id, validate_model_id, validate_service_id

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from gpumod.db import Database
    from gpumod.models import Mode, ModelInfo, Service

# ---------------------------------------------------------------------------
# Help text (static)
# ---------------------------------------------------------------------------

HELP_TEXT = """\
# gpumod - GPU Service Manager

gpumod manages ML workloads (vLLM, llama.cpp, FastAPI) on NVIDIA GPUs.

## Resources

| URI | Description |
|-----|-------------|
| `gpumod://help` | This help text |
| `gpumod://config` | Current configuration and settings |
| `gpumod://modes` | List all defined modes |
| `gpumod://modes/{mode_id}` | Detail view of a specific mode |
| `gpumod://services` | List all registered services |
| `gpumod://services/{service_id}` | Detail view of a specific service |
| `gpumod://models` | List all registered models |
| `gpumod://models/{model_id}` | Detail view of a specific model |

## Tools

gpumod exposes tools for managing GPU services, switching modes,
simulating VRAM usage, and more. Use the MCP tool listing to see
all available tools and their parameters.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_db(ctx: Context) -> Database:
    """Extract the Database instance from the FastMCP lifespan context."""
    lifespan_result: Any = getattr(ctx.fastmcp, "_lifespan_result", None)
    if lifespan_result is None:
        msg = "Lifespan context not available"
        raise RuntimeError(msg)
    db: Database = lifespan_result["db"]
    return db


def _format_modes_table(modes: list[Mode]) -> str:
    """Format a list of modes as a Markdown table."""
    lines = [
        "# Modes",
        "",
        "| ID | Name | Description | VRAM (MB) |",
        "|----|------|-------------|-----------|",
    ]
    for mode in modes:
        desc = mode.description or "-"
        vram = str(mode.total_vram_mb) if mode.total_vram_mb is not None else "-"
        lines.append(f"| {mode.id} | {mode.name} | {desc} | {vram} |")
    return "\n".join(lines)


def _format_services_table(services: list[Service]) -> str:
    """Format a list of services as a Markdown table."""
    lines = [
        "# Services",
        "",
        "| ID | Name | Driver | Port | VRAM (MB) |",
        "|----|------|--------|------|-----------|",
    ]
    for svc in services:
        port = str(svc.port) if svc.port is not None else "-"
        lines.append(f"| {svc.id} | {svc.name} | {svc.driver.value} | {port} | {svc.vram_mb} |")
    return "\n".join(lines)


def _format_models_table(models: list[ModelInfo]) -> str:
    """Format a list of models as a Markdown table."""
    lines = [
        "# Models",
        "",
        "| ID | Source | Parameters (B) | Architecture | Base VRAM (MB) |",
        "|----|--------|----------------|--------------|----------------|",
    ]
    for model in models:
        params = f"{model.parameters_b:.1f}" if model.parameters_b is not None else "-"
        arch = model.architecture or "-"
        vram = str(model.base_vram_mb) if model.base_vram_mb is not None else "-"
        lines.append(f"| {model.id} | {model.source.value} | {params} | {arch} | {vram} |")
    return "\n".join(lines)


def _format_mode_detail(mode: Mode, services: list[Service]) -> str:
    """Format a detailed view of a single mode."""
    lines = [
        f"# Mode: {mode.name}",
        "",
        f"- **ID:** {mode.id}",
        f"- **Description:** {mode.description or '-'}",
        f"- **Total VRAM:** {mode.total_vram_mb or '-'} MB",
        "",
    ]
    if services:
        lines.append("## Services")
        lines.append("")
        lines.append("| ID | Name | Driver | VRAM (MB) |")
        lines.append("|----|------|--------|-----------|")
        for svc in services:
            lines.append(f"| {svc.id} | {svc.name} | {svc.driver.value} | {svc.vram_mb} |")
    else:
        lines.append("*No services assigned to this mode.*")
    return "\n".join(lines)


def _format_service_detail(service: Service) -> str:
    """Format a detailed view of a single service."""
    lines = [
        f"# Service: {service.name}",
        "",
        f"- **ID:** {service.id}",
        f"- **Driver:** {service.driver.value}",
        f"- **Port:** {service.port or '-'}",
        f"- **VRAM:** {service.vram_mb} MB",
        f"- **Sleep Mode:** {service.sleep_mode.value}",
        f"- **Health Endpoint:** {service.health_endpoint}",
        f"- **Model ID:** {service.model_id or '-'}",
        f"- **Startup Timeout:** {service.startup_timeout}s",
    ]
    if service.depends_on:
        lines.append(f"- **Dependencies:** {', '.join(service.depends_on)}")
    return "\n".join(lines)


def _format_model_detail(model: ModelInfo) -> str:
    """Format a detailed view of a single model."""
    params = f"{model.parameters_b:.1f}B" if model.parameters_b is not None else "-"
    lines = [
        f"# Model: {model.id}",
        "",
        f"- **Source:** {model.source.value}",
        f"- **Parameters:** {params}",
        f"- **Architecture:** {model.architecture or '-'}",
        f"- **Base VRAM:** {model.base_vram_mb or '-'} MB",
        f"- **KV Cache per 1k tokens:** {model.kv_cache_per_1k_tokens_mb or '-'} MB",
    ]
    if model.quantizations:
        lines.append(f"- **Quantizations:** {', '.join(model.quantizations)}")
    if model.capabilities:
        lines.append(f"- **Capabilities:** {', '.join(model.capabilities)}")
    if model.notes:
        lines.append(f"- **Notes:** {model.notes}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resource functions (called directly in tests, registered on server via decorator)
# ---------------------------------------------------------------------------


def help_resource() -> str:
    """Overview of gpumod capabilities."""
    return HELP_TEXT


async def config_resource(*, ctx: Context) -> str:
    """Current gpumod configuration."""
    db = _get_db(ctx)
    current_mode = await db.get_setting("current_mode")
    lines = [
        "# Configuration",
        "",
        f"- **Current Mode:** {current_mode or 'none'}",
    ]
    return "\n".join(lines)


async def modes_resource(*, ctx: Context) -> str:
    """List all modes as Markdown table."""
    db = _get_db(ctx)
    modes = await db.list_modes()
    if not modes:
        return "# Modes\n\nNo modes configured."
    return _format_modes_table(modes)


async def mode_detail_resource(mode_id: str, *, ctx: Context) -> str:
    """Detail view of a specific mode with its services."""
    try:
        validate_mode_id(mode_id)
    except ValueError:
        return "Mode not found: the requested mode was not found."
    db = _get_db(ctx)
    mode = await db.get_mode(mode_id)
    if mode is None:
        return f"Mode not found: '{mode_id}' does not exist."
    services = await db.get_mode_services(mode_id)
    return _format_mode_detail(mode, services)


async def services_resource(*, ctx: Context) -> str:
    """List all services as Markdown table."""
    db = _get_db(ctx)
    services = await db.list_services()
    if not services:
        return "# Services\n\nNo services configured."
    return _format_services_table(services)


async def service_detail_resource(service_id: str, *, ctx: Context) -> str:
    """Detail view of a specific service."""
    try:
        validate_service_id(service_id)
    except ValueError:
        return "Service not found: the requested service was not found."
    db = _get_db(ctx)
    service = await db.get_service(service_id)
    if service is None:
        return f"Service not found: '{service_id}' does not exist."
    return _format_service_detail(service)


async def models_resource(*, ctx: Context) -> str:
    """List all registered models as Markdown table."""
    db = _get_db(ctx)
    models = await db.list_models()
    if not models:
        return "# Models\n\nNo models registered."
    return _format_models_table(models)


async def model_detail_resource(model_id: str, *, ctx: Context) -> str:
    """Detail view of a specific model with VRAM estimates."""
    try:
        validate_model_id(model_id)
    except ValueError:
        return "Model not found: the requested model was not found."
    db = _get_db(ctx)
    model = await db.get_model(model_id)
    if model is None:
        return f"Model not found: '{model_id}' does not exist."
    return _format_model_detail(model)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_resources(server: FastMCP) -> None:
    """Register all gpumod resources on the given FastMCP server.

    Parameters
    ----------
    server:
        The FastMCP server instance to register resources on.
    """
    server.resource("gpumod://help", name="help", description="Overview of gpumod capabilities")(
        help_resource
    )
    server.resource("gpumod://config", name="config", description="Current gpumod configuration")(
        config_resource
    )
    server.resource("gpumod://modes", name="modes", description="List all modes")(modes_resource)
    server.resource(
        "gpumod://modes/{mode_id}",
        name="mode_detail",
        description="Detail view of a specific mode",
    )(mode_detail_resource)
    server.resource("gpumod://services", name="services", description="List all services")(
        services_resource
    )
    server.resource(
        "gpumod://services/{service_id}",
        name="service_detail",
        description="Detail view of a specific service",
    )(service_detail_resource)
    server.resource("gpumod://models", name="models", description="List all models")(
        models_resource
    )
    server.resource(
        "gpumod://models/{model_id}",
        name="model_detail",
        description="Detail view of a specific model",
    )(model_detail_resource)
