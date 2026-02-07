"""MCP tools for gpumod -- 6 read-only + 3 mutating tools.

Provides tool functions for GPU status, service/mode/model browsing,
VRAM simulation, and service lifecycle management. All tools follow
SEC-V1 input validation and SEC-A1 audit logging requirements.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastmcp import Context  # noqa: TCH002 -- runtime import needed for FastMCP DI

from gpumod.simulation import SimulationError
from gpumod.validation import (
    sanitize_name,
    validate_context_override,
    validate_mode_id,
    validate_model_id,
    validate_service_id,
)

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from gpumod.db import Database
    from gpumod.registry import ModelRegistry
    from gpumod.services.manager import ServiceManager
    from gpumod.simulation import SimulationEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan context helpers
# ---------------------------------------------------------------------------


def _get_lifespan(ctx: Context) -> dict[str, Any]:
    """Extract the lifespan result dict from the FastMCP context."""
    lifespan: dict[str, Any] | None = getattr(ctx.fastmcp, "_lifespan_result", None)
    if lifespan is None:
        msg = "Lifespan context not available"
        raise RuntimeError(msg)
    return lifespan


def _get_manager(ctx: Context) -> ServiceManager:
    """Extract the ServiceManager from the FastMCP lifespan context."""
    result: ServiceManager = _get_lifespan(ctx)["manager"]
    return result


def _get_db(ctx: Context) -> Database:
    """Extract the Database from the FastMCP lifespan context."""
    result: Database = _get_lifespan(ctx)["db"]
    return result


def _get_model_registry(ctx: Context) -> ModelRegistry:
    """Extract the ModelRegistry from the FastMCP lifespan context."""
    result: ModelRegistry = _get_lifespan(ctx)["model_registry"]
    return result


def _get_simulation(ctx: Context) -> SimulationEngine:
    """Extract the SimulationEngine from the FastMCP lifespan context."""
    result: SimulationEngine = _get_lifespan(ctx)["simulation"]
    return result


def _validation_error(message: str) -> dict[str, str]:
    """Return a standardized validation error response."""
    return {"error": message, "code": "VALIDATION_ERROR"}


def _not_found_error(message: str) -> dict[str, str]:
    """Return a standardized not-found error response."""
    return {"error": message, "code": "NOT_FOUND"}


def _simulation_error(message: str) -> dict[str, str]:
    """Return a standardized simulation error response."""
    return {"error": message, "code": "SIMULATION_ERROR"}


def _sanitize_dict_names(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize 'name' fields in a dict, recursively handling nested structures.

    Strips ANSI escapes, Rich markup, and control characters from any
    string value associated with a 'name' key.

    Parameters
    ----------
    data:
        A dictionary that may contain 'name' keys at various nesting levels.

    Returns
    -------
    dict[str, Any]
        A new dictionary with all 'name' values sanitized.
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key == "name" and isinstance(value, str):
            result[key] = sanitize_name(value)
        elif isinstance(value, dict):
            result[key] = _sanitize_dict_names(value)
        elif isinstance(value, list):
            result[key] = [
                _sanitize_dict_names(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Read-only tools (6)
# ---------------------------------------------------------------------------


async def gpu_status(ctx: Context) -> dict[str, Any]:
    """Get current GPU status: mode, VRAM usage, running services."""
    manager = _get_manager(ctx)
    status = await manager.get_status()
    result: dict[str, Any] = status.model_dump(mode="json")
    return _sanitize_dict_names(result)


async def list_services(ctx: Context) -> dict[str, Any]:
    """List all registered services with driver type and VRAM."""
    db = _get_db(ctx)
    services = await db.list_services()
    return {"services": [_sanitize_dict_names(s.model_dump(mode="json")) for s in services]}


async def list_modes(ctx: Context) -> dict[str, Any]:
    """List all available GPU modes."""
    db = _get_db(ctx)
    modes = await db.list_modes()
    return {"modes": [_sanitize_dict_names(m.model_dump(mode="json")) for m in modes]}


async def service_info(service_id: str, ctx: Context) -> dict[str, Any]:
    """Get detailed info for a specific service."""
    try:
        validate_service_id(service_id)
    except ValueError as exc:
        return _validation_error(str(exc))

    db = _get_db(ctx)
    svc = await db.get_service(service_id)
    if svc is None:
        return _not_found_error(f"Service not found: {service_id!r}")
    result: dict[str, Any] = svc.model_dump(mode="json")
    return _sanitize_dict_names(result)


async def model_info(model_id: str, ctx: Context) -> dict[str, Any]:
    """Get model metadata and VRAM estimates."""
    try:
        validate_model_id(model_id)
    except ValueError as exc:
        return _validation_error(str(exc))

    model_registry = _get_model_registry(ctx)
    model = await model_registry.get(model_id)
    if model is None:
        return _not_found_error(f"Model not found: {model_id!r}")
    result: dict[str, Any] = model.model_dump(mode="json")
    return _sanitize_dict_names(result)


async def simulate_mode(
    mode_id: str,
    ctx: Context,
    add_services: list[str] | None = None,
    remove_services: list[str] | None = None,
    context_overrides: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Simulate VRAM for a mode with optional changes. Non-destructive."""
    try:
        validate_mode_id(mode_id)
    except ValueError as exc:
        return _validation_error(str(exc))

    # Validate add_services IDs
    if add_services:
        for sid in add_services:
            try:
                validate_service_id(sid)
            except ValueError as exc:
                return _validation_error(str(exc))

    # Validate remove_services IDs
    if remove_services:
        for sid in remove_services:
            try:
                validate_service_id(sid)
            except ValueError as exc:
                return _validation_error(str(exc))

    # Validate context overrides
    if context_overrides:
        for key, val in context_overrides.items():
            try:
                validate_context_override(key, val)
            except ValueError as exc:
                return _validation_error(str(exc))

    simulation = _get_simulation(ctx)
    try:
        sim_result = await simulation.simulate_mode(
            mode_id,
            add=add_services,
            remove=remove_services,
            context_overrides=context_overrides,
        )
    except ValueError as exc:
        return _not_found_error(str(exc))
    except SimulationError as exc:
        return _simulation_error(str(exc))

    result: dict[str, Any] = sim_result.model_dump(mode="json")
    return _sanitize_dict_names(result)


# ---------------------------------------------------------------------------
# Mutating tools (3)
# ---------------------------------------------------------------------------


async def switch_mode(mode_id: str, ctx: Context) -> dict[str, Any]:
    """Switch to a different GPU mode. Starts/stops services. [MUTATING]"""
    try:
        validate_mode_id(mode_id)
    except ValueError as exc:
        return _validation_error(str(exc))

    logger.info("MCP tool switch_mode called: mode_id=%s", mode_id)

    manager = _get_manager(ctx)
    try:
        mode_result = await manager.switch_mode(mode_id)
    except ValueError as exc:
        return _not_found_error(str(exc))

    result: dict[str, Any] = mode_result.model_dump(mode="json")
    return _sanitize_dict_names(result)


async def start_service(service_id: str, ctx: Context) -> dict[str, Any]:
    """Start a specific service. [MUTATING]"""
    try:
        validate_service_id(service_id)
    except ValueError as exc:
        return _validation_error(str(exc))

    logger.info("MCP tool start_service called: service_id=%s", service_id)

    manager = _get_manager(ctx)
    try:
        await manager.start_service(service_id)
    except ValueError as exc:
        return _not_found_error(str(exc))

    return {"success": True, "service_id": service_id, "action": "started"}


async def stop_service(service_id: str, ctx: Context) -> dict[str, Any]:
    """Stop a specific service. [MUTATING]"""
    try:
        validate_service_id(service_id)
    except ValueError as exc:
        return _validation_error(str(exc))

    logger.info("MCP tool stop_service called: service_id=%s", service_id)

    manager = _get_manager(ctx)
    try:
        await manager.stop_service(service_id)
    except ValueError as exc:
        return _not_found_error(str(exc))

    return {"success": True, "service_id": service_id, "action": "stopped"}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_tools(server: FastMCP[Any]) -> None:
    """Register all gpumod MCP tools on the given FastMCP server.

    Parameters
    ----------
    server:
        The FastMCP server instance to register tools on.
    """
    server.tool(
        name="gpu_status",
        description="Get current GPU status: mode, VRAM usage, running services.",
    )(gpu_status)

    server.tool(
        name="list_services",
        description="List all registered services with driver type and VRAM.",
    )(list_services)

    server.tool(
        name="list_modes",
        description="List all available GPU modes.",
    )(list_modes)

    server.tool(
        name="service_info",
        description="Get detailed info for a specific service.",
    )(service_info)

    server.tool(
        name="model_info",
        description="Get model metadata and VRAM estimates.",
    )(model_info)

    server.tool(
        name="simulate_mode",
        description="Simulate VRAM for a mode with optional changes. Non-destructive.",
    )(simulate_mode)

    server.tool(
        name="switch_mode",
        description="Switch to a different GPU mode. Starts/stops services. [MUTATING]",
    )(switch_mode)

    server.tool(
        name="start_service",
        description="Start a specific service. [MUTATING]",
    )(start_service)

    server.tool(
        name="stop_service",
        description="Stop a specific service. [MUTATING]",
    )(stop_service)
