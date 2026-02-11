"""MCP tools for gpumod -- 6 read-only + 3 mutating + 6 discovery + 1 consult tools.

Provides tool functions for GPU status, service/mode/model browsing,
VRAM simulation, and service lifecycle management. All tools follow
SEC-V1 input validation and SEC-A1 audit logging requirements.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import re
from typing import TYPE_CHECKING, Any

from fastmcp import Context  # noqa: TC002 -- runtime import needed for FastMCP DI

from gpumod.discovery.config_fetcher import ConfigFetcher, ConfigNotFoundError
from gpumod.discovery.docs_fetcher import DocsNotFoundError, DriverDocsFetcher
from gpumod.discovery.gguf_metadata import GGUFMetadataFetcher, RepoNotFoundError
from gpumod.discovery.hf_searcher import HuggingFaceSearcher
from gpumod.rlm.orchestrator import RLMOrchestrator
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


def _validate_service_ids(ids: list[str] | None) -> dict[str, Any] | None:
    """Validate a list of service IDs. Returns error dict or None."""
    if not ids:
        return None
    for sid in ids:
        try:
            validate_service_id(sid)
        except ValueError as exc:
            return _validation_error(str(exc))
    return None


def _validate_overrides(overrides: dict[str, int] | None) -> dict[str, Any] | None:
    """Validate context overrides. Returns error dict or None."""
    if not overrides:
        return None
    for key, val in overrides.items():
        try:
            validate_context_override(key, val)
        except ValueError as exc:
            return _validation_error(str(exc))
    return None


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

    for error in (
        _validate_service_ids(add_services),
        _validate_service_ids(remove_services),
        _validate_overrides(context_overrides),
    ):
        if error is not None:
            return error

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
# Discovery tools (3 read-only)
# ---------------------------------------------------------------------------

# Valid task types for search filtering
_VALID_TASKS = frozenset({"code", "chat", "embed", "reasoning"})

# Valid driver types for search filtering
_VALID_DRIVERS = frozenset({"llamacpp", "vllm", "any"})

# Limits for validation
_MAX_SEARCH_LIMIT = 100
_MIN_CONTEXT_SIZE = 512
_MAX_CONTEXT_SIZE = 262144  # 256k
_MAX_REPO_ID_LENGTH = 200  # HuggingFace limit is 96 per part, we allow some margin

# Dangerous patterns that should never appear in repo_id (SEC-V1)
_DANGEROUS_PATTERNS = (
    ";",  # Shell command separator
    "|",  # Pipe
    "&",  # Background/AND
    "$",  # Variable expansion
    "`",  # Command substitution
    ">",  # Redirect
    "<",  # Redirect
    "{{",  # Jinja2 template
    "{%",  # Jinja2 template
    "..",  # Path traversal
    "\x00",  # Null byte
    "'",  # SQL injection
    '"',  # SQL injection
    "--",  # SQL comment
    "/*",  # SQL comment
)


def _validate_repo_id(repo_id: str) -> str | None:
    """Validate repo_id format (org/name). Returns error message or None.

    Implements SEC-V1 input validation for HuggingFace repo IDs.
    Rejects shell injection, SQL injection, path traversal, and template injection.
    """
    if not repo_id or not repo_id.strip():
        return "repo_id is required"

    # Length check to prevent DoS
    if len(repo_id) > _MAX_REPO_ID_LENGTH:
        return f"repo_id exceeds maximum length of {_MAX_REPO_ID_LENGTH}"

    # Check for dangerous patterns (SEC-V1: T1-T4)
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in repo_id:
            return "repo_id contains invalid characters"

    if "/" not in repo_id or repo_id.count("/") != 1:
        return "repo_id must be in format 'org/name' (e.g., 'unsloth/Qwen-GGUF')"

    org, name = repo_id.split("/")
    if not org or not name:
        return "repo_id must have non-empty org and name"

    return None


async def search_hf_models(  # noqa: PLR0911
    ctx: Context,
    author: str | None = None,
    search: str | None = None,
    task: str | None = None,
    driver: str | None = None,
    limit: int = 20,
    no_cache: bool = False,
) -> dict[str, Any]:
    """Search HuggingFace for models by author, keyword, task, or driver.

    Args:
        author: HuggingFace organization (default: searches all).
        search: Keyword to search in model names.
        task: Filter by task type (code, chat, embed, reasoning).
        driver: Filter by driver type (llamacpp for GGUF, vllm for Safetensors, any).
        limit: Maximum results to return (1-100, default 20).
        no_cache: Bypass cache for fresh results.

    Returns:
        Dict with 'models' list containing repo_id, name, description, tags,
        and optionally model_format and driver_hint when driver param used.
    """
    from gpumod.discovery.unsloth_lister import HuggingFaceAPIError, UnslothModelLister

    # Validate limit
    if limit < 1:
        return _validation_error("limit must be at least 1")
    if limit > _MAX_SEARCH_LIMIT:
        return _validation_error(f"limit cannot exceed {_MAX_SEARCH_LIMIT}")

    # Validate task
    if task is not None and task not in _VALID_TASKS:
        return _validation_error(f"task must be one of: {', '.join(sorted(_VALID_TASKS))}")

    # Validate driver
    if driver is not None and driver not in _VALID_DRIVERS:
        return _validation_error(f"driver must be one of: {', '.join(sorted(_VALID_DRIVERS))}")

    try:
        # Use HuggingFaceSearcher when driver param is specified (new unified search)
        if driver is not None:
            searcher = HuggingFaceSearcher(cache_ttl_seconds=3600)
            results = await searcher.search(
                query=search or "",
                author=author,
                driver=driver,
                limit=limit,
                force_refresh=no_cache,
            )

            # Convert SearchResult to serializable dicts with sanitization (SEC-E3)
            result_models: list[dict[str, Any]] = [
                {
                    "repo_id": r.repo_id,
                    "name": sanitize_name(r.name),
                    "description": sanitize_name(r.description) if r.description else None,
                    "tags": list(r.tags),
                    "model_format": r.model_format,
                    "driver_hint": r.driver_hint,
                }
                for r in results
            ]

            return {"models": result_models, "count": len(result_models)}

        # Legacy path: use UnslothModelLister for backward compatibility
        lister = UnslothModelLister(author=author, cache_ttl_seconds=3600)
        models = await lister.list_models(
            task=task,
            search=search,
            force_refresh=no_cache,
        )

        # Apply limit
        models = models[:limit]

        # Convert to serializable dicts with sanitization (SEC-E3)
        result_models = [
            {
                "repo_id": m.repo_id,
                "name": sanitize_name(m.name),
                "description": sanitize_name(m.description) if m.description else None,
                "tags": list(m.tags),
                "has_gguf": m.has_gguf,
            }
            for m in models
        ]

        return {"models": result_models, "count": len(result_models)}

    except HuggingFaceAPIError as exc:
        return {"error": str(exc), "code": "API_ERROR"}


async def list_gguf_files(
    repo_id: str,
    ctx: Context,
    vram_budget_mb: int | None = None,
) -> dict[str, Any]:
    """List GGUF files in a HuggingFace repo with size and VRAM estimates.

    Args:
        repo_id: HuggingFace repo ID (e.g., 'unsloth/Qwen3-Coder-Next-GGUF').
        vram_budget_mb: Optional VRAM budget to filter files that fit.

    Returns:
        Dict with 'files' list containing filename, size, quant_type, vram estimate.
    """
    # Validate repo_id
    error = _validate_repo_id(repo_id)
    if error:
        return _validation_error(error)

    # Validate vram_budget_mb
    if vram_budget_mb is not None and vram_budget_mb <= 0:
        return _validation_error("vram_budget_mb must be positive")

    try:
        fetcher = GGUFMetadataFetcher()
        files = await fetcher.list_gguf_files(repo_id)

        # Filter by VRAM budget if specified
        if vram_budget_mb is not None:
            files = [f for f in files if f.estimated_vram_mb <= vram_budget_mb]

        # Convert to serializable dicts
        result_files = [
            {
                "filename": f.filename,
                "size_bytes": f.size_bytes,
                "quant_type": f.quant_type,
                "estimated_vram_mb": f.estimated_vram_mb,
                "is_split": f.is_split,
                "split_parts": f.split_parts,
            }
            for f in files
        ]

        return {
            "repo_id": repo_id,
            "files": result_files,
            "count": len(result_files),
        }

    except RepoNotFoundError:
        return _not_found_error(f"Repository not found: {repo_id}")


async def list_model_files(
    repo_id: str,
    ctx: Context,
    vram_budget_mb: int | None = None,
) -> dict[str, Any]:
    """List model files in a HuggingFace repo with format detection and VRAM estimates.

    Unified tool that supports both GGUF and Safetensors model formats.

    Args:
        repo_id: HuggingFace repo ID (e.g., 'unsloth/Qwen3-Coder-Next-GGUF').
        vram_budget_mb: Optional VRAM budget to filter files that fit.

    Returns:
        Dict with 'files' list, 'model_format', and 'driver_hint'.
    """
    # Validate repo_id
    error = _validate_repo_id(repo_id)
    if error:
        return _validation_error(error)

    # Validate vram_budget_mb
    if vram_budget_mb is not None and vram_budget_mb <= 0:
        return _validation_error("vram_budget_mb must be positive")

    # Try GGUF format first (most common for quantized models)
    try:
        fetcher = GGUFMetadataFetcher()
        files = await fetcher.list_gguf_files(repo_id)

        if files:
            # Filter by VRAM budget if specified
            if vram_budget_mb is not None:
                files = [f for f in files if f.estimated_vram_mb <= vram_budget_mb]

            # Convert to serializable dicts
            result_files = [
                {
                    "filename": f.filename,
                    "size_bytes": f.size_bytes,
                    "quant_type": f.quant_type,
                    "estimated_vram_mb": f.estimated_vram_mb,
                    "is_split": f.is_split,
                    "split_parts": f.split_parts,
                }
                for f in files
            ]

            return {
                "repo_id": repo_id,
                "files": result_files,
                "count": len(result_files),
                "model_format": "gguf",
                "driver_hint": "llamacpp",
            }

    except RepoNotFoundError:
        return _not_found_error(f"Repository not found: {repo_id}")

    # No GGUF files found - return empty with unknown format
    return {
        "repo_id": repo_id,
        "files": [],
        "count": 0,
        "model_format": "unknown",
        "driver_hint": None,
    }


async def fetch_model_config(
    repo_id: str,
    ctx: Context,
) -> dict[str, Any]:
    """Fetch config.json from a HuggingFace repo for model architecture detection.

    Args:
        repo_id: HuggingFace repo ID (e.g., 'meta-llama/Llama-3.1-8B').

    Returns:
        Dict with model architecture info: architectures, context_length, is_moe, etc.
    """
    # Validate repo_id
    error = _validate_repo_id(repo_id)
    if error:
        return _validation_error(error)

    try:
        fetcher = ConfigFetcher()
        config = await fetcher.fetch(repo_id)

        return {
            "repo_id": config.repo_id,
            "architectures": config.architectures,
            "total_params": config.total_params,
            "is_moe": config.is_moe,
            "num_experts": config.num_experts,
            "context_length": config.context_length,
            "vocab_size": config.vocab_size,
        }

    except ConfigNotFoundError:
        return _not_found_error(f"config.json not found for {repo_id}")
    except RepoNotFoundError:
        return _not_found_error(f"Repository not found: {repo_id}")


async def generate_preset(  # noqa: PLR0911, C901
    repo_id: str,
    gguf_file: str,
    ctx: Context,
    context_size: int = 8192,
    service_id: str | None = None,
) -> dict[str, Any]:
    """Generate a preset YAML configuration for a GGUF model.

    Args:
        repo_id: HuggingFace repo ID.
        gguf_file: GGUF filename to use.
        context_size: Context window size (512-262144, default 8192).
        service_id: Custom service ID (auto-generated if not provided).

    Returns:
        Dict with 'preset' containing YAML string.
    """
    import re

    # Validate repo_id
    error = _validate_repo_id(repo_id)
    if error:
        return _validation_error(error)

    # Validate gguf_file (SEC-V1)
    if not gguf_file or not gguf_file.lower().endswith(".gguf"):
        return _validation_error("gguf_file must end with .gguf")

    # Check for dangerous patterns in gguf_file (path traversal, etc.)
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in gguf_file:
            return _validation_error("gguf_file contains invalid characters")

    # Reject absolute paths in gguf_file
    if gguf_file.startswith("/"):
        return _validation_error("gguf_file cannot be an absolute path")

    # Validate context_size
    if context_size < _MIN_CONTEXT_SIZE:
        return _validation_error(f"context_size must be at least {_MIN_CONTEXT_SIZE}")
    if context_size > _MAX_CONTEXT_SIZE:
        return _validation_error(f"context_size cannot exceed {_MAX_CONTEXT_SIZE}")

    # Generate service_id from repo if not provided, or validate if provided (SEC-V1)
    if service_id is None:
        # Extract model name from repo_id: "unsloth/Qwen3-GGUF" -> "qwen3"
        _, repo_name = repo_id.split("/")
        # Remove -GGUF suffix and clean up
        clean_name = re.sub(r"-?GGUF$", "", repo_name, flags=re.IGNORECASE)
        service_id = sanitize_name(clean_name.lower().replace(" ", "-"))
    else:
        # Validate user-provided service_id for dangerous patterns (SEC-V1)
        for pattern in _DANGEROUS_PATTERNS:
            if pattern in service_id:
                return _validation_error("service_id contains invalid characters")
        # Sanitize for terminal escapes (SEC-E3)
        service_id = sanitize_name(service_id)

    # Build preset YAML
    preset_yaml = f"""# Generated preset for {repo_id}
# Model: {gguf_file}

services:
  {service_id}:
    driver: llamacpp
    port: 8080
    vram_mb: auto
    extra_config:
      model: hf://{repo_id}/{gguf_file}
      context_size: {context_size}
      n_gpu_layers: -1

modes:
  {service_id}:
    name: "{service_id.replace("-", " ").title()}"
    services:
      - {service_id}
"""

    return {"preset": preset_yaml, "service_id": service_id}


# ---------------------------------------------------------------------------
# Driver docs tool (read-only)
# ---------------------------------------------------------------------------

# SEC-V1: version must be alphanumeric, dots, hyphens only
_VERSION_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9.\-]{0,63}$")

# SEC-V1: section name must not contain dangerous patterns
_SECTION_MAX_LENGTH = 200


def _validate_driver(driver: str) -> dict[str, str] | None:
    """Validate driver name for docs fetcher. Returns error dict or None."""
    allowed = {"llamacpp", "vllm"}
    if driver not in allowed:
        return _validation_error(f"driver must be one of: {', '.join(sorted(allowed))}")
    return None


def _validate_version(version: str) -> dict[str, str] | None:
    """Validate version string format (SEC-V1). Returns error dict or None."""
    if not _VERSION_RE.match(version):
        return _validation_error("version contains invalid characters")
    return None


def _validate_section(section: str) -> dict[str, str] | None:
    """Validate section name (SEC-V1). Returns error dict or None."""
    if len(section) > _SECTION_MAX_LENGTH:
        return _validation_error(f"section exceeds maximum length of {_SECTION_MAX_LENGTH}")
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in section:
            return _validation_error("section contains invalid characters")
    return None


async def fetch_driver_docs(
    driver: str,
    ctx: Context,
    version: str | None = None,
    section: str | None = None,
) -> dict[str, Any]:
    """Fetch driver documentation (llama.cpp or vLLM) for command-line flags and config.

    Args:
        driver: Driver name ("llamacpp" or "vllm").
        version: Explicit version. Auto-detected if omitted.
        section: Optional section header to return (e.g., "Server options").

    Returns:
        Dict with driver, version, source_url, content, and sections.
    """
    # Validate inputs (SEC-V1)
    error = _validate_driver(driver)
    if error:
        return error

    if version is not None:
        error = _validate_version(version)
        if error:
            return error

    if section is not None:
        error = _validate_section(section)
        if error:
            return error

    try:
        fetcher = DriverDocsFetcher()
        docs = await fetcher.fetch(driver=driver, version=version, section=section)

        return {
            "driver": docs.driver,
            "version": docs.version,
            "source_url": docs.source_url,
            "content": docs.content,
            "sections": docs.sections,
            "cached_at": docs.cached_at.isoformat(),
        }

    except DocsNotFoundError:
        return _not_found_error(f"Documentation not found for {driver}")


# ---------------------------------------------------------------------------
# RLM Consult tool (read-only, multi-step reasoning)
# ---------------------------------------------------------------------------

# SEC-V1: query must not be empty and must not exceed a reasonable length
_QUERY_MAX_LENGTH = 2000
_MAX_TURNS_HARD_LIMIT = 10


def _make_sync_wrapper(
    coro_func: Any,
    loop: asyncio.AbstractEventLoop,
    **bound_kwargs: Any,
) -> Any:
    """Create a sync callable that bridges to an async MCP tool.

    The RLM environment's ``execute_code`` uses ``exec()`` in a thread,
    so tool wrappers must be synchronous. This creates a sync function
    that submits the async coroutine back to the main event loop and
    blocks until it completes.
    """

    @functools.wraps(coro_func)
    def wrapper(**kwargs: Any) -> Any:
        merged = {**bound_kwargs, **kwargs}
        future = asyncio.run_coroutine_threadsafe(coro_func(**merged), loop)
        return future.result(timeout=30)

    return wrapper


def _build_tool_wrappers(ctx: Context) -> dict[str, Any]:
    """Build sync tool wrappers that bridge MCP Context to the RLM environment.

    Only read-only tools are exposed. Each wrapper captures ``ctx`` and the
    running event loop so it can be called synchronously from the REPL thread.
    """
    loop = asyncio.get_running_loop()

    return {
        "gpu_status": _make_sync_wrapper(gpu_status, loop, ctx=ctx),
        "list_gguf_files": _make_sync_wrapper(list_gguf_files, loop, ctx=ctx),
        "fetch_model_config": _make_sync_wrapper(fetch_model_config, loop, ctx=ctx),
        "fetch_driver_docs": _make_sync_wrapper(fetch_driver_docs, loop, ctx=ctx),
        "search_hf_models": _make_sync_wrapper(search_hf_models, loop, ctx=ctx),
        "simulate_mode": _make_sync_wrapper(simulate_mode, loop, ctx=ctx),
        "generate_preset": _make_sync_wrapper(generate_preset, loop, ctx=ctx),
    }


def _validate_consult_inputs(query: str, max_turns: int) -> dict[str, str] | None:
    """Validate consult tool inputs. Returns error dict or None."""
    if not query or not query.strip():
        return _validation_error("query is required")
    if len(query) > _QUERY_MAX_LENGTH:
        return _validation_error(f"query exceeds maximum length of {_QUERY_MAX_LENGTH}")
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in query:
            return _validation_error("query contains invalid characters")
    if max_turns < 1:
        return _validation_error("max_turns must be at least 1")
    if max_turns > _MAX_TURNS_HARD_LIMIT:
        return _validation_error(f"max_turns cannot exceed {_MAX_TURNS_HARD_LIMIT}")
    return None


async def consult(
    query: str,
    ctx: Context,
    max_turns: int = 5,
) -> dict[str, Any]:
    """Multi-step reasoning for complex GPU/model questions.

    Uses RLM to programmatically explore model metadata, driver docs,
    and VRAM constraints. Returns recommendations, never executes actions.

    Args:
        query: Natural language question (e.g., "Can I run Qwen3-235B on 24GB?").
        max_turns: Maximum REPL iterations (1-10, default 5).

    Returns:
        Dict with can_run, recommendation, reasoning_steps, suggested_commands,
        sources, turns_used, and incomplete flag.
    """
    error = _validate_consult_inputs(query, max_turns)
    if error:
        return error

    logger.info("MCP tool consult called: query=%r, max_turns=%d", query[:100], max_turns)

    # Build sync tool wrappers for the RLM environment.
    tool_wrappers = _build_tool_wrappers(ctx)

    try:
        orchestrator = RLMOrchestrator(tool_wrappers=tool_wrappers)
    except Exception as exc:
        logger.exception("Failed to create RLM orchestrator")
        return {"error": f"Failed to initialize consult: {exc}", "code": "INIT_ERROR"}

    # Call the orchestrator.  The result may be a coroutine (AsyncMock in
    # tests) or a plain value (sync RLMOrchestrator in production).
    # We handle both uniformly via ``inspect.isawaitable``.
    try:
        result = orchestrator.consult(query, max_turns=max_turns)
        if inspect.isawaitable(result):
            result = await result
    except TimeoutError:
        logger.exception("RLM consultation timed out")
        return {"error": "Consultation timed out", "code": "TIMEOUT_ERROR"}
    except Exception as exc:
        logger.exception("RLM consultation failed")
        error_msg = str(exc)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return {
                "error": "LLM API key not configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.",
                "code": "AUTH_ERROR",
            }
        return {"error": f"Consultation failed: {error_msg}", "code": "RLM_ERROR"}

    return result.model_dump(mode="json")


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

    # Discovery tools (read-only)
    server.tool(
        name="search_hf_models",
        description=(
            "Search HuggingFace for GGUF models. "
            "Filter by author (org), keyword search, or task type (code/chat/embed/reasoning)."
        ),
    )(search_hf_models)

    server.tool(
        name="list_gguf_files",
        description=(
            "List GGUF files in a HuggingFace repo with size and VRAM estimates. "
            "Optionally filter by VRAM budget."
        ),
    )(list_gguf_files)

    server.tool(
        name="list_model_files",
        description=(
            "List model files (GGUF or Safetensors) with format detection. "
            "Unified tool supporting both llama.cpp and vLLM formats."
        ),
    )(list_model_files)

    server.tool(
        name="fetch_model_config",
        description=(
            "Fetch config.json from a HuggingFace repo. "
            "Returns model architecture, context length, MoE status, and expert count."
        ),
    )(fetch_model_config)

    server.tool(
        name="generate_preset",
        description=(
            "Generate a preset YAML configuration for a GGUF model. "
            "Creates llama.cpp service config with specified context size."
        ),
    )(generate_preset)

    server.tool(
        name="fetch_driver_docs",
        description=(
            "Fetch driver documentation (llama.cpp or vLLM) for command-line flags and config. "
            "Auto-detects installed version. Supports section filtering."
        ),
    )(fetch_driver_docs)

    # RLM Consult tool (read-only, multi-step reasoning)
    server.tool(
        name="consult",
        description=(
            "Multi-step reasoning for complex GPU/model questions. "
            "Uses RLM to explore model metadata, driver docs, and VRAM constraints. "
            "Returns recommendations with source citations, never executes actions."
        ),
    )(consult)
