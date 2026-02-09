"""MCP server foundation for gpumod.

Creates and configures the FastMCP server with:
- Lifespan management (DB connect on startup, close on shutdown)
- Error sanitization middleware (SEC-E1)
- Request ID middleware (SEC-A2)
- Rate limiting middleware (SEC-R2)
- Strict input validation (SEC-V2)
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext

from gpumod.config import get_settings
from gpumod.db import Database
from gpumod.mcp_resources import register_resources
from gpumod.mcp_tools import register_tools
from gpumod.registry import ModelRegistry
from gpumod.services.lifecycle import LifecycleManager
from gpumod.services.manager import ServiceManager
from gpumod.services.unit_installer import UnitFileInstaller
from gpumod.services.registry import ServiceRegistry
from gpumod.services.sleep import SleepController
from gpumod.services.vram import VRAMTracker
from gpumod.simulation import SimulationEngine
from gpumod.templates.engine import TemplateEngine
from gpumod.templates.presets import PresetLoader

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request ID context variable (SEC-A2)
# ---------------------------------------------------------------------------

request_id_var: ContextVar[str] = ContextVar("request_id_var", default="")

# ---------------------------------------------------------------------------
# Error sanitization (SEC-E1)
# ---------------------------------------------------------------------------

# Patterns for absolute paths to strip from error messages.
# Each pattern matches a common system path prefix through to whitespace or end.
_PATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"/home/\S+"),
    re.compile(r"/tmp/\S+"),  # noqa: S108 -- sanitizing /tmp paths in output
    re.compile(r"/var/\S+"),
    re.compile(r"/usr/\S+"),
    re.compile(r"/etc/\S+"),
    re.compile(r"/opt/\S+"),
]

# Pattern for Python traceback blocks.
_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\):.*",
    re.DOTALL,
)

# Pattern for individual 'File "..." , line N' references.
_FILE_LINE_RE = re.compile(r'File "[^"]+", line \d+')

# Pattern for .py or .db file references that may remain after path stripping.
_PY_DB_FILE_RE = re.compile(r"\S+\.(?:py|db)\b")


def sanitize_error_message(message: str) -> str:
    """Sanitize an error message by removing internal paths and tracebacks.

    Implements SEC-E1 from the security specification. Strips absolute file
    paths, Python traceback blocks, and file/line references to prevent
    leaking internal architecture details to LLM clients.

    Parameters
    ----------
    message:
        The raw error message string.

    Returns
    -------
    str
        A cleaned error message safe for returning to MCP clients.
    """
    # If the message contains a full traceback, replace the whole thing.
    if _TRACEBACK_RE.search(message):
        return "Internal error: please check server logs"

    result = message

    # Strip absolute paths.
    for pattern in _PATH_PATTERNS:
        result = pattern.sub("[redacted]", result)

    # Strip File "...", line N references.
    result = _FILE_LINE_RE.sub("[redacted]", result)

    # Strip any remaining .py or .db file references.
    result = _PY_DB_FILE_RE.sub("[redacted]", result)

    # Clean up multiple consecutive [redacted] tokens and whitespace.
    result = re.sub(r"(\[redacted\]\s*)+", "[redacted] ", result).strip()

    # If the result is just redacted markers, give a generic message.
    if result == "[redacted]" or not result:
        return "Internal error: please check server logs"

    return result


# ---------------------------------------------------------------------------
# Error sanitization middleware (SEC-E1)
# ---------------------------------------------------------------------------


class ErrorSanitizationMiddleware(Middleware):
    """Middleware that sanitizes error messages before they reach the client.

    Catches exceptions from tool and resource calls, logs the full error
    for operators, then re-raises with a sanitized message.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext[Any],
        call_next: Any,
    ) -> Any:
        """Intercept tool call errors and sanitize their messages."""
        try:
            return await call_next(context)
        except Exception as exc:
            raw_message = str(exc)
            sanitized = sanitize_error_message(raw_message)
            logger.exception("Tool call error (sanitized for client)")
            raise type(exc)(sanitized) from exc

    async def on_read_resource(
        self,
        context: MiddlewareContext[Any],
        call_next: Any,
    ) -> Any:
        """Intercept resource read errors and sanitize their messages."""
        try:
            return await call_next(context)
        except Exception as exc:
            raw_message = str(exc)
            sanitized = sanitize_error_message(raw_message)
            logger.exception("Resource read error (sanitized for client)")
            raise type(exc)(sanitized) from exc


# ---------------------------------------------------------------------------
# Request ID middleware (SEC-A2)
# ---------------------------------------------------------------------------


class RequestIDMiddleware(Middleware):
    """Middleware that generates a UUID per MCP request for correlation (SEC-A2).

    Sets a unique request ID in the ``request_id_var`` context variable for
    each tool call and resource read. The ID is logged at INFO level and can
    be used downstream for request correlation across service layers.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext[Any],
        call_next: Any,
    ) -> Any:
        """Generate a request ID and set it before processing the tool call."""
        rid = str(uuid.uuid4())
        token = request_id_var.set(rid)
        try:
            logger.info("MCP tool call [request_id=%s]", rid)
            return await call_next(context)
        finally:
            request_id_var.reset(token)

    async def on_read_resource(
        self,
        context: MiddlewareContext[Any],
        call_next: Any,
    ) -> Any:
        """Generate a request ID and set it before reading the resource."""
        rid = str(uuid.uuid4())
        token = request_id_var.set(rid)
        try:
            logger.info("MCP resource read [request_id=%s]", rid)
            return await call_next(context)
        finally:
            request_id_var.reset(token)


# ---------------------------------------------------------------------------
# Rate limiting middleware (SEC-R2)
# ---------------------------------------------------------------------------


class RateLimitMiddleware(Middleware):
    """Middleware that enforces per-client request rate limiting (SEC-R2, SEC-R3).

    Uses a sliding window algorithm to track request timestamps per client
    and reject requests that exceed the configured limit. Each client gets
    an independent quota keyed by client ID.

    Parameters
    ----------
    max_requests:
        Maximum number of requests allowed within the window. Default 10.
    window_seconds:
        Duration of the sliding window in seconds. Default 1.0.
    """

    def __init__(self, max_requests: int = 10, window_seconds: float = 1.0) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._client_requests: dict[str, list[float]] = {}

    def _get_client_id(self, context: MiddlewareContext[Any]) -> str:
        """Extract a client identifier from the middleware context.

        Falls back to ``'__default__'`` if no client info is available.
        """
        client_id = getattr(context, "client_id", None)
        if client_id is not None:
            return str(client_id)
        return "__default__"

    def _check_rate(self, client_id: str) -> None:
        """Check if the current request exceeds the rate limit for a client.

        Raises
        ------
        RuntimeError
            If the rate limit has been exceeded.
        """
        now = time.monotonic()
        # Remove expired timestamps outside the sliding window
        requests = self._client_requests.get(client_id, [])
        requests = [t for t in requests if now - t < self._window_seconds]
        if len(requests) >= self._max_requests:
            msg = "Rate limit exceeded"
            raise RuntimeError(msg)
        requests.append(now)
        self._client_requests[client_id] = requests

    async def on_call_tool(
        self,
        context: MiddlewareContext[Any],
        call_next: Any,
    ) -> Any:
        """Enforce rate limit on tool calls."""
        client_id = self._get_client_id(context)
        self._check_rate(client_id)
        return await call_next(context)

    async def on_read_resource(
        self,
        context: MiddlewareContext[Any],
        call_next: Any,
    ) -> Any:
        """Enforce rate limit on resource reads."""
        client_id = self._get_client_id(context)
        self._check_rate(client_id)
        return await call_next(context)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def gpumod_lifespan(
    server: FastMCP[dict[str, Any]],
    *,
    db_path: Path | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Async context manager for gpumod MCP server lifecycle.

    On startup: creates the Database and all backend services (similar to
    ``cli.py:create_context``).  Yields a dict containing all services.
    On shutdown: closes the database connection.

    Parameters
    ----------
    server:
        The FastMCP server instance (received from the lifespan protocol).
    db_path:
        Optional path to the SQLite database.  Defaults to
        the value from centralized settings.
    """
    settings = get_settings()
    resolved_db_path = db_path if db_path is not None else settings.db_path

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
    simulation = SimulationEngine(db=db, vram=vram, model_registry=model_registry)
    template_engine = TemplateEngine()

    # Discover built-in presets directory from centralized settings.
    builtin_presets_dir = settings.presets_dir
    preset_dirs: list[Path] = []
    if builtin_presets_dir.is_dir():
        preset_dirs.append(builtin_presets_dir)
    preset_loader = PresetLoader(preset_dirs=preset_dirs)

    context: dict[str, Any] = {
        "db": db,
        "registry": registry,
        "lifecycle": lifecycle,
        "vram": vram,
        "sleep": sleep,
        "manager": manager,
        "model_registry": model_registry,
        "simulation": simulation,
        "template_engine": template_engine,
        "preset_loader": preset_loader,
    }

    logger.info("gpumod MCP server started (db=%s)", resolved_db_path)

    try:
        yield context
    finally:
        await db.close()
        logger.info("gpumod MCP server shut down")


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_mcp_server(
    *,
    db_path: Path | None = None,
) -> FastMCP[dict[str, Any]]:
    """Create and configure the gpumod MCP server.

    Parameters
    ----------
    db_path:
        Optional path to the SQLite database.  Passed through to the
        lifespan context manager.

    Returns
    -------
    FastMCP
        A fully-configured FastMCP server instance with error sanitization
        middleware, request ID middleware, rate limiting, strict input
        validation, and a lifespan that manages the backend service
        dependencies.
    """

    @asynccontextmanager
    async def _lifespan(server: FastMCP[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        async with gpumod_lifespan(server, db_path=db_path) as ctx:
            yield ctx

    server: FastMCP[dict[str, Any]] = FastMCP(
        name="gpumod",
        instructions=(
            "GPU Service Manager for ML workloads. "
            "Manages vLLM, llama.cpp, and FastAPI services on NVIDIA GPUs."
        ),
        lifespan=_lifespan,
        middleware=[
            RequestIDMiddleware(),
            ErrorSanitizationMiddleware(),
            RateLimitMiddleware(),
        ],
        strict_input_validation=True,
    )

    register_resources(server)
    register_tools(server)

    return server
