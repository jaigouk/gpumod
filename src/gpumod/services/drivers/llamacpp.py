"""LlamaCppDriver — service driver for llama.cpp servers in router mode.

Manages llama-server processes via systemd and communicates with the
llama.cpp HTTP API for health checks and model load/unload operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from gpumod.models import ServiceState, ServiceStatus
from gpumod.services import systemd
from gpumod.services.base import ServiceDriver
from gpumod.services.vram import InsufficientVRAMError

if TYPE_CHECKING:
    from gpumod.models import Service
    from gpumod.services.vram import VRAMTracker

logger = logging.getLogger(__name__)

# Safety margin for VRAM checks (in MB)
_VRAM_SAFETY_MARGIN_MB = 512


class LlamaCppDriver(ServiceDriver):
    """Driver for llama.cpp-based inference services (router mode).

    Parameters
    ----------
    http_timeout:
        Timeout in seconds for HTTP requests to the llama-server API.
    vram_tracker:
        Optional VRAMTracker for VRAM preflight checks before loading models.
        If provided, wake() will verify sufficient VRAM before calling /models/load.
    """

    def __init__(
        self,
        http_timeout: float = 10.0,
        vram_tracker: VRAMTracker | None = None,
    ) -> None:
        self._http_timeout = http_timeout
        self._vram_tracker = vram_tracker

    # ------------------------------------------------------------------
    # ServiceDriver interface
    # ------------------------------------------------------------------

    async def start(self, service: Service) -> None:
        """Start the llama-server via systemd."""
        self._validate_unit_name(service)
        assert service.unit_name is not None  # for type narrowing
        await systemd.start(service.unit_name)

    async def stop(self, service: Service) -> None:
        """Stop the llama-server via systemd."""
        self._validate_unit_name(service)
        assert service.unit_name is not None  # for type narrowing
        await systemd.stop(service.unit_name)

    async def health_check(self, service: Service) -> bool:
        """Check if the llama-server is healthy via GET /health."""
        if service.port is None:
            return False
        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                resp = await client.get(f"http://localhost:{service.port}/health")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def status(self, service: Service) -> ServiceStatus:
        """Determine current service state by combining systemd + loaded models."""
        try:
            assert service.unit_name is not None  # for type narrowing
            active = await systemd.is_active(service.unit_name)

            if not active:
                return ServiceStatus(state=ServiceState.STOPPED)

            models = await self._get_loaded_models(service)

            if not models:
                return ServiceStatus(state=ServiceState.SLEEPING, health_ok=True)

            return ServiceStatus(state=ServiceState.RUNNING, health_ok=True)

        except Exception:
            return ServiceStatus(state=ServiceState.UNKNOWN)

    # ------------------------------------------------------------------
    # Sleep / Wake (router mode: unload / load model)
    # ------------------------------------------------------------------

    async def sleep(self, service: Service, level: int = 1) -> None:
        """Unload model(s) to free VRAM.

        Discovers currently loaded models from the server and unloads them.
        Falls back to ``_resolve_model_name()`` if the server query fails.

        Note: llama.cpp router mode doesn't use sleep levels. The level
        parameter is accepted for interface compatibility but ignored.
        """
        del level  # unused in llama.cpp router mode
        loaded = await self._get_loaded_models(service)
        model_name = loaded[0]["id"] if loaded else self._resolve_model_name(service)

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            await client.post(
                f"http://localhost:{service.port}/models/unload",
                json={"model": model_name},
            )

    async def wake(self, service: Service) -> None:
        """Load a model into VRAM.

        Resolution order for which model to load:
        1. ``unit_vars.default_model`` — explicit model alias in the preset
        2. Discover available models from the server and match by ``model_id``
        3. Load the first available unloaded model

        Raises
        ------
        InsufficientVRAMError
            If a VRAMTracker is configured and there is not enough free VRAM
            to load the model (including safety margin).
        """
        # VRAM preflight check (gpumod-277)
        if self._vram_tracker is not None:
            usage = await self._vram_tracker.get_usage()
            required_mb = service.vram_mb + _VRAM_SAFETY_MARGIN_MB
            if usage.free_mb < required_mb:
                logger.error(
                    "Insufficient VRAM to load model for %r: need %dMB, only %dMB free",
                    service.id,
                    service.vram_mb,
                    usage.free_mb,
                )
                raise InsufficientVRAMError(
                    required_mb=service.vram_mb,
                    available_mb=usage.free_mb,
                )

        model = self._get_default_model(service)

        if not model:
            model = await self._discover_model_to_load(service)

        if not model:
            logger.warning("No model found to load for service %r", service.id)
            return

        logger.info("Loading model %r for service %r", model, service.id)
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            await client.post(
                f"http://localhost:{service.port}/models/load",
                json={"model": model},
            )

    @property
    def supports_sleep(self) -> bool:
        """LlamaCppDriver supports router-mode sleep (model unload/load)."""
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_all_models(self, service: Service) -> list[dict[str, Any]]:
        """Fetch all models (loaded + unloaded) from GET /models.

        Parses the OAI-compatible response format: ``{"data": [...], "object": "list"}``.
        Returns an empty list on any connection or parsing error.
        """
        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                resp = await client.get(f"http://localhost:{service.port}/models")
                body = resp.json()
                if isinstance(body, dict):
                    result: list[dict[str, Any]] = body.get("data", [])
                    return result
                # Fallback for flat-list responses (older llama.cpp versions)
                if isinstance(body, list):
                    return body
                return []
        except (httpx.ConnectError, httpx.TimeoutException, OSError, ValueError):
            return []

    async def _get_loaded_models(self, service: Service) -> list[dict[str, Any]]:
        """Fetch list of currently loaded models from GET /models.

        Filters the full model list to only models with ``status.value == "loaded"``.
        Returns an empty list on any connection or parsing error.
        """
        all_models = await self._get_all_models(service)
        return [m for m in all_models if m.get("status", {}).get("value") == "loaded"]

    async def _discover_model_to_load(self, service: Service) -> str | None:
        """Find a model to load from the server's available models.

        Tries to match by ``service.model_id`` first, then falls back to
        the first unloaded model.
        """
        all_models = await self._get_all_models(service)
        unloaded = [m for m in all_models if m.get("status", {}).get("value") != "loaded"]

        if not unloaded:
            return None

        # Try to match by model_id (check if model_id substring is in alias)
        if service.model_id:
            # Extract the model name part (after last /)
            model_stem = service.model_id.rsplit("/", 1)[-1]
            # Strip common suffixes like -GGUF
            for suffix in ("-GGUF", "-gguf"):
                if model_stem.endswith(suffix):
                    model_stem = model_stem[: -len(suffix)]
                    break

            for m in unloaded:
                alias: str = m.get("id", "")
                if model_stem and model_stem in alias:
                    return alias

        # Fall back to first unloaded model
        first_id: str = unloaded[0].get("id", "")
        return first_id

    @staticmethod
    def _get_default_model(service: Service) -> str | None:
        """Get the explicit default_model from unit_vars, if configured."""
        unit_vars = service.extra_config.get("unit_vars", {})
        if isinstance(unit_vars, dict):
            default = unit_vars.get("default_model")
            if default:
                return str(default)
        return None

    @staticmethod
    def _resolve_model_name(service: Service) -> str:
        """Get the model name from extra_config, falling back to model_id."""
        name: str | None = service.extra_config.get("model_name")
        if name:
            return name
        return service.model_id or ""

    @staticmethod
    def _validate_unit_name(service: Service) -> None:
        """Raise ValueError if the service has no unit_name."""
        if not service.unit_name:
            msg = f"Service {service.id!r} has no unit_name configured"
            raise ValueError(msg)
