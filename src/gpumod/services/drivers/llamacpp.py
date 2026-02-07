"""LlamaCppDriver — service driver for llama.cpp servers in router mode.

Manages llama-server processes via systemd and communicates with the
llama.cpp HTTP API for health checks and model load/unload operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from gpumod.models import ServiceState, ServiceStatus
from gpumod.services import systemd
from gpumod.services.base import ServiceDriver

if TYPE_CHECKING:
    from gpumod.models import Service


class LlamaCppDriver(ServiceDriver):
    """Driver for llama.cpp-based inference services (router mode).

    Parameters
    ----------
    http_timeout:
        Timeout in seconds for HTTP requests to the llama-server API.
    """

    def __init__(self, http_timeout: float = 10.0) -> None:
        self._http_timeout = http_timeout

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

    async def sleep(self, service: Service, level: str = "l1") -> None:
        """Unload the model to free VRAM (router-mode sleep)."""
        model_name = self._resolve_model_name(service)
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            await client.post(
                f"http://localhost:{service.port}/models/unload",
                json={"model": model_name},
            )

    async def wake(self, service: Service) -> None:
        """Load the model back into VRAM (router-mode wake)."""
        model_path = service.extra_config.get("model_path", "")
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            await client.post(
                f"http://localhost:{service.port}/models/load",
                json={"model": model_path},
            )

    @property
    def supports_sleep(self) -> bool:
        """LlamaCppDriver supports router-mode sleep (model unload/load)."""
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_loaded_models(self, service: Service) -> list[dict[str, Any]]:
        """Fetch list of loaded models from GET /models.

        Returns an empty list on any connection or parsing error.
        """
        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                resp = await client.get(f"http://localhost:{service.port}/models")
                data: list[dict[str, Any]] = resp.json()
                return data
        except (httpx.ConnectError, httpx.TimeoutException, OSError, ValueError):
            return []

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
