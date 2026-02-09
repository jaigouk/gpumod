"""VLLMDriver — service driver for vLLM inference servers.

Manages vLLM processes via systemd and communicates with the
vLLM HTTP API for health checks, sleep, and wake operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from gpumod.models import ServiceState, ServiceStatus
from gpumod.services import systemd
from gpumod.services.base import ServiceDriver

if TYPE_CHECKING:
    from gpumod.models import Service


class VLLMDriver(ServiceDriver):
    """Driver for vLLM-based inference services.

    Parameters
    ----------
    http_timeout:
        Timeout in seconds for HTTP requests to the vLLM API.
    """

    def __init__(self, http_timeout: float = 10.0) -> None:
        self._http_timeout = http_timeout

    # ------------------------------------------------------------------
    # ServiceDriver interface
    # ------------------------------------------------------------------

    async def start(self, service: Service) -> None:
        """Start the vLLM service via systemd."""
        self._validate_unit_name(service)
        assert service.unit_name is not None  # for type narrowing
        await systemd.start(service.unit_name)

    async def stop(self, service: Service) -> None:
        """Stop the vLLM service via systemd."""
        self._validate_unit_name(service)
        assert service.unit_name is not None  # for type narrowing
        await systemd.stop(service.unit_name)

    async def health_check(self, service: Service) -> bool:
        """Check if the vLLM service is healthy via GET /health."""
        if service.port is None:
            return False
        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                resp = await client.get(f"http://localhost:{service.port}/health")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def status(self, service: Service) -> ServiceStatus:
        """Determine the current service state by combining systemd + health."""
        try:
            assert service.unit_name is not None  # for type narrowing
            active = await systemd.is_active(service.unit_name)

            if not active:
                return ServiceStatus(state=ServiceState.STOPPED)

            healthy = await self.health_check(service)
            if not healthy:
                return ServiceStatus(
                    state=ServiceState.UNHEALTHY,
                    health_ok=False,
                )

            sleeping = await self.is_sleeping(service)
            if sleeping:
                return ServiceStatus(state=ServiceState.SLEEPING, health_ok=True)

            return ServiceStatus(state=ServiceState.RUNNING, health_ok=True)

        except Exception:
            return ServiceStatus(state=ServiceState.UNKNOWN)

    # ------------------------------------------------------------------
    # Sleep / Wake
    # ------------------------------------------------------------------

    async def sleep(self, service: Service, level: int = 1) -> None:
        """Put the vLLM model to sleep via POST /sleep?level=N.

        Parameters
        ----------
        service:
            The service to put to sleep.
        level:
            Sleep level (1 or 2).
            L1: Offloads weights to CPU RAM, discards KV cache. Wake is instant.
            L2: Discards weights and KV cache. Wake takes 2-5s.

        Raises
        ------
        ValueError:
            If level is not 1 or 2.
        httpx.ConnectError:
            If the service is not reachable.
        """
        if level not in (1, 2):
            msg = f"Invalid sleep level: {level}. Must be 1 or 2."
            raise ValueError(msg)

        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            await client.post(
                f"http://localhost:{service.port}/sleep",
                params={"level": level},
            )

    async def wake(self, service: Service, tags: str | None = None) -> None:
        """Wake the vLLM model via POST /wake_up.

        Parameters
        ----------
        service:
            The service to wake.
        tags:
            Optional comma-separated tags for selective wake (e.g., "weights").
            Used for L2 sleep to wake components incrementally.

        Raises
        ------
        httpx.ConnectError:
            If the service is not reachable.
        """
        params = {"tags": tags} if tags else None
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            await client.post(
                f"http://localhost:{service.port}/wake_up",
                params=params,
            )

    async def is_sleeping(self, service: Service) -> bool:
        """Check if the vLLM model is currently sleeping via GET /is_sleeping.

        Returns False if the service is not reachable, endpoint doesn't exist
        (dev mode off), or response is malformed.

        Returns
        -------
        bool:
            True if the model is sleeping, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                resp = await client.get(f"http://localhost:{service.port}/is_sleeping")
                if resp.status_code != 200:
                    return False
                data: dict[str, bool] = resp.json()
                return bool(data.get("is_sleeping", False))
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    @property
    def supports_sleep(self) -> bool:
        """VLLMDriver supports L1 and L2 sleep."""
        return True

    @staticmethod
    def _validate_unit_name(service: Service) -> None:
        """Raise ValueError if the service has no unit_name."""
        if not service.unit_name:
            msg = f"Service {service.id!r} has no unit_name configured"
            raise ValueError(msg)
