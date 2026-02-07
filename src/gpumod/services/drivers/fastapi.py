"""FastAPI service driver -- manages custom FastAPI servers via systemd."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from gpumod.models import ServiceState, ServiceStatus
from gpumod.services import systemd
from gpumod.services.base import ServiceDriver

if TYPE_CHECKING:
    from gpumod.models import Service


class FastAPIDriver(ServiceDriver):
    """Driver for custom FastAPI services (e.g., ASR, TTS).

    Simpler than vLLM/llama.cpp drivers: no sleep/wake support.
    Uses ``service.health_endpoint`` for health checks.
    """

    async def start(self, service: Service) -> None:
        """Start the service via systemd."""
        if not service.unit_name:
            msg = "Service has no unit_name configured"
            raise ValueError(msg)
        await systemd.start(service.unit_name)

    async def stop(self, service: Service) -> None:
        """Stop the service via systemd."""
        if not service.unit_name:
            msg = "Service has no unit_name configured"
            raise ValueError(msg)
        await systemd.stop(service.unit_name)

    async def health_check(self, service: Service) -> bool:
        """Check service health via its HTTP health endpoint."""
        endpoint = service.health_endpoint
        url = f"http://localhost:{service.port}{endpoint}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5.0)
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def status(self, service: Service) -> ServiceStatus:
        """Get the current status combining systemd state and health check.

        Never raises -- returns UNKNOWN on unexpected errors.
        """
        try:
            active = await systemd.is_active(service.unit_name or "")
            if not active:
                return ServiceStatus(state=ServiceState.STOPPED)

            healthy = await self.health_check(service)
            if healthy:
                return ServiceStatus(
                    state=ServiceState.RUNNING,
                    health_ok=True,
                )
            return ServiceStatus(
                state=ServiceState.UNHEALTHY,
                health_ok=False,
            )
        except Exception as exc:
            return ServiceStatus(
                state=ServiceState.UNKNOWN,
                last_error=str(exc),
            )

    @property
    def supports_sleep(self) -> bool:
        """FastAPI driver does not support sleep/wake."""
        return False
