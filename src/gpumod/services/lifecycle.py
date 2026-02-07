"""LifecycleManager — handles service start/stop with dependency ordering and health waiting."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from gpumod.models import ServiceState

if TYPE_CHECKING:
    from gpumod.models import Service
    from gpumod.services.base import ServiceDriver
    from gpumod.services.registry import ServiceRegistry

logger = logging.getLogger(__name__)


class LifecycleError(Exception):
    """Raised when a lifecycle operation fails.

    Attributes
    ----------
    service_id:
        The ID of the service that caused the error.
    operation:
        The operation that failed (e.g. "start", "stop").
    reason:
        A human-readable description of what went wrong.
    """

    def __init__(self, service_id: str, operation: str, reason: str) -> None:
        self.service_id = service_id
        self.operation = operation
        self.reason = reason
        super().__init__(f"Lifecycle error for {service_id!r} during {operation}: {reason}")


class LifecycleManager:
    """Manages service start/stop with dependency ordering and health waiting.

    Parameters
    ----------
    registry:
        The ServiceRegistry used to look up services and their drivers.
    """

    def __init__(self, registry: ServiceRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, service_id: str) -> None:
        """Start a service and all its transitive dependencies in topological order.

        Already-running dependencies are skipped.
        After each service is started, waits for its health check to pass.
        """
        logger.info("Starting service %r (with dependencies)", service_id)
        service = await self._registry.get(service_id)
        start_order = await self._resolve_start_order(service)

        for svc in start_order:
            driver = self._registry.get_driver(svc.driver)

            # Skip already-running services
            status = await driver.status(svc)
            if status.state in (ServiceState.RUNNING, ServiceState.SLEEPING):
                logger.info("Service %r already running, skipping", svc.id)
                continue

            logger.info("Starting service %r", svc.id)
            await driver.start(svc)
            await self._wait_for_healthy(svc, driver)
            logger.info("Service %r started and healthy", svc.id)

    async def stop(self, service_id: str) -> None:
        """Stop a service and all its transitive dependents in reverse dependency order.

        Already-stopped dependents are skipped.
        Dependents are stopped first (deepest first), then the target service.
        """
        logger.info("Stopping service %r (with dependents)", service_id)
        service = await self._registry.get(service_id)
        stop_order = await self._resolve_stop_order(service)

        for svc in stop_order:
            driver = self._registry.get_driver(svc.driver)

            # Skip already-stopped services
            status = await driver.status(svc)
            if status.state in (ServiceState.STOPPED, ServiceState.UNKNOWN):
                logger.info("Service %r already stopped, skipping", svc.id)
                continue

            logger.info("Stopping service %r", svc.id)
            await driver.stop(svc)
            logger.info("Service %r stopped", svc.id)

    async def restart(self, service_id: str) -> None:
        """Restart a service by stopping it (and dependents) then starting it (and deps)."""
        logger.info("Restarting service %r", service_id)
        await self.stop(service_id)
        await self.start(service_id)

    # ------------------------------------------------------------------
    # Health waiting
    # ------------------------------------------------------------------

    async def _wait_for_healthy(
        self,
        service: Service,
        driver: ServiceDriver,
        timeout: float = 120.0,
        poll_interval: float = 1.0,
    ) -> None:
        """Poll driver.health_check until it returns True or timeout is exceeded.

        Parameters
        ----------
        service:
            The service to check health for.
        driver:
            The driver to use for health checking.
        timeout:
            Maximum seconds to wait for health.
        poll_interval:
            Seconds between health check polls.

        Raises
        ------
        LifecycleError
            If the health check does not pass within the timeout.
        """
        start_time = time.monotonic()

        while True:
            healthy = await driver.health_check(service)
            if healthy:
                return

            elapsed = time.monotonic() - start_time
            if elapsed + poll_interval > timeout:
                logger.warning(
                    "Health check timed out for service %r after %.1fs",
                    service.id,
                    timeout,
                )
                raise LifecycleError(
                    service_id=service.id,
                    operation="start",
                    reason=f"health check timed out after {timeout}s",
                )

            await asyncio.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    async def _resolve_start_order(self, service: Service) -> list[Service]:
        """Resolve transitive dependencies and return them in topological start order.

        The returned list starts with the deepest dependency and ends with
        the target service itself.
        """
        visited: set[str] = set()
        order: list[Service] = []

        async def _visit(svc: Service) -> None:
            if svc.id in visited:
                return
            visited.add(svc.id)

            for dep_id in svc.depends_on:
                dep = await self._registry.get(dep_id)
                await _visit(dep)

            order.append(svc)

        await _visit(service)
        return order

    async def _resolve_stop_order(self, service: Service) -> list[Service]:
        """Resolve transitive dependents and return them in stop order.

        The returned list starts with the deepest dependents (leaf nodes)
        and ends with the target service itself.
        """
        visited: set[str] = set()
        order: list[Service] = []

        async def _visit(svc: Service) -> None:
            if svc.id in visited:
                return
            visited.add(svc.id)

            dependents = await self._registry.get_dependents(svc.id)
            for dep in dependents:
                await _visit(dep)

            order.append(svc)

        await _visit(service)
        return order
