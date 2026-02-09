"""LifecycleManager — handles service start/stop with dependency ordering and health waiting."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gpumod.models import ServiceState, SleepMode
from gpumod.services.systemd import get_unit_state, journal_logs

if TYPE_CHECKING:
    from gpumod.models import Service
    from gpumod.services.base import ServiceDriver
    from gpumod.services.registry import ServiceRegistry
    from gpumod.services.unit_installer import UnitFileInstaller

# States that indicate the process has exited and won't recover.
_DEAD_STATES: frozenset[str] = frozenset({"failed", "inactive", "dead"})

logger = logging.getLogger(__name__)


@dataclass
class SleepResult:
    """Result of a sleep operation.

    Attributes
    ----------
    success:
        True if the operation completed (or was idempotent no-op).
    skipped:
        True if the operation was skipped (already sleeping, or not applicable).
    reason:
        Human-readable explanation if operation failed or was skipped.
    latency_ms:
        Time taken in milliseconds, if measured.
    """

    success: bool
    skipped: bool = False
    reason: str | None = None
    latency_ms: float | None = None


@dataclass
class WakeResult:
    """Result of a wake operation.

    Attributes
    ----------
    success:
        True if the operation completed (or was idempotent no-op).
    skipped:
        True if the operation was skipped (already running, or not applicable).
    reason:
        Human-readable explanation if operation failed or was skipped.
    latency_ms:
        Time taken in milliseconds, if measured.
    """

    success: bool
    skipped: bool = False
    reason: str | None = None
    latency_ms: float | None = None


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

    def __init__(
        self,
        registry: ServiceRegistry,
        *,
        unit_installer: UnitFileInstaller | None = None,
    ) -> None:
        self._registry = registry
        self._unit_installer = unit_installer

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

        # Auto-install missing unit files before starting
        if self._unit_installer is not None:
            for svc in start_order:
                await self._unit_installer.ensure_unit_file(svc)
            await self._unit_installer.daemon_reload_if_needed()

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

            # Router-mode services need an explicit model load after start
            if driver.supports_sleep and svc.sleep_mode == SleepMode.ROUTER:
                logger.info("Loading model for router-mode service %r", svc.id)
                await driver.wake(svc)

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

    async def sleep(self, service_id: str, level: int = 1) -> SleepResult:
        """Put a service to sleep without stopping the process.

        This is an idempotent operation: sleeping an already-sleeping service
        returns success with skipped=True.

        Parameters
        ----------
        service_id:
            The ID of the service to sleep.
        level:
            Sleep level (1 or 2). Level semantics are driver-specific.

        Returns
        -------
        SleepResult:
            Result indicating success, skip, or failure with reason.
        """
        start_time = time.monotonic()

        try:
            service = await self._registry.get(service_id)
        except KeyError:
            return SleepResult(
                success=False,
                reason=f"Service not found: {service_id!r}",
            )

        driver = self._registry.get_driver(service.driver)
        status = await driver.status(service)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Check if driver supports sleep
        if not driver.supports_sleep:
            return SleepResult(
                success=False,
                skipped=True,
                reason=f"Driver does not support sleep: {service.driver}",
                latency_ms=elapsed_ms,
            )

        # Idempotent: already sleeping is a success
        if status.state == ServiceState.SLEEPING:
            return SleepResult(
                success=True,
                skipped=True,
                reason="Already sleeping",
                latency_ms=elapsed_ms,
            )

        # Must be running to sleep
        if status.state != ServiceState.RUNNING:
            return SleepResult(
                success=False,
                reason=f"Service is {status.state}, must be RUNNING to sleep",
                latency_ms=elapsed_ms,
            )

        # Perform sleep
        logger.info("Putting service %r to sleep (level=%d)", service_id, level)
        await driver.sleep(service, level)
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info("Service %r is now sleeping", service_id)

        return SleepResult(success=True, latency_ms=elapsed_ms)

    async def wake(self, service_id: str) -> WakeResult:
        """Wake a sleeping service.

        This is an idempotent operation: waking an already-running service
        returns success with skipped=True.

        Parameters
        ----------
        service_id:
            The ID of the service to wake.

        Returns
        -------
        WakeResult:
            Result indicating success, skip, or failure with reason.
        """
        start_time = time.monotonic()

        try:
            service = await self._registry.get(service_id)
        except KeyError:
            return WakeResult(
                success=False,
                reason=f"Service not found: {service_id!r}",
            )

        driver = self._registry.get_driver(service.driver)
        status = await driver.status(service)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Check if driver supports sleep/wake
        if not driver.supports_sleep:
            return WakeResult(
                success=False,
                reason=f"Driver does not support sleep/wake: {service.driver}",
                latency_ms=elapsed_ms,
            )

        # Idempotent: already running is a success
        if status.state == ServiceState.RUNNING:
            return WakeResult(
                success=True,
                skipped=True,
                reason="Already running",
                latency_ms=elapsed_ms,
            )

        # Must be sleeping to wake (stopped services can't be woken)
        if status.state != ServiceState.SLEEPING:
            return WakeResult(
                success=False,
                reason=f"Service is {status.state}, must be SLEEPING to wake",
                latency_ms=elapsed_ms,
            )

        # Perform wake
        logger.info("Waking service %r", service_id)
        await driver.wake(service)
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info("Service %r is now awake", service_id)

        return WakeResult(success=True, latency_ms=elapsed_ms)

    # ------------------------------------------------------------------
    # Health waiting
    # ------------------------------------------------------------------

    @staticmethod
    async def _reason_with_journal(reason: str, unit_name: str) -> str:
        """Append journal tail to a failure reason string."""
        tail = await journal_logs(unit_name)
        if tail:
            reason += "\n--- journal tail ---\n" + "\n".join(tail)
        return reason

    async def _wait_for_healthy(
        self,
        service: Service,
        driver: ServiceDriver,
        timeout_s: float = 120.0,
        poll_interval: float = 1.0,
    ) -> None:
        """Poll driver.health_check until it returns True or timeout is exceeded.

        If the systemd unit dies during polling, exits immediately instead of
        waiting for the full timeout.  On any failure, captures the last 20
        journal lines and includes them in the :class:`LifecycleError`.

        Parameters
        ----------
        service:
            The service to check health for.
        driver:
            The driver to use for health checking.
        timeout_s:
            Maximum seconds to wait for health.
        poll_interval:
            Seconds between health check polls.

        Raises
        ------
        LifecycleError
            If the health check does not pass within the timeout, or if the
            process exits before becoming healthy.
        """
        start_time = time.monotonic()
        unit_name = service.unit_name or f"{service.id}.service"

        while True:
            healthy = await driver.health_check(service)
            if healthy:
                return

            # Early exit: if the process is already dead, don't wait
            state = await get_unit_state(unit_name)
            if state in _DEAD_STATES:
                logger.warning(
                    "Service %r died (%s) before becoming healthy",
                    service.id,
                    state,
                )
                reason = await self._reason_with_journal(
                    f"process exited ({state})",
                    unit_name,
                )
                raise LifecycleError(
                    service_id=service.id,
                    operation="start",
                    reason=reason,
                )

            elapsed = time.monotonic() - start_time
            if elapsed + poll_interval > timeout_s:
                logger.warning(
                    "Health check timed out for service %r after %.1fs",
                    service.id,
                    timeout_s,
                )
                reason = await self._reason_with_journal(
                    f"health check timed out after {timeout_s}s",
                    unit_name,
                )
                raise LifecycleError(
                    service_id=service.id,
                    operation="start",
                    reason=reason,
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
