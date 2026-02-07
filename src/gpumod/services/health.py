"""HealthMonitor — continuous health checking for registered services.

Single Responsibility: monitors health and reports state changes.
Does NOT manage lifecycle (start/stop) — that is LifecycleManager's job.

Security controls:
  SEC-H3: Minimum poll interval enforced.
  SEC-H4: Per-check timeout via asyncio.wait_for.
  SEC-H5: Consecutive-failure threshold before state change.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from gpumod.services.registry import ServiceRegistry

logger = logging.getLogger(__name__)

_MAX_BACKOFF: float = 120.0


@dataclass(frozen=True)
class ServiceHealthInfo:
    """Snapshot of a service's health monitoring state."""

    service_id: str
    healthy: bool
    consecutive_failures: int
    consecutive_successes: int
    last_check_at: float
    last_healthy_at: float | None
    last_unhealthy_at: float | None


class _ServiceHealthTask:
    """Internal mutable state for a single service's monitoring task."""

    __slots__ = (
        "service_id",
        "interval",
        "failure_threshold",
        "recovery_threshold",
        "check_timeout",
        "task",
        "consecutive_failures",
        "consecutive_successes",
        "healthy",
        "last_check_at",
        "last_healthy_at",
        "last_unhealthy_at",
    )

    def __init__(
        self,
        service_id: str,
        interval: float,
        failure_threshold: int,
        recovery_threshold: int,
        check_timeout: float,
    ) -> None:
        self.service_id = service_id
        self.interval = interval
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold
        self.check_timeout = check_timeout
        self.task: asyncio.Task[None] | None = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.healthy = True
        self.last_check_at = 0.0
        self.last_healthy_at: float | None = None
        self.last_unhealthy_at: float | None = None

    def to_info(self) -> ServiceHealthInfo:
        """Create an immutable snapshot of the current state."""
        return ServiceHealthInfo(
            service_id=self.service_id,
            healthy=self.healthy,
            consecutive_failures=self.consecutive_failures,
            consecutive_successes=self.consecutive_successes,
            last_check_at=self.last_check_at,
            last_healthy_at=self.last_healthy_at,
            last_unhealthy_at=self.last_unhealthy_at,
        )


class HealthMonitor:
    """Continuous health monitoring for registered services.

    Parameters
    ----------
    registry:
        ServiceRegistry for looking up services and drivers.
    on_state_change:
        Callback invoked when a service's health state changes.
    default_interval:
        Default polling interval in seconds.
    failure_threshold:
        Consecutive failures before declaring unhealthy.
    recovery_threshold:
        Consecutive successes before declaring healthy again.
    check_timeout:
        Per-health-check timeout in seconds.
    min_interval:
        Minimum allowed polling interval (SEC-H3).
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        on_state_change: Callable[[str, bool], Awaitable[None]] | None = None,
        default_interval: float = 15.0,
        failure_threshold: int = 3,
        recovery_threshold: int = 2,
        check_timeout: float = 10.0,
        min_interval: float = 5.0,
    ) -> None:
        if default_interval < min_interval:
            msg = f"default_interval ({default_interval}s) is below min_interval ({min_interval}s)"
            raise ValueError(msg)

        self._registry = registry
        self._on_state_change = on_state_change
        self._default_interval = default_interval
        self._failure_threshold = failure_threshold
        self._recovery_threshold = recovery_threshold
        self._check_timeout = check_timeout
        self._min_interval = min_interval
        self._tasks: dict[str, _ServiceHealthTask] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_monitoring(
        self,
        service_id: str,
        interval: float | None = None,
    ) -> None:
        """Begin health monitoring for a service. Idempotent."""
        if service_id in self._tasks:
            return

        resolved_interval = interval if interval is not None else self._default_interval
        if resolved_interval < self._min_interval:
            msg = f"Interval ({resolved_interval}s) is below min_interval ({self._min_interval}s)"
            raise ValueError(msg)

        entry = _ServiceHealthTask(
            service_id=service_id,
            interval=resolved_interval,
            failure_threshold=self._failure_threshold,
            recovery_threshold=self._recovery_threshold,
            check_timeout=self._check_timeout,
        )
        entry.task = asyncio.create_task(self._poll_loop(entry))
        self._tasks[service_id] = entry
        logger.info("Started monitoring %r (interval=%.1fs)", service_id, resolved_interval)

    async def stop_monitoring(self, service_id: str) -> None:
        """Stop health monitoring for a service. Idempotent."""
        entry = self._tasks.pop(service_id, None)
        if entry is None:
            return
        if entry.task is not None and not entry.task.done():
            entry.task.cancel()
            # Yield once so the event loop can deliver the cancellation.
            # We intentionally do NOT ``await entry.task`` because on
            # Python 3.11 the cancelled task may never complete due to
            # an asyncio.wait_for race condition (fixed in 3.12).
            await asyncio.sleep(0)
        logger.info("Stopped monitoring %r", service_id)

    async def stop_all(self) -> None:
        """Stop monitoring all services and cancel all tasks."""
        service_ids = list(self._tasks.keys())
        for service_id in service_ids:
            await self.stop_monitoring(service_id)

    def get_health_status(self, service_id: str) -> ServiceHealthInfo | None:
        """Get the current health status of a monitored service."""
        entry = self._tasks.get(service_id)
        if entry is None:
            return None
        return entry.to_info()

    @property
    def monitored_services(self) -> frozenset[str]:
        """Set of service IDs currently being monitored."""
        return frozenset(self._tasks.keys())

    # ------------------------------------------------------------------
    # Internal polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self, entry: _ServiceHealthTask) -> None:
        """Per-service polling loop. Runs as an asyncio.Task."""
        try:
            while True:
                await self._check_once(entry)
                sleep_time = self._compute_sleep(entry)
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            return

    async def _check_once(self, entry: _ServiceHealthTask) -> None:
        """Perform a single health check and update state."""
        service = await self._registry.get(entry.service_id)
        driver = self._registry.get_driver(service.driver)

        try:
            healthy = await asyncio.wait_for(
                driver.health_check(service),
                timeout=entry.check_timeout,
            )
        except (TimeoutError, Exception):  # noqa: BLE001
            healthy = False

        now = time.monotonic()
        entry.last_check_at = now

        if healthy:
            entry.consecutive_failures = 0
            entry.consecutive_successes += 1
            entry.last_healthy_at = now

            if not entry.healthy and entry.consecutive_successes >= entry.recovery_threshold:
                entry.healthy = True
                logger.info("Service %r recovered", entry.service_id)
                await self._fire_callback(entry.service_id, healthy=True)
        else:
            entry.consecutive_successes = 0
            entry.consecutive_failures += 1
            entry.last_unhealthy_at = now

            if entry.healthy and entry.consecutive_failures >= entry.failure_threshold:
                entry.healthy = False
                logger.warning(
                    "Service %r unhealthy (%d consecutive failures)",
                    entry.service_id,
                    entry.consecutive_failures,
                )
                await self._fire_callback(entry.service_id, healthy=False)

    def _compute_sleep(self, entry: _ServiceHealthTask) -> float:
        """Compute the next sleep interval with jitter and backoff."""
        base = entry.interval

        if not entry.healthy:
            backoff_exponent = min(entry.consecutive_failures, 10)
            base = min(entry.interval * (2**backoff_exponent), _MAX_BACKOFF)

        jitter = 0.8 + random.random() * 0.4  # noqa: S311
        return base * jitter

    async def _fire_callback(self, service_id: str, *, healthy: bool) -> None:
        """Invoke the on_state_change callback if set."""
        if self._on_state_change is not None:
            try:
                await self._on_state_change(service_id, healthy)
            except Exception:
                logger.exception(
                    "Error in on_state_change callback for %r",
                    service_id,
                )
