"""Unit tests for HealthMonitor (P7-T3).

Tests are deterministic: instead of sleeping a fixed duration and hoping
the poll loop ran enough iterations, we poll for the expected condition
with a generous timeout. This avoids CI hangs on slow runners.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpumod.models import DriverType, Service
from gpumod.services.base import ServiceDriver
from gpumod.services.health import HealthMonitor, ServiceHealthInfo, _ServiceHealthTask
from gpumod.services.registry import ServiceRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def _make_service(
    id: str = "svc-1",
    name: str = "Test Service",
    port: int = 8080,
    vram_mb: int = 1024,
) -> Service:
    return Service(
        id=id,
        name=name,
        driver=DriverType.VLLM,
        port=port,
        vram_mb=vram_mb,
    )


async def _poll_until(
    condition: object,
    *,
    timeout: float = 5.0,
    interval: float = 0.02,
) -> None:
    """Poll *condition* (a callable returning bool) until truthy or *timeout*."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if callable(condition) and condition():
            return
        await asyncio.sleep(interval)
    msg = f"Condition not met within {timeout}s"
    raise TimeoutError(msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_driver() -> MagicMock:
    driver = MagicMock(spec=ServiceDriver)
    driver.health_check = AsyncMock(return_value=True)
    return driver


@pytest.fixture()
def mock_registry(mock_driver: MagicMock) -> MagicMock:
    registry = MagicMock(spec=ServiceRegistry)
    service = _make_service()
    registry.get = AsyncMock(return_value=service)
    registry.get_driver = MagicMock(return_value=mock_driver)
    return registry


@pytest.fixture()
async def monitor(mock_registry: MagicMock) -> AsyncGenerator[HealthMonitor, None]:
    mon = HealthMonitor(
        registry=mock_registry,
        default_interval=0.05,
        failure_threshold=3,
        recovery_threshold=2,
        check_timeout=0.5,
        min_interval=0.01,
    )
    yield mon
    await mon.stop_all()


# ---------------------------------------------------------------------------
# start_monitoring / stop_monitoring
# ---------------------------------------------------------------------------


class TestStartMonitoring:
    """Tests for HealthMonitor.start_monitoring()."""

    async def test_start_monitoring_creates_task(self, monitor: HealthMonitor) -> None:
        await monitor.start_monitoring("svc-1")
        assert "svc-1" in monitor.monitored_services

    async def test_start_monitoring_idempotent(
        self, monitor: HealthMonitor, mock_driver: MagicMock
    ) -> None:
        await monitor.start_monitoring("svc-1")
        await monitor.start_monitoring("svc-1")
        assert len(monitor.monitored_services) == 1


class TestStopMonitoring:
    """Tests for HealthMonitor.stop_monitoring()."""

    async def test_stop_monitoring_cancels_task(self, monitor: HealthMonitor) -> None:
        await monitor.start_monitoring("svc-1")
        assert "svc-1" in monitor.monitored_services

        await monitor.stop_monitoring("svc-1")
        assert "svc-1" not in monitor.monitored_services

    async def test_stop_monitoring_idempotent(self, monitor: HealthMonitor) -> None:
        await monitor.stop_monitoring("svc-nonexistent")


class TestStopAll:
    """Tests for HealthMonitor.stop_all()."""

    async def test_stop_all_cancels_all_tasks(self, monitor: HealthMonitor) -> None:
        await monitor.start_monitoring("svc-1")
        await monitor.start_monitoring("svc-2")
        assert len(monitor.monitored_services) == 2

        await monitor.stop_all()
        assert len(monitor.monitored_services) == 0


# ---------------------------------------------------------------------------
# Health state transitions (SEC-H5)
# ---------------------------------------------------------------------------


class TestHealthStateTransitions:
    """Tests for failure threshold and recovery logic."""

    async def test_healthy_service_stays_healthy(
        self, monitor: HealthMonitor, mock_driver: MagicMock
    ) -> None:
        callback = AsyncMock()
        monitor._on_state_change = callback
        mock_driver.health_check = AsyncMock(return_value=True)

        await monitor.start_monitoring("svc-1")

        # Wait for a few successful checks to accumulate
        await _poll_until(lambda: mock_driver.health_check.call_count >= 3)

        callback.assert_not_called()

    async def test_single_failure_no_state_change(
        self, monitor: HealthMonitor, mock_driver: MagicMock
    ) -> None:
        callback = AsyncMock()
        monitor._on_state_change = callback

        call_count = 0

        async def _health_check(service: Service) -> bool:
            nonlocal call_count
            call_count += 1
            return call_count != 2

        mock_driver.health_check = AsyncMock(side_effect=_health_check)
        await monitor.start_monitoring("svc-1")

        # Wait until enough checks have run (past the single failure)
        await _poll_until(lambda: call_count >= 5)

        callback.assert_not_called()

    async def test_consecutive_failures_trigger_unhealthy(
        self, monitor: HealthMonitor, mock_driver: MagicMock
    ) -> None:
        callback = AsyncMock()
        monitor._on_state_change = callback
        mock_driver.health_check = AsyncMock(return_value=False)

        await monitor.start_monitoring("svc-1")

        await _poll_until(lambda: callback.called)

        args = callback.call_args_list[0]
        assert args[0][0] == "svc-1"
        assert args[0][1] is False

    async def test_recovery_after_unhealthy(
        self, monitor: HealthMonitor, mock_driver: MagicMock
    ) -> None:
        callback = AsyncMock()
        monitor._on_state_change = callback

        call_count = 0

        async def _health_check(service: Service) -> bool:
            nonlocal call_count
            call_count += 1
            # First 3 calls fail, then all succeed
            return call_count > 3

        mock_driver.health_check = AsyncMock(side_effect=_health_check)
        await monitor.start_monitoring("svc-1")

        # Wait for both unhealthy and recovery callbacks
        await _poll_until(lambda: callback.call_count >= 2)

        unhealthy_call = callback.call_args_list[0]
        assert unhealthy_call[0] == ("svc-1", False)

        recovery_call = callback.call_args_list[1]
        assert recovery_call[0] == ("svc-1", True)


# ---------------------------------------------------------------------------
# Timeout handling (SEC-H4)
# ---------------------------------------------------------------------------


class TestCheckTimeout:
    """Tests for health check timeout."""

    async def test_check_timeout_prevents_hang(
        self, monitor: HealthMonitor, mock_driver: MagicMock
    ) -> None:
        callback = AsyncMock()
        monitor._on_state_change = callback

        async def _slow_health(service: Service) -> bool:
            await asyncio.sleep(60)
            return True

        mock_driver.health_check = AsyncMock(side_effect=_slow_health)
        monitor._check_timeout = 0.05

        await monitor.start_monitoring("svc-1")

        await _poll_until(lambda: callback.called)

        assert callback.call_args_list[0][0] == ("svc-1", False)


# ---------------------------------------------------------------------------
# Min interval enforcement (SEC-H3)
# ---------------------------------------------------------------------------


class TestMinInterval:
    """Tests for minimum poll interval enforcement."""

    def test_min_interval_enforced_on_construction(self) -> None:
        mock_reg = MagicMock(spec=ServiceRegistry)

        with pytest.raises(ValueError, match="[Ii]nterval"):
            HealthMonitor(
                registry=mock_reg,
                default_interval=1.0,
                min_interval=5.0,
            )

    async def test_start_monitoring_rejects_low_interval(self, monitor: HealthMonitor) -> None:
        with pytest.raises(ValueError, match="[Ii]nterval"):
            await monitor.start_monitoring("svc-1", interval=0.001)


# ---------------------------------------------------------------------------
# get_health_status
# ---------------------------------------------------------------------------


class TestGetHealthStatus:
    """Tests for HealthMonitor.get_health_status()."""

    async def test_get_health_status_returns_info(
        self, monitor: HealthMonitor, mock_driver: MagicMock
    ) -> None:
        mock_driver.health_check = AsyncMock(return_value=True)
        await monitor.start_monitoring("svc-1")

        await _poll_until(lambda: mock_driver.health_check.call_count >= 1)

        info = monitor.get_health_status("svc-1")
        assert info is not None
        assert isinstance(info, ServiceHealthInfo)
        assert info.service_id == "svc-1"
        assert info.healthy is True

    async def test_get_health_status_returns_none_for_unmonitored(
        self, monitor: HealthMonitor
    ) -> None:
        info = monitor.get_health_status("svc-nonexistent")
        assert info is None


# ---------------------------------------------------------------------------
# monitored_services property
# ---------------------------------------------------------------------------


class TestMonitoredServices:
    """Tests for HealthMonitor.monitored_services property."""

    async def test_monitored_services_tracks_correctly(self, monitor: HealthMonitor) -> None:
        assert monitor.monitored_services == frozenset()

        await monitor.start_monitoring("svc-1")
        assert monitor.monitored_services == frozenset({"svc-1"})

        await monitor.start_monitoring("svc-2")
        assert monitor.monitored_services == frozenset({"svc-1", "svc-2"})

        await monitor.stop_monitoring("svc-1")
        assert monitor.monitored_services == frozenset({"svc-2"})

        await monitor.stop_all()
        assert monitor.monitored_services == frozenset()


# ---------------------------------------------------------------------------
# Backoff behavior (SEC-H3)
# ---------------------------------------------------------------------------


class TestBackoff:
    """Tests for exponential backoff on failures."""

    def test_backoff_increases_interval(self, monitor: HealthMonitor) -> None:
        entry = _ServiceHealthTask(
            service_id="svc-1",
            interval=0.1,
            failure_threshold=3,
            recovery_threshold=2,
            check_timeout=1.0,
        )

        entry.healthy = True
        entry.consecutive_failures = 0
        healthy_sleep = monitor._compute_sleep(entry)
        assert 0.05 < healthy_sleep < 0.15

        entry.healthy = False
        entry.consecutive_failures = 3
        backoff_3 = monitor._compute_sleep(entry)
        assert backoff_3 > healthy_sleep

        entry.consecutive_failures = 5
        backoff_5 = monitor._compute_sleep(entry)
        assert backoff_5 > backoff_3

    def test_backoff_resets_on_recovery(self, monitor: HealthMonitor) -> None:
        entry = _ServiceHealthTask(
            service_id="svc-1",
            interval=0.1,
            failure_threshold=3,
            recovery_threshold=2,
            check_timeout=1.0,
        )

        entry.healthy = False
        entry.consecutive_failures = 5
        backoff_sleep = monitor._compute_sleep(entry)

        entry.healthy = True
        entry.consecutive_failures = 0
        recovered_sleep = monitor._compute_sleep(entry)

        assert recovered_sleep < backoff_sleep


class TestJitter:
    """Tests for jitter application."""

    def test_jitter_varies_sleep_times(self, monitor: HealthMonitor) -> None:
        entry = _ServiceHealthTask(
            service_id="svc-1",
            interval=10.0,
            failure_threshold=3,
            recovery_threshold=2,
            check_timeout=1.0,
        )
        entry.healthy = True

        durations = [monitor._compute_sleep(entry) for _ in range(20)]

        for d in durations:
            assert 7.5 < d < 12.5

        unique = {round(d, 4) for d in durations}
        assert len(unique) > 1


# ---------------------------------------------------------------------------
# ServiceManager integration
# ---------------------------------------------------------------------------


class TestServiceManagerIntegration:
    """Tests for HealthMonitor integration with ServiceManager."""

    async def test_service_manager_accepts_health_param(self) -> None:
        from gpumod.services.manager import ServiceManager

        mock_db = MagicMock()
        mock_registry = MagicMock(spec=ServiceRegistry)
        mock_lifecycle = MagicMock()
        mock_vram = MagicMock()
        mock_sleep = MagicMock()
        mock_health = MagicMock(spec=HealthMonitor)

        mgr = ServiceManager(
            db=mock_db,
            registry=mock_registry,
            lifecycle=mock_lifecycle,
            vram=mock_vram,
            sleep=mock_sleep,
            health=mock_health,
        )

        assert mgr._health is mock_health

    async def test_service_manager_creates_default_health(self) -> None:
        from gpumod.services.manager import ServiceManager

        mock_db = MagicMock()
        mock_registry = MagicMock(spec=ServiceRegistry)
        mock_lifecycle = MagicMock()
        mock_vram = MagicMock()
        mock_sleep = MagicMock()

        mgr = ServiceManager(
            db=mock_db,
            registry=mock_registry,
            lifecycle=mock_lifecycle,
            vram=mock_vram,
            sleep=mock_sleep,
        )

        assert isinstance(mgr._health, HealthMonitor)
