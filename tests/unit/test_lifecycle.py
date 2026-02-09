"""Tests for gpumod.services.lifecycle — LifecycleManager."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gpumod.models import DriverType, Service, ServiceState, ServiceStatus, SleepMode
from gpumod.services.lifecycle import LifecycleError, LifecycleManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(
    id: str,
    name: str = "Test Service",
    driver: DriverType = DriverType.VLLM,
    port: int = 8000,
    vram_mb: int = 2500,
    depends_on: list[str] | None = None,
) -> Service:
    return Service(
        id=id,
        name=name,
        driver=driver,
        port=port,
        vram_mb=vram_mb,
        sleep_mode=SleepMode.NONE,
        health_endpoint="/health",
        model_id="org/model",
        unit_name=f"{id}.service",
        depends_on=depends_on or [],
        startup_timeout=60,
        extra_config={},
    )


# Pre-configured services for dependency chain: a -> b -> c
SVC_A = _make_service(id="svc-a", name="Service A")
SVC_B = _make_service(id="svc-b", name="Service B", depends_on=["svc-a"])
SVC_C = _make_service(id="svc-c", name="Service C", depends_on=["svc-b"])

ALL_SERVICES = [SVC_A, SVC_B, SVC_C]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_mock_registry(
    services: dict[str, Service] | None = None,
) -> AsyncMock:
    """Build a mock ServiceRegistry with pre-configured services and drivers."""
    registry = AsyncMock()

    svc_map = services or {s.id: s for s in ALL_SERVICES}

    async def _get(service_id: str) -> Service:
        if service_id not in svc_map:
            raise KeyError(f"Service not found: {service_id!r}")
        return svc_map[service_id]

    registry.get = AsyncMock(side_effect=_get)

    async def _list_all() -> list[Service]:
        return list(svc_map.values())

    registry.list_all = AsyncMock(side_effect=_list_all)

    async def _get_dependents(service_id: str) -> list[Service]:
        return [s for s in svc_map.values() if service_id in s.depends_on]

    registry.get_dependents = AsyncMock(side_effect=_get_dependents)

    return registry


def _build_mock_driver(
    healthy: bool = True,
    state: ServiceState = ServiceState.STOPPED,
) -> AsyncMock:
    """Build a mock ServiceDriver with configurable health and state."""
    driver = AsyncMock()
    driver.start = AsyncMock()
    driver.stop = AsyncMock()
    driver.health_check = AsyncMock(return_value=healthy)
    driver.status = AsyncMock(return_value=ServiceStatus(state=state))
    return driver


@pytest.fixture
def mock_registry() -> AsyncMock:
    """A mock ServiceRegistry with svc-a, svc-b, svc-c configured."""
    return _build_mock_registry()


@pytest.fixture
def mock_driver() -> AsyncMock:
    """A mock ServiceDriver that starts healthy and reports STOPPED state."""
    return _build_mock_driver()


@pytest.fixture
def lifecycle(mock_registry: AsyncMock, mock_driver: AsyncMock) -> LifecycleManager:
    """A LifecycleManager wired to mock registry and driver."""
    mock_registry.get_driver = lambda dtype: mock_driver
    return LifecycleManager(mock_registry)


# ---------------------------------------------------------------------------
# Test: start with no deps
# ---------------------------------------------------------------------------


class TestStartNoDeps:
    """Test start(svc-a) with no dependencies."""

    async def test_start_calls_driver_start_and_waits_for_health(
        self, lifecycle: LifecycleManager, mock_driver: AsyncMock
    ) -> None:
        """start(svc-a) should call driver.start then wait for health check."""
        await lifecycle.start("svc-a")

        # driver.start was called with svc-a
        mock_driver.start.assert_called_once_with(SVC_A)
        # health_check was called at least once
        mock_driver.health_check.assert_called()


# ---------------------------------------------------------------------------
# Test: start with chain
# ---------------------------------------------------------------------------


class TestStartWithChain:
    """Test start(svc-c) with dependency chain a -> b -> c."""

    async def test_start_starts_dependencies_in_order(
        self, lifecycle: LifecycleManager, mock_driver: AsyncMock
    ) -> None:
        """start(svc-c) should start svc-a first, then svc-b, then svc-c."""
        await lifecycle.start("svc-c")

        # All three should be started
        started_services = [c.args[0].id for c in mock_driver.start.call_args_list]
        assert started_services == ["svc-a", "svc-b", "svc-c"]


# ---------------------------------------------------------------------------
# Test: start skips already-running deps
# ---------------------------------------------------------------------------


class TestStartSkipsRunning:
    """Test that start() skips already-running dependencies."""

    async def test_start_skips_already_running_dependency(
        self,
        mock_registry: AsyncMock,
    ) -> None:
        """start(svc-c) should not restart svc-a if it is already running."""
        driver = _build_mock_driver(healthy=True, state=ServiceState.STOPPED)

        # svc-a is already running — status returns RUNNING
        async def _status_side_effect(svc: Service) -> ServiceStatus:
            if svc.id == "svc-a":
                return ServiceStatus(state=ServiceState.RUNNING)
            return ServiceStatus(state=ServiceState.STOPPED)

        driver.status = AsyncMock(side_effect=_status_side_effect)
        mock_registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(mock_registry)
        await lm.start("svc-c")

        # svc-a should NOT be started; only svc-b and svc-c
        started_services = [c.args[0].id for c in driver.start.call_args_list]
        assert "svc-a" not in started_services
        assert started_services == ["svc-b", "svc-c"]


# ---------------------------------------------------------------------------
# Test: stop with dependents
# ---------------------------------------------------------------------------


class TestStopWithDependents:
    """Test stop(svc-a) with dependents: stops svc-c, then svc-b, then svc-a."""

    async def test_stop_stops_dependents_in_reverse_order(
        self,
        mock_registry: AsyncMock,
    ) -> None:
        """stop(svc-a) should stop svc-c first, then svc-b, then svc-a."""
        driver = _build_mock_driver(state=ServiceState.RUNNING)
        mock_registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(mock_registry)
        await lm.stop("svc-a")

        stopped_services = [c.args[0].id for c in driver.stop.call_args_list]
        assert stopped_services == ["svc-c", "svc-b", "svc-a"]


# ---------------------------------------------------------------------------
# Test: stop skips already-stopped dependents
# ---------------------------------------------------------------------------


class TestStopSkipsStopped:
    """Test that stop() skips already-stopped dependents."""

    async def test_stop_skips_already_stopped_dependents(
        self,
        mock_registry: AsyncMock,
    ) -> None:
        """stop(svc-a) should skip svc-c if it's already stopped."""
        driver = _build_mock_driver(state=ServiceState.RUNNING)

        async def _status_side_effect(svc: Service) -> ServiceStatus:
            if svc.id == "svc-c":
                return ServiceStatus(state=ServiceState.STOPPED)
            return ServiceStatus(state=ServiceState.RUNNING)

        driver.status = AsyncMock(side_effect=_status_side_effect)
        mock_registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(mock_registry)
        await lm.stop("svc-a")

        stopped_services = [c.args[0].id for c in driver.stop.call_args_list]
        assert "svc-c" not in stopped_services
        assert stopped_services == ["svc-b", "svc-a"]


# ---------------------------------------------------------------------------
# Test: restart
# ---------------------------------------------------------------------------


class TestRestart:
    """Test restart() calls stop then start in order."""

    async def test_restart_calls_stop_then_start(
        self,
        mock_registry: AsyncMock,
    ) -> None:
        """restart(svc-b) should stop dependents, stop svc-b, then start deps and svc-b."""
        # Track state transitions: services start RUNNING, become STOPPED after stop()
        stopped_ids: set[str] = set()
        driver = _build_mock_driver(state=ServiceState.RUNNING)

        async def _stop_side_effect(svc: Service) -> None:
            stopped_ids.add(svc.id)

        async def _status_side_effect(svc: Service) -> ServiceStatus:
            if svc.id in stopped_ids:
                return ServiceStatus(state=ServiceState.STOPPED)
            return ServiceStatus(state=ServiceState.RUNNING)

        driver.stop = AsyncMock(side_effect=_stop_side_effect)
        driver.status = AsyncMock(side_effect=_status_side_effect)
        mock_registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(mock_registry)
        await lm.restart("svc-b")

        # stop should happen before start — collect all calls in order
        stop_ids_list = [c.args[0].id for c in driver.stop.call_args_list]
        start_ids = [c.args[0].id for c in driver.start.call_args_list]

        # svc-c depends on svc-b, so stop order: svc-c, svc-b
        assert stop_ids_list == ["svc-c", "svc-b"]
        # start resolves deps for svc-b: svc-a (RUNNING, skip), svc-b (STOPPED, start)
        assert "svc-b" in start_ids


# ---------------------------------------------------------------------------
# Test: _wait_for_healthy — immediate success
# ---------------------------------------------------------------------------


class TestWaitForHealthyImmediate:
    """Test _wait_for_healthy returns immediately when health passes first try."""

    async def test_returns_immediately_on_first_healthy(
        self, lifecycle: LifecycleManager, mock_driver: AsyncMock
    ) -> None:
        """_wait_for_healthy should return immediately if health_check passes."""
        mock_driver.health_check.return_value = True

        await lifecycle._wait_for_healthy(SVC_A, mock_driver, timeout_s=10.0, poll_interval=0.01)

        mock_driver.health_check.assert_called_once_with(SVC_A)


# ---------------------------------------------------------------------------
# Test: _wait_for_healthy — retries then succeeds
# ---------------------------------------------------------------------------


class TestWaitForHealthyRetry:
    """Test _wait_for_healthy retries on initial failure, succeeds on 3rd try."""

    async def test_retries_and_succeeds_on_third_attempt(
        self, lifecycle: LifecycleManager, mock_driver: AsyncMock
    ) -> None:
        """_wait_for_healthy should retry and succeed on the third health check."""
        mock_driver.health_check.side_effect = [False, False, True]

        await lifecycle._wait_for_healthy(SVC_A, mock_driver, timeout_s=10.0, poll_interval=0.01)

        assert mock_driver.health_check.call_count == 3


# ---------------------------------------------------------------------------
# Test: _wait_for_healthy — timeout raises LifecycleError
# ---------------------------------------------------------------------------


class TestWaitForHealthyTimeout:
    """Test _wait_for_healthy raises LifecycleError on timeout."""

    async def test_raises_lifecycle_error_on_timeout(
        self, lifecycle: LifecycleManager, mock_driver: AsyncMock
    ) -> None:
        """_wait_for_healthy should raise LifecycleError when health never passes."""
        mock_driver.health_check.return_value = False

        with pytest.raises(LifecycleError, match="svc-a"):
            await lifecycle._wait_for_healthy(
                SVC_A, mock_driver, timeout_s=0.05, poll_interval=0.01
            )


# ---------------------------------------------------------------------------
# Test: LifecycleError message
# ---------------------------------------------------------------------------


class TestLifecycleError:
    """Test LifecycleError message includes service_id, operation, reason."""

    def test_error_message_includes_all_fields(self) -> None:
        """LifecycleError message should include service_id, operation, and reason."""
        err = LifecycleError(
            service_id="svc-a",
            operation="start",
            reason="health check timed out",
        )

        msg = str(err)
        assert "svc-a" in msg
        assert "start" in msg
        assert "health check timed out" in msg

    def test_error_attributes_accessible(self) -> None:
        """LifecycleError should expose service_id, operation, and reason attributes."""
        err = LifecycleError(
            service_id="svc-b",
            operation="stop",
            reason="process did not exit",
        )

        assert err.service_id == "svc-b"
        assert err.operation == "stop"
        assert err.reason == "process did not exit"


# ---------------------------------------------------------------------------
# Test: start calls wake() for router-mode services
# ---------------------------------------------------------------------------


class TestStartWakesRouterService:
    """Test that start() calls driver.wake() for router-mode services."""

    async def test_start_calls_wake_for_router_sleep_mode(self) -> None:
        """Router services should have wake() called after health check."""
        router_svc = Service(
            id="router-svc",
            name="Router Service",
            driver=DriverType.LLAMACPP,
            port=7070,
            vram_mb=20000,
            sleep_mode=SleepMode.ROUTER,
            model_id="org/model",
            unit_name="router-svc.service",
            extra_config={},
        )
        registry = _build_mock_registry(services={"router-svc": router_svc})
        driver = _build_mock_driver(healthy=True, state=ServiceState.STOPPED)
        driver.supports_sleep = True
        driver.wake = AsyncMock()
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)
        await lm.start("router-svc")

        driver.start.assert_called_once_with(router_svc)
        driver.wake.assert_awaited_once_with(router_svc)

    async def test_start_does_not_call_wake_for_non_router(self) -> None:
        """Non-router services should NOT have wake() called."""
        registry = _build_mock_registry()
        driver = _build_mock_driver(healthy=True, state=ServiceState.STOPPED)
        driver.supports_sleep = False
        driver.wake = AsyncMock()
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)
        await lm.start("svc-a")

        driver.start.assert_called_once_with(SVC_A)
        driver.wake.assert_not_awaited()

    async def test_start_does_not_wake_already_running_router(self) -> None:
        """Already-running router services should be skipped entirely."""
        router_svc = Service(
            id="router-svc",
            name="Router Service",
            driver=DriverType.LLAMACPP,
            port=7070,
            vram_mb=20000,
            sleep_mode=SleepMode.ROUTER,
            model_id="org/model",
            unit_name="router-svc.service",
            extra_config={},
        )
        registry = _build_mock_registry(services={"router-svc": router_svc})
        driver = _build_mock_driver(state=ServiceState.RUNNING)
        driver.supports_sleep = True
        driver.wake = AsyncMock()
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)
        await lm.start("router-svc")

        driver.start.assert_not_called()
        driver.wake.assert_not_awaited()
