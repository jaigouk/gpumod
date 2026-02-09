"""Tests for gpumod.services.lifecycle — LifecycleManager."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

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
# Test: _wait_for_healthy uses service.startup_timeout
# ---------------------------------------------------------------------------


class TestWaitForHealthyUsesServiceTimeout:
    """Test that _wait_for_healthy uses the service's startup_timeout field."""

    @patch("gpumod.services.lifecycle.journal_logs", return_value=[])
    @patch("gpumod.services.lifecycle.get_unit_state", return_value="activating")
    async def test_uses_service_startup_timeout(
        self,
        mock_state: AsyncMock,
        mock_journal: AsyncMock,
    ) -> None:
        """_wait_for_healthy should use service.startup_timeout instead of hardcoded default."""
        # Create service with a 1-second timeout (much shorter than 120s default)
        svc = Service(
            id="short-timeout-svc",
            name="Short Timeout Service",
            driver=DriverType.VLLM,
            port=8000,
            vram_mb=2500,
            startup_timeout=1,  # 1 second - much shorter than 120s default
            model_id="org/model",
            unit_name="short-timeout-svc.service",
        )
        registry = _build_mock_registry(services={"short-timeout-svc": svc})
        driver = _build_mock_driver(healthy=False, state=ServiceState.STOPPED)
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)

        # Should timeout after ~1s based on service's startup_timeout, not 120s default
        # Using poll_interval=0.1s, we expect ~10 health checks max before timeout
        with pytest.raises(LifecycleError, match="timed out"):
            await lm._wait_for_healthy(svc, driver, poll_interval=0.1)

        # Verify it didn't wait the full 120s default - should be ~10 calls for 1s timeout
        assert driver.health_check.call_count <= 15

    @patch("gpumod.services.lifecycle.get_unit_state", return_value="activating")
    async def test_long_service_timeout_allows_more_retries(
        self,
        mock_state: AsyncMock,
    ) -> None:
        """Services with longer startup_timeout should wait longer."""
        # Create service with a 300s timeout (like vLLM pooling models)
        svc = Service(
            id="vllm-embedding",
            name="vLLM Embedding",
            driver=DriverType.VLLM,
            port=8001,
            vram_mb=4000,
            startup_timeout=300,  # 5 minutes - like vLLM pooling
            model_id="BAAI/bge-m3",
            unit_name="vllm-embedding.service",
        )
        registry = _build_mock_registry(services={"vllm-embedding": svc})
        # Health check fails 5 times, then succeeds
        driver = _build_mock_driver(healthy=False, state=ServiceState.STOPPED)
        driver.health_check.side_effect = [False, False, False, False, False, True]
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)

        # Should succeed after 6 health checks (doesn't timeout at 120s)
        await lm._wait_for_healthy(svc, driver, poll_interval=0.01)

        assert driver.health_check.call_count == 6


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

    @patch("gpumod.services.lifecycle.get_unit_state", return_value="activating")
    async def test_retries_and_succeeds_on_third_attempt(
        self,
        mock_state: AsyncMock,
        lifecycle: LifecycleManager,
        mock_driver: AsyncMock,
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

    @patch("gpumod.services.lifecycle.journal_logs", return_value=[])
    @patch("gpumod.services.lifecycle.get_unit_state", return_value="activating")
    async def test_raises_lifecycle_error_on_timeout(
        self,
        mock_state: AsyncMock,
        mock_journal: AsyncMock,
        lifecycle: LifecycleManager,
        mock_driver: AsyncMock,
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


# ---------------------------------------------------------------------------
# Test: _wait_for_healthy — early exit on process death (AC2)
# ---------------------------------------------------------------------------


class TestWaitForHealthyEarlyExit:
    """Test that _wait_for_healthy exits immediately when the process dies."""

    @patch("gpumod.services.lifecycle.get_unit_state", return_value="failed")
    async def test_exits_immediately_on_failed_state(
        self,
        mock_state: AsyncMock,
    ) -> None:
        """If systemd reports 'failed', stop polling immediately."""
        registry = _build_mock_registry()
        driver = _build_mock_driver(healthy=False, state=ServiceState.STOPPED)
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)

        with pytest.raises(LifecycleError, match="process exited"):
            await lm._wait_for_healthy(
                SVC_A,
                driver,
                timeout_s=60.0,
                poll_interval=0.01,
            )

        # Should NOT have polled for the full timeout — only a few attempts
        assert driver.health_check.call_count < 10

    @patch("gpumod.services.lifecycle.get_unit_state", return_value="inactive")
    async def test_exits_immediately_on_inactive_state(
        self,
        mock_state: AsyncMock,
    ) -> None:
        """If systemd reports 'inactive', stop polling immediately."""
        registry = _build_mock_registry()
        driver = _build_mock_driver(healthy=False, state=ServiceState.STOPPED)
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)

        with pytest.raises(LifecycleError, match="process exited"):
            await lm._wait_for_healthy(
                SVC_A,
                driver,
                timeout_s=60.0,
                poll_interval=0.01,
            )

    @patch("gpumod.services.lifecycle.get_unit_state", return_value="activating")
    async def test_keeps_polling_while_activating(
        self,
        mock_state: AsyncMock,
    ) -> None:
        """If systemd reports 'activating', keep polling (process still starting)."""
        registry = _build_mock_registry()
        # Health check fails twice, then succeeds
        driver = _build_mock_driver(healthy=False, state=ServiceState.STOPPED)
        driver.health_check.side_effect = [False, False, True]
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)

        # Should succeed on 3rd health check, not raise
        await lm._wait_for_healthy(
            SVC_A,
            driver,
            timeout_s=10.0,
            poll_interval=0.01,
        )

        assert driver.health_check.call_count == 3


# ---------------------------------------------------------------------------
# Test: _wait_for_healthy — journal tail included in error (AC3)
# ---------------------------------------------------------------------------


class TestWaitForHealthyJournalInError:
    """Test that LifecycleError includes journal log tail on failure."""

    @patch("gpumod.services.lifecycle.journal_logs")
    @patch("gpumod.services.lifecycle.get_unit_state", return_value="failed")
    async def test_error_contains_journal_lines_on_process_death(
        self,
        mock_state: AsyncMock,
        mock_journal: AsyncMock,
    ) -> None:
        """When process dies, LifecycleError.reason includes journal output."""
        mock_journal.return_value = [
            "AttributeError: 'MistralTokenizer' has no attribute 'all_special_ids'",
            "Main process exited, code=exited, status=1/FAILURE",
        ]

        registry = _build_mock_registry()
        driver = _build_mock_driver(healthy=False, state=ServiceState.STOPPED)
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)

        with pytest.raises(LifecycleError) as exc_info:
            await lm._wait_for_healthy(
                SVC_A,
                driver,
                timeout_s=60.0,
                poll_interval=0.01,
            )

        reason = exc_info.value.reason
        assert "MistralTokenizer" in reason
        assert "FAILURE" in reason

    @patch("gpumod.services.lifecycle.journal_logs")
    @patch("gpumod.services.lifecycle.get_unit_state", return_value="activating")
    async def test_error_contains_journal_lines_on_timeout(
        self,
        mock_state: AsyncMock,
        mock_journal: AsyncMock,
    ) -> None:
        """When health check times out, LifecycleError.reason includes journal output."""
        mock_journal.return_value = [
            "No available memory for the cache blocks",
        ]

        registry = _build_mock_registry()
        driver = _build_mock_driver(healthy=False, state=ServiceState.STOPPED)
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)

        with pytest.raises(LifecycleError) as exc_info:
            await lm._wait_for_healthy(
                SVC_A,
                driver,
                timeout_s=0.05,
                poll_interval=0.01,
            )

        assert "No available memory" in exc_info.value.reason

    @patch("gpumod.services.lifecycle.journal_logs")
    @patch("gpumod.services.lifecycle.get_unit_state", return_value="failed")
    async def test_error_graceful_when_journal_empty(
        self,
        mock_state: AsyncMock,
        mock_journal: AsyncMock,
    ) -> None:
        """When journal returns no lines, error still works cleanly."""
        mock_journal.return_value = []

        registry = _build_mock_registry()
        driver = _build_mock_driver(healthy=False, state=ServiceState.STOPPED)
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)

        with pytest.raises(LifecycleError) as exc_info:
            await lm._wait_for_healthy(
                SVC_A,
                driver,
                timeout_s=60.0,
                poll_interval=0.01,
            )

        # Should still have a valid reason, just without journal content
        assert exc_info.value.reason
        assert "process exited" in exc_info.value.reason

    @patch("gpumod.services.lifecycle.journal_logs")
    @patch("gpumod.services.lifecycle.get_unit_state", return_value="failed")
    async def test_journal_called_with_unit_name(
        self,
        mock_state: AsyncMock,
        mock_journal: AsyncMock,
    ) -> None:
        """journal_logs() is called with the service's unit_name."""
        mock_journal.return_value = ["some log"]

        registry = _build_mock_registry()
        driver = _build_mock_driver(healthy=False, state=ServiceState.STOPPED)
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)

        with pytest.raises(LifecycleError):
            await lm._wait_for_healthy(
                SVC_A,
                driver,
                timeout_s=60.0,
                poll_interval=0.01,
            )

        mock_journal.assert_called_once_with("svc-a.service")


# ---------------------------------------------------------------------------
# Test: SleepResult / WakeResult dataclasses
# ---------------------------------------------------------------------------


class TestSleepResult:
    """Test SleepResult dataclass."""

    def test_sleep_result_success(self) -> None:
        """SleepResult should have success, skipped, reason, latency_ms fields."""
        from gpumod.services.lifecycle import SleepResult

        result = SleepResult(success=True, latency_ms=42.5)
        assert result.success is True
        assert result.skipped is False
        assert result.reason is None
        assert result.latency_ms == 42.5

    def test_sleep_result_skipped(self) -> None:
        """SleepResult can indicate a skipped operation."""
        from gpumod.services.lifecycle import SleepResult

        result = SleepResult(success=False, skipped=True, reason="driver does not support sleep")
        assert result.success is False
        assert result.skipped is True
        assert result.reason == "driver does not support sleep"


class TestWakeResult:
    """Test WakeResult dataclass."""

    def test_wake_result_success(self) -> None:
        """WakeResult should have success, skipped, reason, latency_ms fields."""
        from gpumod.services.lifecycle import WakeResult

        result = WakeResult(success=True, latency_ms=15.0)
        assert result.success is True
        assert result.skipped is False
        assert result.reason is None
        assert result.latency_ms == 15.0

    def test_wake_result_skipped(self) -> None:
        """WakeResult can indicate a skipped operation."""
        from gpumod.services.lifecycle import WakeResult

        result = WakeResult(success=False, skipped=True, reason="service not sleeping")
        assert result.success is False
        assert result.skipped is True
        assert result.reason == "service not sleeping"


# ---------------------------------------------------------------------------
# Test: LifecycleManager.sleep()
# ---------------------------------------------------------------------------


class TestLifecycleManagerSleep:
    """Tests for LifecycleManager.sleep() method."""

    async def test_sleep_delegates_to_driver(self) -> None:
        """sleep() should call driver.sleep() for sleep-capable services."""
        from gpumod.services.lifecycle import SleepResult

        svc = _make_service(id="vllm-chat")
        registry = _build_mock_registry(services={"vllm-chat": svc})
        driver = _build_mock_driver(state=ServiceState.RUNNING)
        driver.supports_sleep = True
        driver.sleep = AsyncMock()
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)
        result = await lm.sleep("vllm-chat", level=1)

        assert isinstance(result, SleepResult)
        assert result.success is True
        driver.sleep.assert_awaited_once_with(svc, 1)

    async def test_sleep_returns_skipped_for_non_capable_driver(self) -> None:
        """sleep() should return skipped=True for services with supports_sleep=False."""
        from gpumod.services.lifecycle import SleepResult

        svc = _make_service(id="embedding-svc")
        registry = _build_mock_registry(services={"embedding-svc": svc})
        driver = _build_mock_driver(state=ServiceState.RUNNING)
        driver.supports_sleep = False
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)
        result = await lm.sleep("embedding-svc", level=1)

        assert isinstance(result, SleepResult)
        assert result.success is False
        assert result.skipped is True
        assert "does not support sleep" in (result.reason or "")

    async def test_sleep_returns_skipped_if_already_sleeping(self) -> None:
        """sleep() should be idempotent - skip if already sleeping."""
        from gpumod.services.lifecycle import SleepResult

        svc = _make_service(id="vllm-chat")
        registry = _build_mock_registry(services={"vllm-chat": svc})
        driver = _build_mock_driver(state=ServiceState.SLEEPING)
        driver.supports_sleep = True
        driver.sleep = AsyncMock()
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)
        result = await lm.sleep("vllm-chat", level=1)

        assert isinstance(result, SleepResult)
        assert result.success is True  # Idempotent success
        assert result.skipped is True
        driver.sleep.assert_not_awaited()

    async def test_sleep_returns_error_if_not_running(self) -> None:
        """sleep() should return error if service is not RUNNING or SLEEPING."""
        from gpumod.services.lifecycle import SleepResult

        svc = _make_service(id="vllm-chat")
        registry = _build_mock_registry(services={"vllm-chat": svc})
        driver = _build_mock_driver(state=ServiceState.STOPPED)
        driver.supports_sleep = True
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)
        result = await lm.sleep("vllm-chat", level=1)

        assert isinstance(result, SleepResult)
        assert result.success is False
        assert result.skipped is False
        reason_lower = (result.reason or "").lower()
        assert "stopped" in reason_lower or "not running" in reason_lower

    async def test_sleep_returns_error_for_unknown_service(self) -> None:
        """sleep() should return error for non-existent service."""
        from gpumod.services.lifecycle import SleepResult

        registry = _build_mock_registry(services={})
        lm = LifecycleManager(registry)
        result = await lm.sleep("unknown-svc", level=1)

        assert isinstance(result, SleepResult)
        assert result.success is False
        assert "not found" in (result.reason or "").lower()

    async def test_sleep_includes_latency_ms(self) -> None:
        """sleep() result should include latency in milliseconds."""
        from gpumod.services.lifecycle import SleepResult

        svc = _make_service(id="vllm-chat")
        registry = _build_mock_registry(services={"vllm-chat": svc})
        driver = _build_mock_driver(state=ServiceState.RUNNING)
        driver.supports_sleep = True
        driver.sleep = AsyncMock()
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)
        result = await lm.sleep("vllm-chat", level=1)

        assert isinstance(result, SleepResult)
        assert result.latency_ms is not None
        assert result.latency_ms >= 0

    async def test_sleep_passes_level_to_driver(self) -> None:
        """sleep(level=2) should pass level=2 to driver."""
        svc = _make_service(id="vllm-chat")
        registry = _build_mock_registry(services={"vllm-chat": svc})
        driver = _build_mock_driver(state=ServiceState.RUNNING)
        driver.supports_sleep = True
        driver.sleep = AsyncMock()
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)
        await lm.sleep("vllm-chat", level=2)

        driver.sleep.assert_awaited_once_with(svc, 2)


# ---------------------------------------------------------------------------
# Test: LifecycleManager.wake()
# ---------------------------------------------------------------------------


class TestLifecycleManagerWake:
    """Tests for LifecycleManager.wake() method."""

    async def test_wake_delegates_to_driver(self) -> None:
        """wake() should call driver.wake() for sleeping services."""
        from gpumod.services.lifecycle import WakeResult

        svc = _make_service(id="vllm-chat")
        registry = _build_mock_registry(services={"vllm-chat": svc})
        driver = _build_mock_driver(state=ServiceState.SLEEPING)
        driver.supports_sleep = True
        driver.wake = AsyncMock()
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)
        result = await lm.wake("vllm-chat")

        assert isinstance(result, WakeResult)
        assert result.success is True
        driver.wake.assert_awaited_once_with(svc)

    async def test_wake_returns_skipped_if_not_sleeping(self) -> None:
        """wake() should return skipped=True if service is not sleeping."""
        from gpumod.services.lifecycle import WakeResult

        svc = _make_service(id="vllm-chat")
        registry = _build_mock_registry(services={"vllm-chat": svc})
        driver = _build_mock_driver(state=ServiceState.RUNNING)
        driver.supports_sleep = True
        driver.wake = AsyncMock()
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)
        result = await lm.wake("vllm-chat")

        assert isinstance(result, WakeResult)
        assert result.success is True  # Idempotent success
        assert result.skipped is True
        driver.wake.assert_not_awaited()

    async def test_wake_returns_error_for_non_capable_driver(self) -> None:
        """wake() should return error for drivers that don't support sleep."""
        from gpumod.services.lifecycle import WakeResult

        svc = _make_service(id="embedding-svc")
        registry = _build_mock_registry(services={"embedding-svc": svc})
        driver = _build_mock_driver(state=ServiceState.SLEEPING)
        driver.supports_sleep = False
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)
        result = await lm.wake("embedding-svc")

        assert isinstance(result, WakeResult)
        assert result.success is False
        assert "does not support" in (result.reason or "").lower()

    async def test_wake_returns_error_for_stopped_service(self) -> None:
        """wake() should return error if service is STOPPED."""
        from gpumod.services.lifecycle import WakeResult

        svc = _make_service(id="vllm-chat")
        registry = _build_mock_registry(services={"vllm-chat": svc})
        driver = _build_mock_driver(state=ServiceState.STOPPED)
        driver.supports_sleep = True
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)
        result = await lm.wake("vllm-chat")

        assert isinstance(result, WakeResult)
        assert result.success is False
        assert "stopped" in (result.reason or "").lower()

    async def test_wake_returns_error_for_unknown_service(self) -> None:
        """wake() should return error for non-existent service."""
        from gpumod.services.lifecycle import WakeResult

        registry = _build_mock_registry(services={})
        lm = LifecycleManager(registry)
        result = await lm.wake("unknown-svc")

        assert isinstance(result, WakeResult)
        assert result.success is False
        assert "not found" in (result.reason or "").lower()

    async def test_wake_includes_latency_ms(self) -> None:
        """wake() result should include latency in milliseconds."""
        from gpumod.services.lifecycle import WakeResult

        svc = _make_service(id="vllm-chat")
        registry = _build_mock_registry(services={"vllm-chat": svc})
        driver = _build_mock_driver(state=ServiceState.SLEEPING)
        driver.supports_sleep = True
        driver.wake = AsyncMock()
        registry.get_driver = lambda dtype: driver

        lm = LifecycleManager(registry)
        result = await lm.wake("vllm-chat")

        assert isinstance(result, WakeResult)
        assert result.latency_ms is not None
        assert result.latency_ms >= 0
