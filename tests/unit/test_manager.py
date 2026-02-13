"""Tests for gpumod.services.manager — ServiceManager orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gpumod.models import (
    DriverType,
    GPUInfo,
    Service,
    ServiceState,
    ServiceStatus,
    SleepMode,
    SystemStatus,
    VRAMUsage,
)
from gpumod.services.manager import ServiceManager
from gpumod.services.vram import NvidiaSmiError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(
    id: str,
    name: str = "Test Service",
    driver: DriverType = DriverType.VLLM,
    port: int = 8000,
    vram_mb: int = 2500,
    sleep_mode: SleepMode = SleepMode.NONE,
) -> Service:
    return Service(
        id=id,
        name=name,
        driver=driver,
        port=port,
        vram_mb=vram_mb,
        sleep_mode=sleep_mode,
        health_endpoint="/health",
        model_id="org/model",
        unit_name=f"{id}.service",
        depends_on=[],
        startup_timeout=60,
        extra_config={},
    )


# Pre-configured services
SVC_EMBED = _make_service(id="svc-embed", name="Embedding", vram_mb=2500)
SVC_DEVSTRAL = _make_service(id="svc-devstral", name="Devstral", vram_mb=19000)
SVC_RAG_LLM = _make_service(id="svc-rag-llm", name="RAG LLM", vram_mb=5000)
SVC_RERANKER = _make_service(id="svc-reranker", name="Reranker", vram_mb=3000)

ALL_SERVICES = [SVC_EMBED, SVC_DEVSTRAL, SVC_RAG_LLM, SVC_RERANKER]

# Mode definitions:
# "code" = [svc-embed, svc-devstral]
# "rag"  = [svc-embed, svc-rag-llm, svc-reranker]
CODE_SERVICES = [SVC_EMBED, SVC_DEVSTRAL]
RAG_SERVICES = [SVC_EMBED, SVC_RAG_LLM, SVC_RERANKER]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_mock_db(current_mode: str | None = "code") -> AsyncMock:
    """Build a mock Database with code and rag modes."""
    from gpumod.models import Mode

    db = AsyncMock()

    modes = {
        "code": Mode(id="code", name="Code", services=["svc-embed", "svc-devstral"]),
        "rag": Mode(id="rag", name="RAG", services=["svc-embed", "svc-rag-llm", "svc-reranker"]),
    }
    mode_services = {
        "code": CODE_SERVICES,
        "rag": RAG_SERVICES,
    }

    async def _get_mode(mode_id: str) -> Mode | None:
        return modes.get(mode_id)

    async def _get_mode_services(mode_id: str) -> list[Service]:
        return mode_services.get(mode_id, [])

    async def _get_current_mode() -> str | None:
        return current_mode

    async def _set_current_mode(mode_id: str) -> None:
        pass

    async def _list_services() -> list[Service]:
        return ALL_SERVICES

    db.get_mode = AsyncMock(side_effect=_get_mode)
    db.get_mode_services = AsyncMock(side_effect=_get_mode_services)
    db.get_current_mode = AsyncMock(side_effect=_get_current_mode)
    db.set_current_mode = AsyncMock(side_effect=_set_current_mode)
    db.list_services = AsyncMock(side_effect=_list_services)

    return db


def _build_mock_registry() -> AsyncMock:
    """Build a mock ServiceRegistry."""
    registry = AsyncMock()

    svc_map = {s.id: s for s in ALL_SERVICES}

    async def _get(service_id: str) -> Service:
        if service_id not in svc_map:
            raise KeyError(f"Service not found: {service_id!r}")
        return svc_map[service_id]

    async def _list_all() -> list[Service]:
        return ALL_SERVICES

    async def _list_running() -> list[Service]:
        # By default, no services are running (for basic tests)
        return []

    registry.get = AsyncMock(side_effect=_get)
    registry.list_all = AsyncMock(side_effect=_list_all)
    registry.list_running = AsyncMock(side_effect=_list_running)

    # get_driver returns a mock driver per service
    driver = AsyncMock()
    driver.status = AsyncMock(return_value=ServiceStatus(state=ServiceState.RUNNING))
    registry.get_driver = lambda dtype: driver
    registry._default_driver = driver

    return registry


def _build_mock_lifecycle() -> AsyncMock:
    """Build a mock LifecycleManager."""
    lifecycle = AsyncMock()
    lifecycle.start = AsyncMock()
    lifecycle.stop = AsyncMock()
    return lifecycle


def _build_mock_vram(
    gpu_info: GPUInfo | None = None,
    usage: VRAMUsage | None = None,
) -> AsyncMock:
    """Build a mock VRAMTracker."""
    vram = AsyncMock()

    if gpu_info is None:
        gpu_info = GPUInfo(name="RTX 4090", vram_total_mb=24000, driver="550.0")
    if usage is None:
        usage = VRAMUsage(total_mb=24000, used_mb=0, free_mb=24000)

    vram.get_gpu_info = AsyncMock(return_value=gpu_info)
    vram.get_usage = AsyncMock(return_value=usage)

    async def _estimate_service_vram(service: Service) -> int:
        return service.vram_mb

    vram.estimate_service_vram = AsyncMock(side_effect=_estimate_service_vram)

    return vram


def _build_mock_sleep() -> AsyncMock:
    """Build a mock SleepController."""
    return AsyncMock()


@pytest.fixture
def mock_db() -> AsyncMock:
    return _build_mock_db(current_mode="code")


@pytest.fixture
def mock_registry() -> AsyncMock:
    return _build_mock_registry()


@pytest.fixture
def mock_lifecycle() -> AsyncMock:
    return _build_mock_lifecycle()


@pytest.fixture
def mock_vram() -> AsyncMock:
    return _build_mock_vram()


@pytest.fixture
def mock_sleep() -> AsyncMock:
    return _build_mock_sleep()


@pytest.fixture
def manager(
    mock_db: AsyncMock,
    mock_registry: AsyncMock,
    mock_lifecycle: AsyncMock,
    mock_vram: AsyncMock,
    mock_sleep: AsyncMock,
) -> ServiceManager:
    return ServiceManager(
        db=mock_db,
        registry=mock_registry,
        lifecycle=mock_lifecycle,
        vram=mock_vram,
        sleep=mock_sleep,
    )


# ---------------------------------------------------------------------------
# Mode switch tests
# ---------------------------------------------------------------------------


class TestSwitchCodeToRag:
    """Test switching from 'code' to 'rag' mode."""

    async def test_switch_stops_devstral_starts_rag_services(
        self, manager: ServiceManager, mock_lifecycle: AsyncMock
    ) -> None:
        """Switching code->rag should stop svc-devstral, start svc-rag-llm + svc-reranker."""
        result = await manager.switch_mode("rag")

        assert result.success is True
        # svc-devstral should be stopped (in code, not in rag)
        mock_lifecycle.stop.assert_any_call("svc-devstral")
        # svc-rag-llm and svc-reranker should be started (in rag, not in code)
        started = {c.args[0] for c in mock_lifecycle.start.call_args_list}
        assert "svc-rag-llm" in started
        assert "svc-reranker" in started
        # svc-embed should NOT be stopped or started (shared)
        stopped = {c.args[0] for c in mock_lifecycle.stop.call_args_list}
        assert "svc-embed" not in stopped
        assert "svc-embed" not in started


class TestSwitchReturnsModeResult:
    """Test switch returns correct ModeResult."""

    async def test_switch_returns_correct_started_stopped_lists(
        self, manager: ServiceManager
    ) -> None:
        """ModeResult should contain correct started and stopped service lists."""
        result = await manager.switch_mode("rag")

        assert result.success is True
        assert result.mode_id == "rag"
        assert set(result.stopped) == {"svc-devstral"}
        assert set(result.started) == {"svc-rag-llm", "svc-reranker"}


class TestSwitchToUnknownMode:
    """Test switch to a mode that doesn't exist."""

    async def test_switch_to_unknown_mode_returns_failure(self, manager: ServiceManager) -> None:
        """Switching to a non-existent mode should return ModeResult(success=False)."""
        result = await manager.switch_mode("nonexistent")

        assert result.success is False
        assert "Mode not found: nonexistent" in (result.errors[0] if result.errors else "")


class TestSwitchVramExceeds:
    """Test switch when VRAM would exceed GPU capacity."""

    async def test_switch_returns_vram_error_when_exceeds(
        self,
        mock_db: AsyncMock,
        mock_registry: AsyncMock,
        mock_lifecycle: AsyncMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """Switching to a mode that exceeds VRAM should return failure with VRAM error."""
        # Build a vram tracker with a tiny GPU (only 1000 MB)
        small_gpu = GPUInfo(name="Tiny GPU", vram_total_mb=1000, driver="550.0")
        vram = _build_mock_vram(gpu_info=small_gpu)

        mgr = ServiceManager(
            db=mock_db,
            registry=mock_registry,
            lifecycle=mock_lifecycle,
            vram=vram,
            sleep=mock_sleep,
        )

        result = await mgr.switch_mode("rag")

        assert result.success is False
        assert any("VRAM" in e for e in result.errors)
        # Should not have started or stopped anything
        mock_lifecycle.start.assert_not_called()
        mock_lifecycle.stop.assert_not_called()


class TestSwitchFromNone:
    """Test switch from None (no current mode) starts all target services."""

    async def test_switch_from_none_starts_all_target_services(
        self,
        mock_registry: AsyncMock,
        mock_lifecycle: AsyncMock,
        mock_vram: AsyncMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """When no current mode, switching should start all services in target mode."""
        db = _build_mock_db(current_mode=None)
        mgr = ServiceManager(
            db=db,
            registry=mock_registry,
            lifecycle=mock_lifecycle,
            vram=mock_vram,
            sleep=mock_sleep,
        )

        result = await mgr.switch_mode("rag")

        assert result.success is True
        started = {c.args[0] for c in mock_lifecycle.start.call_args_list}
        assert started == {"svc-embed", "svc-rag-llm", "svc-reranker"}
        # Nothing to stop
        mock_lifecycle.stop.assert_not_called()


class TestSwitchStopsBeforeStarts:
    """Test that stops happen BEFORE starts (free VRAM first)."""

    async def test_stops_happen_before_starts(
        self, manager: ServiceManager, mock_lifecycle: AsyncMock
    ) -> None:
        """Services should be stopped before new services are started."""
        call_order: list[str] = []

        async def _track_stop(service_id: str) -> None:
            call_order.append(f"stop:{service_id}")

        async def _track_start(service_id: str) -> None:
            call_order.append(f"start:{service_id}")

        mock_lifecycle.stop = AsyncMock(side_effect=_track_stop)
        mock_lifecycle.start = AsyncMock(side_effect=_track_start)

        await manager.switch_mode("rag")

        # Find the indices: all stops should come before all starts
        stop_indices = [i for i, c in enumerate(call_order) if c.startswith("stop:")]
        start_indices = [i for i, c in enumerate(call_order) if c.startswith("start:")]

        assert stop_indices, "Expected at least one stop call"
        assert start_indices, "Expected at least one start call"
        assert max(stop_indices) < min(start_indices), (
            f"All stops should happen before all starts. Order: {call_order}"
        )


class TestSwitchUpdatesDbOnSuccess:
    """Test that switch updates current_mode in DB after successful switch."""

    async def test_updates_current_mode_in_db(
        self, manager: ServiceManager, mock_db: AsyncMock
    ) -> None:
        """Successful switch should call db.set_current_mode."""
        await manager.switch_mode("rag")

        mock_db.set_current_mode.assert_called_once_with("rag")


class TestSwitchDoesNotUpdateDbOnFailure:
    """Test that switch does NOT update current_mode on failure."""

    async def test_does_not_update_mode_on_failure(
        self,
        mock_db: AsyncMock,
        mock_registry: AsyncMock,
        mock_lifecycle: AsyncMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """Failed switch (e.g., VRAM exceeds) should not update current_mode."""
        small_gpu = GPUInfo(name="Tiny GPU", vram_total_mb=1000, driver="550.0")
        vram = _build_mock_vram(gpu_info=small_gpu)

        mgr = ServiceManager(
            db=mock_db,
            registry=mock_registry,
            lifecycle=mock_lifecycle,
            vram=vram,
            sleep=mock_sleep,
        )

        result = await mgr.switch_mode("rag")

        assert result.success is False
        mock_db.set_current_mode.assert_not_called()


# ---------------------------------------------------------------------------
# Status tests
# ---------------------------------------------------------------------------


class TestGetStatus:
    """Test get_status() returns full system status."""

    async def test_get_status_includes_services(self, manager: ServiceManager) -> None:
        """get_status should include all services with their statuses."""
        status = await manager.get_status()

        assert isinstance(status, SystemStatus)
        assert len(status.services) == len(ALL_SERVICES)
        service_ids = {si.service.id for si in status.services}
        assert service_ids == {s.id for s in ALL_SERVICES}

    async def test_get_status_includes_gpu_info(self, manager: ServiceManager) -> None:
        """get_status should include GPU info from VRAMTracker."""
        status = await manager.get_status()

        assert status.gpu is not None
        assert status.gpu.name == "RTX 4090"
        assert status.gpu.vram_total_mb == 24000


class TestGetStatusNvidiaSmiFailure:
    """Test get_status gracefully handles nvidia-smi failure."""

    async def test_get_status_gpu_none_on_nvidia_smi_failure(
        self,
        mock_db: AsyncMock,
        mock_registry: AsyncMock,
        mock_lifecycle: AsyncMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """get_status should return gpu=None when nvidia-smi fails."""
        vram = AsyncMock()
        vram.get_gpu_info = AsyncMock(side_effect=NvidiaSmiError("nvidia-smi not found"))
        vram.get_usage = AsyncMock(side_effect=NvidiaSmiError("nvidia-smi not found"))

        mgr = ServiceManager(
            db=mock_db,
            registry=mock_registry,
            lifecycle=mock_lifecycle,
            vram=vram,
            sleep=mock_sleep,
        )

        status = await mgr.get_status()

        assert isinstance(status, SystemStatus)
        assert status.gpu is None
        assert status.vram is None
        # Services should still be populated
        assert len(status.services) == len(ALL_SERVICES)


# ---------------------------------------------------------------------------
# Convenience tests
# ---------------------------------------------------------------------------


class TestStartServiceDelegation:
    """Test start_service delegates to lifecycle.start."""

    async def test_start_service_delegates(
        self, manager: ServiceManager, mock_lifecycle: AsyncMock
    ) -> None:
        """start_service should call lifecycle.start with the given service_id."""
        await manager.start_service("svc-embed")

        mock_lifecycle.start.assert_called_once_with("svc-embed")


class TestStopServiceDelegation:
    """Test stop_service delegates to lifecycle.stop."""

    async def test_stop_service_delegates(
        self, manager: ServiceManager, mock_lifecycle: AsyncMock
    ) -> None:
        """stop_service should call lifecycle.stop with the given service_id."""
        await manager.stop_service("svc-embed")

        mock_lifecycle.stop.assert_called_once_with("svc-embed")


# ---------------------------------------------------------------------------
# Sleep-capable services for sleep-aware mode switch tests
# ---------------------------------------------------------------------------

# Sleep-capable versions of services (vLLM with L1 sleep mode)
SVC_DEVSTRAL_SLEEPABLE = _make_service(
    id="svc-devstral",
    name="Devstral",
    vram_mb=19000,
    sleep_mode=SleepMode.L1,
)
SVC_RAG_LLM_SLEEPABLE = _make_service(
    id="svc-rag-llm",
    name="RAG LLM",
    vram_mb=5000,
    sleep_mode=SleepMode.L1,
)

# Non-sleepable services (embedding, reranker typically don't need sleep)
SVC_EMBED_NON_SLEEP = _make_service(
    id="svc-embed",
    name="Embedding",
    vram_mb=2500,
    sleep_mode=SleepMode.NONE,
)
SVC_RERANKER_NON_SLEEP = _make_service(
    id="svc-reranker",
    name="Reranker",
    vram_mb=3000,
    sleep_mode=SleepMode.NONE,
)

# Mixed services for mode tests
SLEEPABLE_CODE_SERVICES = [SVC_EMBED_NON_SLEEP, SVC_DEVSTRAL_SLEEPABLE]
SLEEPABLE_RAG_SERVICES = [SVC_EMBED_NON_SLEEP, SVC_RAG_LLM_SLEEPABLE, SVC_RERANKER_NON_SLEEP]
ALL_SLEEPABLE_SERVICES = [
    SVC_EMBED_NON_SLEEP,
    SVC_DEVSTRAL_SLEEPABLE,
    SVC_RAG_LLM_SLEEPABLE,
    SVC_RERANKER_NON_SLEEP,
]


def _build_sleepable_mock_db(current_mode: str | None = "code") -> AsyncMock:
    """Build a mock Database with sleep-capable services."""
    from gpumod.models import Mode

    db = AsyncMock()

    modes = {
        "code": Mode(id="code", name="Code", services=["svc-embed", "svc-devstral"]),
        "rag": Mode(id="rag", name="RAG", services=["svc-embed", "svc-rag-llm", "svc-reranker"]),
    }
    mode_services = {
        "code": SLEEPABLE_CODE_SERVICES,
        "rag": SLEEPABLE_RAG_SERVICES,
    }

    async def _get_mode(mode_id: str) -> Mode | None:
        return modes.get(mode_id)

    async def _get_mode_services(mode_id: str) -> list[Service]:
        return mode_services.get(mode_id, [])

    async def _get_current_mode() -> str | None:
        return current_mode

    async def _set_current_mode(mode_id: str) -> None:
        pass

    async def _list_services() -> list[Service]:
        return ALL_SLEEPABLE_SERVICES

    db.get_mode = AsyncMock(side_effect=_get_mode)
    db.get_mode_services = AsyncMock(side_effect=_get_mode_services)
    db.get_current_mode = AsyncMock(side_effect=_get_current_mode)
    db.set_current_mode = AsyncMock(side_effect=_set_current_mode)
    db.list_services = AsyncMock(side_effect=_list_services)

    return db


def _build_sleepable_mock_registry(
    service_states: dict[str, ServiceState] | None = None,
) -> AsyncMock:
    """Build a mock ServiceRegistry with sleep-capable drivers.

    Parameters
    ----------
    service_states:
        Map of service_id -> ServiceState. Defaults to RUNNING for all.
    """
    registry = AsyncMock()
    states = service_states or {}

    svc_map = {s.id: s for s in ALL_SLEEPABLE_SERVICES}

    async def _get(service_id: str) -> Service:
        if service_id not in svc_map:
            raise KeyError(f"Service not found: {service_id!r}")
        return svc_map[service_id]

    async def _list_all() -> list[Service]:
        return ALL_SLEEPABLE_SERVICES

    async def _list_running() -> list[Service]:
        # Return services that are RUNNING or SLEEPING based on states dict
        running_states = {ServiceState.RUNNING, ServiceState.SLEEPING}
        return [
            svc
            for svc in ALL_SLEEPABLE_SERVICES
            if states.get(svc.id, ServiceState.RUNNING) in running_states
        ]

    registry.get = AsyncMock(side_effect=_get)
    registry.list_all = AsyncMock(side_effect=_list_all)
    registry.list_running = AsyncMock(side_effect=_list_running)

    # Create a shared driver that returns state based on service
    driver = AsyncMock()

    async def _status(service: Service) -> ServiceStatus:
        state = states.get(service.id, ServiceState.RUNNING)
        return ServiceStatus(state=state)

    driver.status = AsyncMock(side_effect=_status)
    driver.supports_sleep = True  # vLLM supports sleep

    def _get_driver(_dtype: DriverType) -> AsyncMock:
        return driver

    registry.get_driver = _get_driver
    registry._shared_driver = driver

    return registry


def _build_sleepable_mock_lifecycle() -> AsyncMock:
    """Build a mock LifecycleManager with sleep/wake support."""
    from gpumod.services.lifecycle import SleepResult, WakeResult

    lifecycle = AsyncMock()
    lifecycle.start = AsyncMock()
    lifecycle.stop = AsyncMock()
    lifecycle.sleep = AsyncMock(return_value=SleepResult(success=True, latency_ms=50.0))
    lifecycle.wake = AsyncMock(return_value=WakeResult(success=True, latency_ms=100.0))
    return lifecycle


# ---------------------------------------------------------------------------
# Sleep-aware mode switch tests
# ---------------------------------------------------------------------------


class TestSleepAwareSwitchSleepsOutgoing:
    """Test that sleep-capable outgoing services are slept, not stopped."""

    async def test_switch_sleeps_sleep_capable_outgoing_service(self) -> None:
        """Switching code->rag should sleep svc-devstral (sleep_mode=L1), not stop it."""
        db = _build_sleepable_mock_db(current_mode="code")
        registry = _build_sleepable_mock_registry()
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("rag")

        assert result.success is True
        # svc-devstral is sleep-capable, should be slept
        lifecycle.sleep.assert_any_call("svc-devstral")
        # svc-devstral should NOT be stopped
        stopped_ids = {c.args[0] for c in lifecycle.stop.call_args_list}
        assert "svc-devstral" not in stopped_ids


class TestSleepAwareSwitchStopsNonSleepable:
    """Test that non-sleep services still use stop."""

    async def test_switch_stops_non_sleep_outgoing_service(self) -> None:
        """Switching rag->code should stop svc-reranker (sleep_mode=none)."""
        db = _build_sleepable_mock_db(current_mode="rag")
        registry = _build_sleepable_mock_registry()
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("code")

        assert result.success is True
        # svc-reranker is NOT sleep-capable, should be stopped
        lifecycle.stop.assert_any_call("svc-reranker")
        # svc-reranker should NOT be slept
        slept_ids = {c.args[0] for c in lifecycle.sleep.call_args_list}
        assert "svc-reranker" not in slept_ids


class TestSleepAwareSwitchWakesSleeping:
    """Test that sleeping services entering mode are woken."""

    async def test_switch_wakes_sleeping_service(self) -> None:
        """Switching rag->code should wake svc-devstral if it's sleeping."""
        from gpumod.services.lifecycle import WakeResult

        db = _build_sleepable_mock_db(current_mode="rag")
        # svc-devstral is SLEEPING (from previous mode switch)
        registry = _build_sleepable_mock_registry(
            service_states={"svc-devstral": ServiceState.SLEEPING}
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        lifecycle.wake = AsyncMock(return_value=WakeResult(success=True, latency_ms=100.0))
        vram = _build_mock_vram()
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("code")

        assert result.success is True
        # svc-devstral is sleeping, should be woken
        lifecycle.wake.assert_any_call("svc-devstral")
        # svc-devstral should NOT be started (wake is sufficient)
        started_ids = {c.args[0] for c in lifecycle.start.call_args_list}
        assert "svc-devstral" not in started_ids


class TestSleepAwareSwitchStartsStopped:
    """Test that stopped services entering mode are started."""

    async def test_switch_starts_stopped_service(self) -> None:
        """Switching code->rag should start svc-rag-llm if it's stopped."""
        db = _build_sleepable_mock_db(current_mode="code")
        # svc-rag-llm is STOPPED (never started)
        registry = _build_sleepable_mock_registry(
            service_states={"svc-rag-llm": ServiceState.STOPPED}
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("rag")

        assert result.success is True
        # svc-rag-llm is stopped, should be started (not woken)
        lifecycle.start.assert_any_call("svc-rag-llm")
        # svc-rag-llm should NOT be woken (it's stopped, not sleeping)
        woken_ids = {c.args[0] for c in lifecycle.wake.call_args_list}
        assert "svc-rag-llm" not in woken_ids


class TestSleepAwareSwitchMixed:
    """Test mixed transitions with both sleep-capable and non-sleep services."""

    async def test_switch_handles_mixed_transitions(self) -> None:
        """Switching code->rag should sleep devstral, start rag-llm, start reranker."""
        db = _build_sleepable_mock_db(current_mode="code")
        # All services start as RUNNING or STOPPED
        registry = _build_sleepable_mock_registry(
            service_states={
                "svc-embed": ServiceState.RUNNING,  # Shared, no transition
                "svc-devstral": ServiceState.RUNNING,  # Leaving, sleep-capable
                "svc-rag-llm": ServiceState.STOPPED,  # Entering, stopped
                "svc-reranker": ServiceState.STOPPED,  # Entering, stopped
            }
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("rag")

        assert result.success is True

        # Collect all calls
        slept_ids = {c.args[0] for c in lifecycle.sleep.call_args_list}
        stopped_ids = {c.args[0] for c in lifecycle.stop.call_args_list}
        started_ids = {c.args[0] for c in lifecycle.start.call_args_list}
        woken_ids = {c.args[0] for c in lifecycle.wake.call_args_list}

        # svc-devstral: leaving + sleep-capable → slept
        assert "svc-devstral" in slept_ids
        assert "svc-devstral" not in stopped_ids

        # svc-embed: shared → no transition
        assert "svc-embed" not in slept_ids
        assert "svc-embed" not in stopped_ids
        assert "svc-embed" not in started_ids
        assert "svc-embed" not in woken_ids

        # svc-rag-llm: entering + stopped → started
        assert "svc-rag-llm" in started_ids
        assert "svc-rag-llm" not in woken_ids

        # svc-reranker: entering + stopped → started
        assert "svc-reranker" in started_ids


class TestSleepAwareSwitchIdempotent:
    """Test that sleeping services leaving a mode are stopped (gpumod-77o fix)."""

    async def test_switch_stops_already_sleeping_outgoing(self) -> None:
        """Outgoing sleeping service should be stopped to prevent orphans.

        This behavior was changed in gpumod-77o: sleeping services that are
        leaving a mode are now STOPPED (not skipped) to prevent orphan services
        from accumulating across mode switches.
        """
        db = _build_sleepable_mock_db(current_mode="code")
        # svc-devstral is already SLEEPING (orphan from prior mode switch)
        registry = _build_sleepable_mock_registry(
            service_states={"svc-devstral": ServiceState.SLEEPING}
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=True)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("rag")

        assert result.success is True
        # Sleeping services leaving the mode should be STOPPED (not skipped)
        stopped_ids = {c.args[0] for c in lifecycle.stop.call_args_list}
        assert "svc-devstral" in stopped_ids, (
            "Sleeping services leaving a mode should be stopped to prevent orphans"
        )
        # Should NOT try to sleep an already-sleeping service
        slept_ids = {c.args[0] for c in lifecycle.sleep.call_args_list}
        assert "svc-devstral" not in slept_ids


class TestVramWaitOnSwitch:
    """Test that VRAM wait is called between stopping and starting services."""

    async def test_vram_wait_called_after_stops_before_starts(self) -> None:
        """Switch should wait for VRAM release between stop and start phases."""
        db = _build_sleepable_mock_db(current_mode="code")
        registry = _build_sleepable_mock_registry(
            service_states={
                "svc-devstral": ServiceState.RUNNING,
                "svc-rag-llm": ServiceState.STOPPED,
            }
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=True)
        sleep_ctrl = _build_mock_sleep()

        call_order: list[str] = []

        async def _track_stop(service_id: str) -> None:
            call_order.append(f"stop:{service_id}")

        async def _track_start(service_id: str) -> None:
            call_order.append(f"start:{service_id}")

        async def _track_vram_wait(*args, **kwargs) -> bool:
            call_order.append("vram_wait")
            return True

        lifecycle.stop = AsyncMock(side_effect=_track_stop)
        lifecycle.start = AsyncMock(side_effect=_track_start)
        vram.wait_for_vram_release = AsyncMock(side_effect=_track_vram_wait)

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("rag")

        assert result.success is True
        # VRAM wait should be called
        vram.wait_for_vram_release.assert_called_once()
        # Check order: all stops before vram_wait before all starts
        stop_indices = [i for i, c in enumerate(call_order) if c.startswith("stop:")]
        vram_indices = [i for i, c in enumerate(call_order) if c == "vram_wait"]
        start_indices = [i for i, c in enumerate(call_order) if c.startswith("start:")]

        if stop_indices and vram_indices and start_indices:
            # all stops < vram_wait < all starts
            assert max(stop_indices) < min(vram_indices), (
                f"VRAM wait should be after stops. Order: {call_order}"
            )
            assert max(vram_indices) < min(start_indices), (
                f"VRAM wait should be before starts. Order: {call_order}"
            )

    async def test_vram_wait_uses_largest_incoming_service_vram(self) -> None:
        """VRAM wait should use the largest incoming service's vram_mb."""
        db = _build_sleepable_mock_db(current_mode="code")
        registry = _build_sleepable_mock_registry()
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=True)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        await manager.switch_mode("rag")

        # svc-rag-llm (5000 MB) is larger than svc-reranker (3000 MB)
        vram.wait_for_vram_release.assert_called_once()
        call_args = vram.wait_for_vram_release.call_args
        required = call_args.kwargs.get("required_mb", call_args.args[0] if call_args.args else 0)
        assert required == 5000


class TestVramTimeoutAborts:
    """Test that VRAM timeout aborts mode switch instead of proceeding (gpumod-277)."""

    async def test_vram_timeout_returns_failure(self) -> None:
        """Mode switch should fail when VRAM wait times out.

        This tests the fix for gpumod-277: previously, VRAM timeout would
        log a warning and proceed anyway, potentially causing system crashes.
        Now it should return ModeResult(success=False) with an error message.
        """
        db = _build_sleepable_mock_db(current_mode="code")
        registry = _build_sleepable_mock_registry(
            service_states={
                "svc-devstral": ServiceState.RUNNING,
                "svc-rag-llm": ServiceState.STOPPED,
            }
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        # Simulate VRAM timeout - wait_for_vram_release returns False
        vram.wait_for_vram_release = AsyncMock(return_value=False)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("rag")

        # Mode switch should fail
        assert result.success is False
        assert result.mode_id == "rag"
        assert any("VRAM" in e for e in result.errors)
        # Services should NOT be started after timeout
        lifecycle.start.assert_not_called()
        lifecycle.wake.assert_not_called()
        # DB should NOT be updated (mode didn't change)

    async def test_vram_timeout_does_not_update_db(self) -> None:
        """Failed mode switch due to VRAM timeout should not update current_mode."""
        db = _build_sleepable_mock_db(current_mode="code")
        registry = _build_sleepable_mock_registry(
            service_states={
                "svc-devstral": ServiceState.RUNNING,
                "svc-rag-llm": ServiceState.STOPPED,
            }
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=False)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        await manager.switch_mode("rag")

        # DB should NOT be updated on failure
        db.set_current_mode.assert_not_called()

    async def test_vram_timeout_error_message_is_actionable(self) -> None:
        """VRAM timeout error should explain the issue clearly."""
        db = _build_sleepable_mock_db(current_mode="code")
        registry = _build_sleepable_mock_registry(
            service_states={
                "svc-devstral": ServiceState.RUNNING,
                "svc-rag-llm": ServiceState.STOPPED,
            }
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=False)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("rag")

        assert result.success is False
        # Error should mention what happened
        error_text = " ".join(result.errors)
        assert "VRAM" in error_text
        assert "timeout" in error_text.lower() or "released" in error_text.lower()


class TestSleepAwareSwitchOrder:
    """Test that sleep/stop happens before wake/start (free VRAM first)."""

    async def test_sleep_and_stop_before_wake_and_start(self) -> None:
        """Outgoing transitions should complete before incoming transitions."""
        db = _build_sleepable_mock_db(current_mode="code")
        registry = _build_sleepable_mock_registry(
            service_states={
                "svc-devstral": ServiceState.RUNNING,  # Leaving
                "svc-rag-llm": ServiceState.STOPPED,  # Entering
            }
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        sleep_ctrl = _build_mock_sleep()

        call_order: list[str] = []

        async def _track_sleep(service_id: str) -> None:
            from gpumod.services.lifecycle import SleepResult

            call_order.append(f"sleep:{service_id}")
            return SleepResult(success=True)

        async def _track_stop(service_id: str) -> None:
            call_order.append(f"stop:{service_id}")

        async def _track_wake(service_id: str) -> None:
            from gpumod.services.lifecycle import WakeResult

            call_order.append(f"wake:{service_id}")
            return WakeResult(success=True)

        async def _track_start(service_id: str) -> None:
            call_order.append(f"start:{service_id}")

        lifecycle.sleep = AsyncMock(side_effect=_track_sleep)
        lifecycle.stop = AsyncMock(side_effect=_track_stop)
        lifecycle.wake = AsyncMock(side_effect=_track_wake)
        lifecycle.start = AsyncMock(side_effect=_track_start)

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        await manager.switch_mode("rag")

        # Find indices of outgoing (sleep/stop) vs incoming (wake/start)
        outgoing_indices = [
            i for i, c in enumerate(call_order) if c.startswith(("sleep:", "stop:"))
        ]
        incoming_indices = [
            i for i, c in enumerate(call_order) if c.startswith(("wake:", "start:"))
        ]

        if outgoing_indices and incoming_indices:
            assert max(outgoing_indices) < min(incoming_indices), (
                f"Outgoing transitions should complete before incoming. Order: {call_order}"
            )


# ---------------------------------------------------------------------------
# Orphan service cleanup tests (gpumod-77o)
# ---------------------------------------------------------------------------


# Additional services for orphan tests
SVC_ORPHAN = _make_service(
    id="svc-orphan",
    name="Orphan Service",
    vram_mb=8000,
    sleep_mode=SleepMode.L1,
    driver=DriverType.LLAMACPP,
)
SVC_ORPHAN_2 = _make_service(
    id="svc-orphan-2",
    name="Orphan Service 2",
    vram_mb=4000,
    sleep_mode=SleepMode.NONE,
    driver=DriverType.VLLM,
)

# All services including orphans
ALL_ORPHAN_TEST_SERVICES = [*ALL_SLEEPABLE_SERVICES, SVC_ORPHAN, SVC_ORPHAN_2]


def _build_orphan_test_mock_db(current_mode: str | None = "code") -> AsyncMock:
    """Build a mock Database for orphan tests with blank mode."""
    from gpumod.models import Mode

    db = AsyncMock()

    modes = {
        "code": Mode(id="code", name="Code", services=["svc-embed", "svc-devstral"]),
        "rag": Mode(id="rag", name="RAG", services=["svc-embed", "svc-rag-llm", "svc-reranker"]),
        "blank": Mode(id="blank", name="Blank", services=["svc-embed"]),  # Only embed
    }
    mode_services_map = {
        "code": SLEEPABLE_CODE_SERVICES,
        "rag": SLEEPABLE_RAG_SERVICES,
        "blank": [SVC_EMBED_NON_SLEEP],
    }

    async def _get_mode(mode_id: str) -> Mode | None:
        return modes.get(mode_id)

    async def _get_mode_services(mode_id: str) -> list[Service]:
        return mode_services_map.get(mode_id, [])

    async def _get_current_mode() -> str | None:
        return current_mode

    async def _set_current_mode(mode_id: str) -> None:
        pass

    async def _list_services() -> list[Service]:
        return ALL_ORPHAN_TEST_SERVICES

    db.get_mode = AsyncMock(side_effect=_get_mode)
    db.get_mode_services = AsyncMock(side_effect=_get_mode_services)
    db.get_current_mode = AsyncMock(side_effect=_get_current_mode)
    db.set_current_mode = AsyncMock(side_effect=_set_current_mode)
    db.list_services = AsyncMock(side_effect=_list_services)

    return db


def _build_orphan_test_mock_registry(
    service_states: dict[str, ServiceState] | None = None,
    running_services: list[Service] | None = None,
) -> AsyncMock:
    """Build a mock registry that can simulate orphan services.

    Parameters
    ----------
    service_states:
        Map of service_id -> ServiceState for status queries.
    running_services:
        Explicit list of services to return from list_running().
        If None, derives from service_states.
    """
    registry = AsyncMock()
    states = service_states or {}

    svc_map = {s.id: s for s in ALL_ORPHAN_TEST_SERVICES}

    async def _get(service_id: str) -> Service:
        if service_id not in svc_map:
            raise KeyError(f"Service not found: {service_id!r}")
        return svc_map[service_id]

    async def _list_all() -> list[Service]:
        return ALL_ORPHAN_TEST_SERVICES

    async def _list_running() -> list[Service]:
        if running_services is not None:
            return running_services
        # Derive from states: return RUNNING or SLEEPING services
        running_states = {ServiceState.RUNNING, ServiceState.SLEEPING}
        return [
            svc
            for svc in ALL_ORPHAN_TEST_SERVICES
            if states.get(svc.id, ServiceState.STOPPED) in running_states
        ]

    registry.get = AsyncMock(side_effect=_get)
    registry.list_all = AsyncMock(side_effect=_list_all)
    registry.list_running = AsyncMock(side_effect=_list_running)

    # Driver with status based on states
    driver = AsyncMock()

    async def _status(service: Service) -> ServiceStatus:
        state = states.get(service.id, ServiceState.STOPPED)
        return ServiceStatus(state=state)

    driver.status = AsyncMock(side_effect=_status)
    driver.supports_sleep = True

    def _get_driver(_dtype: DriverType) -> AsyncMock:
        return driver

    registry.get_driver = _get_driver
    registry._shared_driver = driver

    return registry


class TestOrphanServiceCleanup:
    """Test that orphan services from prior modes are detected and cleaned up."""

    async def test_orphan_sleeping_service_stopped_on_switch_to_blank(self) -> None:
        """Orphan sleeping service should be stopped when switching to blank mode.

        Scenario: code → nemotron → blank
        - svc-orphan was sleeping from code mode
        - It's not in nemotron's service list
        - When switching to blank, it should be stopped
        """
        # Current mode is "rag" (not code), simulating we already switched
        db = _build_orphan_test_mock_db(current_mode="rag")
        registry = _build_orphan_test_mock_registry(
            service_states={
                "svc-embed": ServiceState.RUNNING,  # Shared, stays running
                "svc-orphan": ServiceState.SLEEPING,  # Orphan from prior mode
                "svc-rag-llm": ServiceState.RUNNING,  # Current mode service
                "svc-reranker": ServiceState.RUNNING,  # Current mode service
            },
            # list_running returns all currently active services
            running_services=[
                SVC_EMBED_NON_SLEEP,
                SVC_ORPHAN,
                SVC_RAG_LLM_SLEEPABLE,
                SVC_RERANKER_NON_SLEEP,
            ],
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=True)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("blank")

        assert result.success is True

        # svc-orphan should be stopped (it's sleeping, not in blank mode)
        stopped_ids = {c.args[0] for c in lifecycle.stop.call_args_list}
        slept_ids = {c.args[0] for c in lifecycle.sleep.call_args_list}

        # Orphan should be handled (either stopped or slept depending on state)
        assert "svc-orphan" in stopped_ids or "svc-orphan" in slept_ids, (
            f"Orphan service should be stopped/slept. Stopped: {stopped_ids}, Slept: {slept_ids}"
        )

    async def test_multiple_orphan_services_cleaned_up(self) -> None:
        """Multiple orphan services from different prior modes should all be cleaned up."""
        db = _build_orphan_test_mock_db(current_mode="code")
        registry = _build_orphan_test_mock_registry(
            service_states={
                "svc-embed": ServiceState.RUNNING,
                "svc-devstral": ServiceState.RUNNING,
                "svc-orphan": ServiceState.SLEEPING,  # Orphan 1
                "svc-orphan-2": ServiceState.RUNNING,  # Orphan 2
            },
            running_services=[
                SVC_EMBED_NON_SLEEP,
                SVC_DEVSTRAL_SLEEPABLE,
                SVC_ORPHAN,
                SVC_ORPHAN_2,
            ],
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=True)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("blank")

        assert result.success is True

        stopped_ids = {c.args[0] for c in lifecycle.stop.call_args_list}
        slept_ids = {c.args[0] for c in lifecycle.sleep.call_args_list}
        all_handled = stopped_ids | slept_ids

        # Both orphans should be handled
        assert "svc-orphan" in all_handled, "svc-orphan should be cleaned up"
        assert "svc-orphan-2" in all_handled, "svc-orphan-2 should be cleaned up"

        # svc-devstral (in code mode) should also be handled (leaving current mode)
        assert "svc-devstral" in all_handled, "svc-devstral should be handled"

    async def test_orphan_detection_includes_running_not_in_current_mode(self) -> None:
        """Services running but not defined in current mode should be detected as orphans."""
        # Current mode is "blank" with only svc-embed
        db = _build_orphan_test_mock_db(current_mode="blank")
        registry = _build_orphan_test_mock_registry(
            service_states={
                "svc-embed": ServiceState.RUNNING,
                # These are running but NOT in blank mode's definition
                "svc-devstral": ServiceState.RUNNING,
                "svc-rag-llm": ServiceState.SLEEPING,
            },
            running_services=[SVC_EMBED_NON_SLEEP, SVC_DEVSTRAL_SLEEPABLE, SVC_RAG_LLM_SLEEPABLE],
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=True)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        # Switch to rag mode
        result = await manager.switch_mode("rag")

        assert result.success is True

        stopped_ids = {c.args[0] for c in lifecycle.stop.call_args_list}
        slept_ids = {c.args[0] for c in lifecycle.sleep.call_args_list}
        all_handled = stopped_ids | slept_ids

        # svc-devstral was running but not in blank or rag, should be cleaned up
        assert "svc-devstral" in all_handled, (
            f"svc-devstral (orphan) should be cleaned up. Handled: {all_handled}"
        )

    async def test_no_orphans_when_all_running_in_target_mode(self) -> None:
        """No extra cleanup when all running services are in target mode."""
        db = _build_orphan_test_mock_db(current_mode="code")
        registry = _build_orphan_test_mock_registry(
            service_states={
                "svc-embed": ServiceState.RUNNING,
                "svc-devstral": ServiceState.RUNNING,
            },
            # Only code mode services
            running_services=[SVC_EMBED_NON_SLEEP, SVC_DEVSTRAL_SLEEPABLE],
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=True)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("rag")

        assert result.success is True

        # Only svc-devstral should be handled (normal mode switch, not orphan)
        stopped_ids = {c.args[0] for c in lifecycle.stop.call_args_list}
        slept_ids = {c.args[0] for c in lifecycle.sleep.call_args_list}

        assert "svc-devstral" in (stopped_ids | slept_ids)
        # Orphan services should not appear (they're not running)
        assert "svc-orphan" not in (stopped_ids | slept_ids)
        assert "svc-orphan-2" not in (stopped_ids | slept_ids)


# ---------------------------------------------------------------------------
# Edge case tests for orphan cleanup (gpumod-77o)
# ---------------------------------------------------------------------------


class TestOrphanEdgeCases:
    """Edge cases for orphan service detection and cleanup."""

    async def test_switch_to_same_mode_no_changes(self) -> None:
        """Switching to current mode should be a no-op."""
        db = _build_orphan_test_mock_db(current_mode="code")
        registry = _build_orphan_test_mock_registry(
            service_states={
                "svc-embed": ServiceState.RUNNING,
                "svc-devstral": ServiceState.RUNNING,
            },
            running_services=[SVC_EMBED_NON_SLEEP, SVC_DEVSTRAL_SLEEPABLE],
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=True)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("code")

        assert result.success is True
        # No services should be stopped or started (same mode)
        assert lifecycle.stop.call_count == 0
        assert lifecycle.start.call_count == 0
        assert lifecycle.sleep.call_count == 0
        assert lifecycle.wake.call_count == 0

    async def test_orphan_in_unknown_state_is_stopped(self) -> None:
        """Services in UNKNOWN state should be stopped during mode switch."""
        db = _build_orphan_test_mock_db(current_mode="code")
        registry = _build_orphan_test_mock_registry(
            service_states={
                "svc-embed": ServiceState.RUNNING,
                "svc-devstral": ServiceState.RUNNING,
                "svc-orphan": ServiceState.UNKNOWN,  # Unknown state
            },
            running_services=[SVC_EMBED_NON_SLEEP, SVC_DEVSTRAL_SLEEPABLE, SVC_ORPHAN],
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=True)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("blank")

        assert result.success is True
        stopped_ids = {c.args[0] for c in lifecycle.stop.call_args_list}
        # UNKNOWN state services should be stopped
        assert "svc-orphan" in stopped_ids

    async def test_empty_running_list_no_orphans(self) -> None:
        """When no services are running, orphan detection should not fail."""
        db = _build_orphan_test_mock_db(current_mode="code")
        registry = _build_orphan_test_mock_registry(
            service_states={},
            running_services=[],  # Nothing running
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=True)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("rag")

        assert result.success is True
        # Should start rag services, no orphans to clean
        started_ids = {c.args[0] for c in lifecycle.start.call_args_list}
        assert "svc-rag-llm" in started_ids
        assert "svc-reranker" in started_ids

    async def test_all_services_are_orphans(self) -> None:
        """All running services being orphans should still work."""
        db = _build_orphan_test_mock_db(current_mode="blank")
        # All running services are orphans (not in blank mode)
        registry = _build_orphan_test_mock_registry(
            service_states={
                "svc-orphan": ServiceState.RUNNING,
                "svc-orphan-2": ServiceState.SLEEPING,
                "svc-devstral": ServiceState.RUNNING,
            },
            running_services=[SVC_ORPHAN, SVC_ORPHAN_2, SVC_DEVSTRAL_SLEEPABLE],
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=True)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("blank")

        assert result.success is True
        stopped_ids = {c.args[0] for c in lifecycle.stop.call_args_list}
        slept_ids = {c.args[0] for c in lifecycle.sleep.call_args_list}
        all_handled = stopped_ids | slept_ids

        # All orphans should be cleaned up
        assert "svc-orphan" in all_handled
        assert "svc-orphan-2" in all_handled
        assert "svc-devstral" in all_handled

    async def test_orphan_mixed_with_shared_service(self) -> None:
        """Shared service should not be stopped when orphans are cleaned."""
        db = _build_orphan_test_mock_db(current_mode="code")
        # svc-embed is shared between code and rag
        registry = _build_orphan_test_mock_registry(
            service_states={
                "svc-embed": ServiceState.RUNNING,  # Shared
                "svc-devstral": ServiceState.RUNNING,  # Leaving
                "svc-orphan": ServiceState.SLEEPING,  # Orphan
            },
            running_services=[SVC_EMBED_NON_SLEEP, SVC_DEVSTRAL_SLEEPABLE, SVC_ORPHAN],
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=True)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("rag")

        assert result.success is True
        stopped_ids = {c.args[0] for c in lifecycle.stop.call_args_list}
        slept_ids = {c.args[0] for c in lifecycle.sleep.call_args_list}
        started_ids = {c.args[0] for c in lifecycle.start.call_args_list}

        # svc-embed is shared - should NOT be stopped, started, or slept
        assert "svc-embed" not in stopped_ids
        assert "svc-embed" not in slept_ids
        assert "svc-embed" not in started_ids

        # Orphan should be stopped
        assert "svc-orphan" in stopped_ids

    async def test_stopped_services_not_in_to_stop(self) -> None:
        """Already-stopped services should not trigger stop calls."""
        db = _build_orphan_test_mock_db(current_mode="code")
        registry = _build_orphan_test_mock_registry(
            service_states={
                "svc-embed": ServiceState.RUNNING,
                "svc-devstral": ServiceState.STOPPED,  # Already stopped
            },
            # Only embed is running
            running_services=[SVC_EMBED_NON_SLEEP],
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=True)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("rag")

        assert result.success is True
        # svc-devstral is already stopped - no action needed for it
        # The key is that no error occurs and we don't try to stop/sleep it

    async def test_mode_switch_logs_orphan_detection(self) -> None:
        """Verify orphan services are properly detected in to_stop calculation."""
        db = _build_orphan_test_mock_db(current_mode="rag")
        # svc-orphan is running but not in rag mode
        registry = _build_orphan_test_mock_registry(
            service_states={
                "svc-embed": ServiceState.RUNNING,
                "svc-rag-llm": ServiceState.RUNNING,
                "svc-orphan": ServiceState.RUNNING,  # Orphan
            },
            running_services=[SVC_EMBED_NON_SLEEP, SVC_RAG_LLM_SLEEPABLE, SVC_ORPHAN],
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        vram = _build_mock_vram()
        vram.wait_for_vram_release = AsyncMock(return_value=True)
        sleep_ctrl = _build_mock_sleep()

        manager = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep_ctrl,
        )

        result = await manager.switch_mode("code")

        assert result.success is True
        # svc-orphan should be stopped (it's running but not in rag or code)
        stopped_ids = {c.args[0] for c in lifecycle.stop.call_args_list}
        slept_ids = {c.args[0] for c in lifecycle.sleep.call_args_list}
        assert "svc-orphan" in (stopped_ids | slept_ids)
