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

    registry.get = AsyncMock(side_effect=_get)
    registry.list_all = AsyncMock(side_effect=_list_all)

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

    registry.get = AsyncMock(side_effect=_get)
    registry.list_all = AsyncMock(side_effect=_list_all)

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
    """Test idempotent behavior for already-sleeping services."""

    async def test_switch_skips_already_sleeping_outgoing(self) -> None:
        """Outgoing service that's already sleeping should not be double-slept."""
        from gpumod.services.lifecycle import SleepResult

        db = _build_sleepable_mock_db(current_mode="code")
        # svc-devstral is already SLEEPING
        registry = _build_sleepable_mock_registry(
            service_states={"svc-devstral": ServiceState.SLEEPING}
        )
        lifecycle = _build_sleepable_mock_lifecycle()
        # sleep returns skipped=True for already-sleeping
        lifecycle.sleep = AsyncMock(
            return_value=SleepResult(success=True, skipped=True, reason="Already sleeping")
        )
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
        # Should not try to stop an already-sleeping service
        stopped_ids = {c.args[0] for c in lifecycle.stop.call_args_list}
        assert "svc-devstral" not in stopped_ids


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
