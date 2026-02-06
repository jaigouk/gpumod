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
