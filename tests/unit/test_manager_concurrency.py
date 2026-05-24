"""Tests for ServiceManager.switch_mode concurrency — asyncio.Lock serialization."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from gpumod.models import (
    DriverType,
    GPUInfo,
    Mode,
    ModeResult,
    RAMUsage,
    Service,
    ServiceState,
    ServiceStatus,
    SleepMode,
    VRAMUsage,
)
from gpumod.services.lifecycle import LifecycleError
from gpumod.services.manager import ServiceManager

if TYPE_CHECKING:
    import pytest


# ---------------------------------------------------------------------------
# Helpers (duplicated from test_manager.py to avoid cross-file coupling)
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


SVC_A = _make_service(id="svc-a", name="Service A", vram_mb=2000)
SVC_B = _make_service(id="svc-b", name="Service B", vram_mb=3000)
ALL_SERVICES = [SVC_A, SVC_B]


def _build_mock_db(current_mode: str | None = None) -> AsyncMock:
    """Build a mock Database with 'alpha' and 'beta' modes."""
    db = AsyncMock()

    modes = {
        "alpha": Mode(id="alpha", name="Alpha", services=["svc-a"]),
        "beta": Mode(id="beta", name="Beta", services=["svc-b"]),
    }
    mode_services: dict[str, list[Service]] = {
        "alpha": [SVC_A],
        "beta": [SVC_B],
    }

    async def _get_mode(mode_id: str) -> Mode | None:
        return modes.get(mode_id)

    async def _get_mode_services(mode_id: str) -> list[Service]:
        return mode_services.get(mode_id, [])

    async def _get_current_mode() -> str | None:
        return current_mode

    async def _set_current_mode(mode_id: str) -> None:
        pass

    db.get_mode = AsyncMock(side_effect=_get_mode)
    db.get_mode_services = AsyncMock(side_effect=_get_mode_services)
    db.get_current_mode = AsyncMock(side_effect=_get_current_mode)
    db.set_current_mode = AsyncMock(side_effect=_set_current_mode)

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
        return []

    registry.get = AsyncMock(side_effect=_get)
    registry.list_all = AsyncMock(side_effect=_list_all)
    registry.list_running = AsyncMock(side_effect=_list_running)

    driver = AsyncMock()
    driver.status = AsyncMock(return_value=ServiceStatus(state=ServiceState.STOPPED))
    driver.supports_sleep = False
    registry.get_driver = lambda _dtype: driver

    return registry


def _build_mock_lifecycle() -> AsyncMock:
    """Build a mock LifecycleManager."""
    lifecycle = AsyncMock()
    lifecycle.start = AsyncMock()
    lifecycle.stop = AsyncMock()
    lifecycle.sleep = AsyncMock()
    lifecycle.wake = AsyncMock()
    return lifecycle


def _build_mock_vram() -> AsyncMock:
    """Build a mock VRAMTracker."""
    vram = AsyncMock()
    gpu_info = GPUInfo(name="RTX 4090", vram_total_mb=24000, driver="550.0")
    usage = VRAMUsage(total_mb=24000, used_mb=0, free_mb=24000)
    vram.get_gpu_info = AsyncMock(return_value=gpu_info)
    vram.get_usage = AsyncMock(return_value=usage)

    async def _estimate_service_vram(service: Service) -> int:
        return service.vram_mb

    vram.estimate_service_vram = AsyncMock(side_effect=_estimate_service_vram)
    vram.wait_for_vram_release = AsyncMock(return_value=True)
    return vram


def _build_mock_sleep() -> AsyncMock:
    """Build a mock SleepController."""
    return AsyncMock()


def _build_mock_ram() -> AsyncMock:
    """Build a mock RAMTracker with comfortable defaults."""
    ram = AsyncMock()
    ram.get_usage = AsyncMock(
        return_value=RAMUsage(total_mb=30_000, available_mb=28_000, free_mb=5_000)
    )
    ram.wait_for_ram_release = AsyncMock(return_value=True)
    return ram


def _build_manager() -> ServiceManager:
    """Build a ServiceManager with all mocks wired up."""
    return ServiceManager(
        db=_build_mock_db(),
        registry=_build_mock_registry(),
        lifecycle=_build_mock_lifecycle(),
        vram=_build_mock_vram(),
        sleep=_build_mock_sleep(),
        ram=_build_mock_ram(),
    )


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------


class TestSwitchModeConcurrency:
    """Verify asyncio.Lock serializes concurrent switch_mode calls."""

    async def test_concurrent_switch_mode_serializes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two concurrent switch_mode calls must run serially, not interleaved.

        We inject a small delay inside the lifecycle.start mock so
        that without a lock the two calls would overlap.  A sequence
        log records entry/exit of each call; serialization means the
        first call exits before the second enters.
        """
        monkeypatch.setenv("GPUMOD_SETTLE_SECONDS", "0")

        sequence: list[str] = []
        call_counter = 0

        db = _build_mock_db()
        registry = _build_mock_registry()
        lifecycle = _build_mock_lifecycle()
        vram = _build_mock_vram()

        original_get_mode = db.get_mode.side_effect

        async def _tracking_get_mode(mode_id: str) -> Mode | None:
            nonlocal call_counter
            call_counter += 1
            tag = call_counter
            sequence.append(f"enter-{tag}")
            result = await original_get_mode(mode_id)
            # Simulate work so that without a lock the tasks overlap
            await asyncio.sleep(0.05)
            sequence.append(f"exit-{tag}")
            return result

        db.get_mode = AsyncMock(side_effect=_tracking_get_mode)

        mgr = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=_build_mock_sleep(),
            ram=_build_mock_ram(),
        )

        r1, r2 = await asyncio.gather(
            mgr.switch_mode("alpha"),
            mgr.switch_mode("beta"),
        )

        assert isinstance(r1, ModeResult)
        assert isinstance(r2, ModeResult)
        assert r1.success is True
        assert r2.success is True

        # Serialization proof: first call must exit before second enters.
        # Sequence should be [enter-1, exit-1, enter-2, exit-2]
        # (not [enter-1, enter-2, ...])
        assert len(sequence) == 4
        assert sequence[0] == "enter-1"
        assert sequence[1] == "exit-1"
        assert sequence[2] == "enter-2"
        assert sequence[3] == "exit-2"

    async def test_concurrent_switch_does_not_deadlock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two concurrent switch_mode calls to the same mode must both
        succeed without deadlocking.
        """
        monkeypatch.setenv("GPUMOD_SETTLE_SECONDS", "0")

        mgr = _build_manager()

        r1, r2 = await asyncio.wait_for(
            asyncio.gather(
                mgr.switch_mode("alpha"),
                mgr.switch_mode("alpha"),
            ),
            timeout=5.0,
        )

        assert isinstance(r1, ModeResult)
        assert isinstance(r2, ModeResult)
        assert r1.success is True
        assert r2.success is True

    async def test_lock_released_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If switch_mode raises mid-flight, the lock must be released
        so the next call can proceed.
        """
        monkeypatch.setenv("GPUMOD_SETTLE_SECONDS", "0")

        db = _build_mock_db()
        registry = _build_mock_registry()
        lifecycle = _build_mock_lifecycle()
        vram = _build_mock_vram()

        call_count = 0

        async def _failing_then_ok(service_id: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise LifecycleError(service_id, "start", "Simulated failure")
            # Subsequent calls succeed

        lifecycle.start = AsyncMock(side_effect=_failing_then_ok)

        mgr = ServiceManager(
            db=db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=_build_mock_sleep(),
            ram=_build_mock_ram(),
        )

        # First call triggers LifecycleError inside switch_mode
        # (switch_mode catches it and returns ModeResult(success=False))
        r1 = await mgr.switch_mode("alpha")
        assert r1.success is False

        # Lock must be released: second call should succeed
        r2 = await asyncio.wait_for(mgr.switch_mode("beta"), timeout=5.0)
        assert isinstance(r2, ModeResult)
        assert r2.success is True
