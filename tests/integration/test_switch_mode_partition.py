"""Integration tests for ServiceManager.switch_mode state partitioning (gpumod-hwbb).

Unlike the unit tests in tests/unit/test_manager.py (which mock Registry,
drivers, and LifecycleManager), these tests exercise the REAL pipeline:

    ServiceManager.switch_mode
        -> ServiceRegistry.list_running        (real)
        -> LlamaCppDriver / VLLMDriver.status  (real)
        -> state partition (manager.py)        (under test)
        -> LifecycleManager.start/wake/stop    (real)

Only the host I/O boundary is stubbed:

- systemd (systemctl subprocess calls) -> ``FakeHost`` unit states
- localhost HTTP (driver /health, /models, /is_sleeping, ...) ->
  ``httpx.MockTransport`` routed into the same ``FakeHost``
- nvidia-smi -> mocked ``VRAMTracker`` (same pattern as conftest.py)
- /proc/meminfo -> real ``RAMTracker`` pointed at a fake meminfo file

Scenarios (from gpumod-hwbb / gpumod-hrgg close-out review):

- R1: all target services STOPPED -> all started
- R2: partial RUNNING -> only the missing service started
- R3: SLEEPING service in target mode -> woken via wake(), not start()
- R4: orphan SLEEPING service not in target -> stopped (gpumod-77o)
- R5: unit enters systemd 'failed' on start -> surfaces in ModeResult.errors

Mutation acceptance (gpumod-hwbb, verified 2026-06-05; mutations run locally,
never committed) against manager.py's ``to_start`` derivation:

- ``target_service_ids - current_service_ids``      -> R1, R2, R3 fail (6 tests)
- ``target_service_ids - running_or_sleeping_ids``  -> R3 fails (3 tests)
  (the pre-gpumod-hrgg behavior: sleeping targets never woken)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import httpx
import pytest

from gpumod.config import _clear_settings_cache
from gpumod.db import Database
from gpumod.models import (
    DriverType,
    GPUInfo,
    Mode,
    Service,
    ServiceState,
    SleepMode,
)
from gpumod.services.lifecycle import LifecycleManager
from gpumod.services.manager import ServiceManager
from gpumod.services.ram import RAMTracker
from gpumod.services.registry import ServiceRegistry
from gpumod.services.sleep import SleepController
from gpumod.services.vram import VRAMTracker

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Generator
    from pathlib import Path

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Stubbed host: systemd unit states + localhost HTTP, shared state
# ---------------------------------------------------------------------------


@dataclass
class FakeUnit:
    """Simulated runtime state of one systemd unit + its HTTP server.

    State combinations map to driver-visible states:

    - STOPPED:  unit_state="inactive"
    - RUNNING (llamacpp):  active + healthy + model_loaded
    - SLEEPING (llamacpp): active + healthy + not model_loaded
    - RUNNING (vllm):      active + healthy + not vllm_sleeping
    - SLEEPING (vllm):     active + healthy + vllm_sleeping
    - FAILED on start:     fail_on_start=True (start() lands in "failed")
    """

    unit_state: str = "inactive"  # active | inactive | failed
    healthy: bool = False
    model_loaded: bool = False
    vllm_sleeping: bool = False
    fail_on_start: bool = False


class FakeHost:
    """Stub for the systemd + localhost-HTTP I/O boundary.

    The real drivers talk to systemctl (via gpumod.services.systemd) and to
    their service's HTTP API (via httpx). Both surfaces are routed here so
    a single per-unit state drives consistent answers on both.
    """

    def __init__(self) -> None:
        self._by_unit: dict[str, FakeUnit] = {}
        self._by_port: dict[int, FakeUnit] = {}
        self.systemd_calls: list[tuple[str, str]] = []
        self.http_calls: list[tuple[int, str]] = []

    def add_service(self, service: Service, unit: FakeUnit) -> FakeUnit:
        assert service.unit_name is not None
        assert service.port is not None
        self._by_unit[service.unit_name] = unit
        self._by_port[service.port] = unit
        return unit

    def unit(self, unit_name: str) -> FakeUnit:
        return self._by_unit[unit_name]

    def started_units(self) -> set[str]:
        return {unit for cmd, unit in self.systemd_calls if cmd == "start"}

    def stopped_units(self) -> set[str]:
        return {unit for cmd, unit in self.systemd_calls if cmd == "stop"}

    # -- systemd boundary ---------------------------------------------------

    async def is_active(self, unit: str) -> bool:
        self.systemd_calls.append(("is-active", unit))
        return self._unit(unit).unit_state == "active"

    async def start(self, unit: str, *, timeout_s: float = 30.0) -> None:
        self.systemd_calls.append(("start", unit))
        fake = self._unit(unit)
        if fake.fail_on_start:
            fake.unit_state = "failed"
            fake.healthy = False
        else:
            fake.unit_state = "active"
            fake.healthy = True

    async def stop(self, unit: str, *, timeout_s: float = 30.0) -> None:
        self.systemd_calls.append(("stop", unit))
        fake = self._unit(unit)
        fake.unit_state = "inactive"
        fake.healthy = False
        fake.model_loaded = False

    async def get_unit_state(self, unit: str) -> str:
        return self._unit(unit).unit_state

    async def journal_logs(self, unit: str, lines: int = 20) -> list[str]:
        return []

    def _unit(self, unit: str) -> FakeUnit:
        return self._by_unit.setdefault(unit, FakeUnit())

    # -- HTTP boundary --------------------------------------------------------

    def handle_http(self, request: httpx.Request) -> httpx.Response:
        port = request.url.port or 80
        path = request.url.path
        self.http_calls.append((port, path))
        fake = self._by_port.get(port)
        if fake is None or fake.unit_state != "active" or not fake.healthy:
            msg = "connection refused (fake host: no server on port)"
            raise httpx.ConnectError(msg, request=request)
        handler = _HTTP_ROUTES.get(path)
        if handler is None:
            return httpx.Response(404)
        return handler(fake)


# Per-endpoint handlers simulating the vLLM / llama.cpp router HTTP APIs.


def _http_health(fake: FakeUnit) -> httpx.Response:
    del fake
    return httpx.Response(200)


def _http_is_sleeping(fake: FakeUnit) -> httpx.Response:
    return httpx.Response(200, json={"is_sleeping": fake.vllm_sleeping})


def _http_wake_up(fake: FakeUnit) -> httpx.Response:
    fake.vllm_sleeping = False
    return httpx.Response(200)


def _http_sleep(fake: FakeUnit) -> httpx.Response:
    fake.vllm_sleeping = True
    return httpx.Response(200)


def _http_models(fake: FakeUnit) -> httpx.Response:
    status = "loaded" if fake.model_loaded else "unloaded"
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [{"id": "test-model", "status": {"value": status}}],
        },
    )


def _http_models_load(fake: FakeUnit) -> httpx.Response:
    fake.model_loaded = True
    return httpx.Response(200)


def _http_models_unload(fake: FakeUnit) -> httpx.Response:
    fake.model_loaded = False
    return httpx.Response(200)


_HTTP_ROUTES: dict[str, Callable[[FakeUnit], httpx.Response]] = {
    "/health": _http_health,
    "/is_sleeping": _http_is_sleeping,
    "/wake_up": _http_wake_up,
    "/sleep": _http_sleep,
    "/models": _http_models,
    "/models/load": _http_models_load,
    "/models/unload": _http_models_unload,
}


# ---------------------------------------------------------------------------
# Services and modes under test
# ---------------------------------------------------------------------------

LLAMA_ID = "llm-router"
LLAMA_UNIT = "llm-router.service"
LLAMA_PORT = 18101

VLLM_ID = "vllm-chat"
VLLM_UNIT = "vllm-chat.service"
VLLM_PORT = 18102

ORPHAN_ID = "orphan-embed"
ORPHAN_UNIT = "orphan-embed.service"
ORPHAN_PORT = 18103

TARGET_MODE = "agent"
LEGACY_MODE = "legacy"


def _make_services() -> dict[str, Service]:
    """Build fresh Service instances (vram_mb=0 keeps preflight hermetic)."""
    return {
        LLAMA_ID: Service(
            id=LLAMA_ID,
            name="Llama Router",
            driver=DriverType.LLAMACPP,
            port=LLAMA_PORT,
            vram_mb=0,
            sleep_mode=SleepMode.ROUTER,
            unit_name=LLAMA_UNIT,
        ),
        VLLM_ID: Service(
            id=VLLM_ID,
            name="vLLM Chat",
            driver=DriverType.VLLM,
            port=VLLM_PORT,
            vram_mb=0,
            sleep_mode=SleepMode.L1,
            unit_name=VLLM_UNIT,
        ),
        ORPHAN_ID: Service(
            id=ORPHAN_ID,
            name="Orphan Embedding",
            driver=DriverType.LLAMACPP,
            port=ORPHAN_PORT,
            vram_mb=0,
            sleep_mode=SleepMode.ROUTER,
            unit_name=ORPHAN_UNIT,
        ),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """No settle sleep, no RAM-threshold flakiness from the host machine."""
    monkeypatch.setenv("GPUMOD_SETTLE_SECONDS", "0")
    monkeypatch.setenv("GPUMOD_RAM_MIN_FREE_MB", "0")
    _clear_settings_cache()
    yield
    _clear_settings_cache()


@pytest.fixture
def fake_host(monkeypatch: pytest.MonkeyPatch) -> FakeHost:
    """Patch the systemd + HTTP I/O boundary; everything above it is real."""
    import gpumod.services.lifecycle as lifecycle_mod
    import gpumod.services.systemd as systemd_mod

    host = FakeHost()

    # Drivers call these as module attributes: systemd.is_active(...) etc.
    monkeypatch.setattr(systemd_mod, "is_active", host.is_active)
    monkeypatch.setattr(systemd_mod, "start", host.start)
    monkeypatch.setattr(systemd_mod, "stop", host.stop)
    monkeypatch.setattr(systemd_mod, "get_unit_state", host.get_unit_state)
    monkeypatch.setattr(systemd_mod, "journal_logs", host.journal_logs)
    # LifecycleManager imports these BY NAME, so patch its bound references too.
    monkeypatch.setattr(lifecycle_mod, "get_unit_state", host.get_unit_state)
    monkeypatch.setattr(lifecycle_mod, "journal_logs", host.journal_logs)

    # Route every httpx.AsyncClient through the fake host's transport.
    real_async_client = httpx.AsyncClient

    def _patched_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(host.handle_http)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _patched_client)
    return host


@pytest.fixture
async def partition_db(tmp_path: Path) -> AsyncGenerator[Database, None]:
    """Real Database populated with the partition-test services and modes."""
    db = Database(tmp_path / "partition.db")
    await db.connect()

    services = _make_services()
    for svc in services.values():
        await db.insert_service(svc)

    await db.insert_mode(Mode(id=TARGET_MODE, name="Agent Mode"))
    await db.set_mode_services(TARGET_MODE, [LLAMA_ID, VLLM_ID])
    await db.insert_mode(Mode(id=LEGACY_MODE, name="Legacy Mode"))
    await db.set_mode_services(LEGACY_MODE, [ORPHAN_ID])

    yield db
    await db.close()


@pytest.fixture
def fake_meminfo(tmp_path: Path) -> Path:
    """Fake /proc/meminfo with plenty of MemAvailable (64 GiB)."""
    path = tmp_path / "meminfo"
    path.write_text(
        "MemTotal:       131072000 kB\n"
        "MemFree:         67108864 kB\n"
        "MemAvailable:    67108864 kB\n"
    )
    return path


@pytest.fixture
def registry(partition_db: Database) -> ServiceRegistry:
    """Real ServiceRegistry with real drivers."""
    return ServiceRegistry(partition_db)


@pytest.fixture
def manager(
    partition_db: Database,
    registry: ServiceRegistry,
    fake_host: FakeHost,
    fake_meminfo: Path,
) -> ServiceManager:
    """Real ServiceManager: real Registry, drivers, and LifecycleManager.

    Only nvidia-smi (VRAMTracker) is mocked, matching the existing
    integration conftest pattern; RAM reads go to the fake meminfo file.
    """
    lifecycle = LifecycleManager(registry, db=partition_db)
    vram = AsyncMock(spec=VRAMTracker)
    vram.get_gpu_info.return_value = GPUInfo(
        name="RTX 4090", vram_total_mb=24576, driver="535.129.03"
    )
    vram.estimate_service_vram.side_effect = lambda svc: svc.vram_mb
    vram.wait_for_vram_release.return_value = True
    return ServiceManager(
        db=partition_db,
        registry=registry,
        lifecycle=lifecycle,
        vram=vram,
        sleep=SleepController(registry),
        ram=RAMTracker(meminfo_path=fake_meminfo),
    )


async def _setup_unit_states(
    fake_host: FakeHost,
    partition_db: Database,
    states: dict[str, FakeUnit],
) -> None:
    """Register every service with the fake host using the given unit states."""
    for svc_id, unit in states.items():
        svc = await partition_db.get_service(svc_id)
        assert svc is not None, f"service {svc_id!r} missing from test DB"
        fake_host.add_service(svc, unit)


def _running_llamacpp() -> FakeUnit:
    return FakeUnit(unit_state="active", healthy=True, model_loaded=True)


def _sleeping_llamacpp() -> FakeUnit:
    return FakeUnit(unit_state="active", healthy=True, model_loaded=False)


def _running_vllm() -> FakeUnit:
    return FakeUnit(unit_state="active", healthy=True, vllm_sleeping=False)


def _sleeping_vllm() -> FakeUnit:
    return FakeUnit(unit_state="active", healthy=True, vllm_sleeping=True)


def _stopped() -> FakeUnit:
    return FakeUnit(unit_state="inactive")


# ---------------------------------------------------------------------------
# R1: all target services STOPPED -> all in to_start, lifecycle.start each
# ---------------------------------------------------------------------------


class TestR1AllStopped:
    """DB current_mode == target but nothing runs (gpumod-hrgg drift case).

    The naive diff ``target - current_mode_services`` would be empty here;
    only the real partition (``target - actively_running``) starts services.
    """

    async def test_all_target_services_started(
        self,
        manager: ServiceManager,
        partition_db: Database,
        fake_host: FakeHost,
    ) -> None:
        await _setup_unit_states(
            fake_host,
            partition_db,
            {LLAMA_ID: _stopped(), VLLM_ID: _stopped(), ORPHAN_ID: _stopped()},
        )
        await partition_db.set_current_mode(TARGET_MODE)

        result = await manager.switch_mode(TARGET_MODE)

        assert result.success is True
        assert set(result.started) == {LLAMA_ID, VLLM_ID}
        assert result.stopped == []
        # Both services were started through the real systemd boundary.
        assert fake_host.started_units() == {LLAMA_UNIT, VLLM_UNIT}
        assert fake_host.stopped_units() == set()

    async def test_started_services_report_running(
        self,
        manager: ServiceManager,
        partition_db: Database,
        fake_host: FakeHost,
        registry: ServiceRegistry,
    ) -> None:
        await _setup_unit_states(
            fake_host,
            partition_db,
            {LLAMA_ID: _stopped(), VLLM_ID: _stopped(), ORPHAN_ID: _stopped()},
        )
        await partition_db.set_current_mode(TARGET_MODE)

        await manager.switch_mode(TARGET_MODE)

        # End-to-end: real drivers now derive RUNNING from the stubbed host.
        for svc_id in (LLAMA_ID, VLLM_ID):
            svc = await registry.get(svc_id)
            driver = registry.get_driver(svc.driver)
            status = await driver.status(svc)
            assert status.state == ServiceState.RUNNING, svc_id


# ---------------------------------------------------------------------------
# R2: partial RUNNING -> only the missing service is in to_start
# ---------------------------------------------------------------------------


class TestR2PartialRunning:
    """One target service runs, the other drifted away -> start only the gap."""

    async def test_only_missing_service_started(
        self,
        manager: ServiceManager,
        partition_db: Database,
        fake_host: FakeHost,
    ) -> None:
        await _setup_unit_states(
            fake_host,
            partition_db,
            {
                LLAMA_ID: _running_llamacpp(),
                VLLM_ID: _stopped(),
                ORPHAN_ID: _stopped(),
            },
        )
        await partition_db.set_current_mode(TARGET_MODE)

        result = await manager.switch_mode(TARGET_MODE)

        assert result.success is True
        assert set(result.started) == {VLLM_ID}
        assert result.stopped == []
        # The already-running llamacpp service is untouched.
        assert fake_host.started_units() == {VLLM_UNIT}
        assert fake_host.stopped_units() == set()


# ---------------------------------------------------------------------------
# R3: SLEEPING service in target mode -> woken via wake(), NOT start()
# ---------------------------------------------------------------------------


class TestR3SleepingTargetIsWoken:
    """A sleeping target service is in to_start but routed to wake()."""

    async def test_sleeping_vllm_woken_not_started(
        self,
        manager: ServiceManager,
        partition_db: Database,
        fake_host: FakeHost,
    ) -> None:
        await _setup_unit_states(
            fake_host,
            partition_db,
            {
                LLAMA_ID: _running_llamacpp(),
                VLLM_ID: _sleeping_vllm(),
                ORPHAN_ID: _stopped(),
            },
        )
        await partition_db.set_current_mode(TARGET_MODE)

        result = await manager.switch_mode(TARGET_MODE)

        assert result.success is True
        assert set(result.started) == {VLLM_ID}
        # Woken via the vLLM HTTP API, NOT via systemctl start.
        assert (VLLM_PORT, "/wake_up") in fake_host.http_calls
        assert fake_host.started_units() == set()
        assert fake_host.stopped_units() == set()

    async def test_woken_service_no_longer_sleeping(
        self,
        manager: ServiceManager,
        partition_db: Database,
        fake_host: FakeHost,
        registry: ServiceRegistry,
    ) -> None:
        await _setup_unit_states(
            fake_host,
            partition_db,
            {
                LLAMA_ID: _running_llamacpp(),
                VLLM_ID: _sleeping_vllm(),
                ORPHAN_ID: _stopped(),
            },
        )
        await partition_db.set_current_mode(TARGET_MODE)

        await manager.switch_mode(TARGET_MODE)

        svc = await registry.get(VLLM_ID)
        driver = registry.get_driver(svc.driver)
        status = await driver.status(svc)
        assert status.state == ServiceState.RUNNING

    async def test_sleeping_llamacpp_woken_via_model_load(
        self,
        manager: ServiceManager,
        partition_db: Database,
        fake_host: FakeHost,
    ) -> None:
        """Same partition path through the llama.cpp router-mode driver."""
        await _setup_unit_states(
            fake_host,
            partition_db,
            {
                LLAMA_ID: _sleeping_llamacpp(),
                VLLM_ID: _running_vllm(),
                ORPHAN_ID: _stopped(),
            },
        )
        await partition_db.set_current_mode(TARGET_MODE)

        result = await manager.switch_mode(TARGET_MODE)

        assert result.success is True
        assert set(result.started) == {LLAMA_ID}
        assert (LLAMA_PORT, "/models/load") in fake_host.http_calls
        assert fake_host.started_units() == set()
        assert fake_host.unit(LLAMA_UNIT).model_loaded is True


# ---------------------------------------------------------------------------
# R4: orphan SLEEPING service not in target -> in to_stop (gpumod-77o)
# ---------------------------------------------------------------------------


class TestR4OrphanSleepingStopped:
    """A sleeping service from a prior mode must be stopped to free VRAM."""

    async def test_orphan_sleeping_service_stopped(
        self,
        manager: ServiceManager,
        partition_db: Database,
        fake_host: FakeHost,
    ) -> None:
        await _setup_unit_states(
            fake_host,
            partition_db,
            {
                LLAMA_ID: _stopped(),
                VLLM_ID: _stopped(),
                ORPHAN_ID: _sleeping_llamacpp(),
            },
        )
        await partition_db.set_current_mode(LEGACY_MODE)

        result = await manager.switch_mode(TARGET_MODE)

        assert result.success is True
        assert ORPHAN_ID in result.stopped
        assert set(result.started) == {LLAMA_ID, VLLM_ID}
        # The orphan was stopped through the real systemd boundary.
        assert ORPHAN_UNIT in fake_host.stopped_units()

    async def test_orphan_from_no_current_mode_still_stopped(
        self,
        manager: ServiceManager,
        partition_db: Database,
        fake_host: FakeHost,
        registry: ServiceRegistry,
    ) -> None:
        """Orphan detection must not depend on DB current_mode being set."""
        await _setup_unit_states(
            fake_host,
            partition_db,
            {
                LLAMA_ID: _stopped(),
                VLLM_ID: _stopped(),
                ORPHAN_ID: _sleeping_llamacpp(),
            },
        )
        # current_mode is None (fresh DB) but the orphan is alive on the host.

        result = await manager.switch_mode(TARGET_MODE)

        assert result.success is True
        assert ORPHAN_ID in result.stopped
        svc = await registry.get(ORPHAN_ID)
        driver = registry.get_driver(svc.driver)
        status = await driver.status(svc)
        assert status.state == ServiceState.STOPPED


# ---------------------------------------------------------------------------
# R5: unit lands in systemd 'failed' on start -> ModeResult.errors
# ---------------------------------------------------------------------------


class TestR5FailedStartSurfacesErrors:
    """A unit that enters 'failed' on start must surface in ModeResult.errors."""

    async def test_failed_unit_reported_in_errors(
        self,
        manager: ServiceManager,
        partition_db: Database,
        fake_host: FakeHost,
    ) -> None:
        await _setup_unit_states(
            fake_host,
            partition_db,
            {
                LLAMA_ID: _stopped(),
                VLLM_ID: FakeUnit(fail_on_start=True),
                ORPHAN_ID: _stopped(),
            },
        )
        await partition_db.set_current_mode(LEGACY_MODE)

        result = await manager.switch_mode(TARGET_MODE)

        assert result.success is False
        assert result.errors, "expected the failed start to surface in errors"
        original = result.errors[0]
        assert original.startswith("[original]")
        assert VLLM_ID in original
        assert "failed" in original

    async def test_failed_switch_does_not_update_mode_and_rolls_back(
        self,
        manager: ServiceManager,
        partition_db: Database,
        fake_host: FakeHost,
    ) -> None:
        await _setup_unit_states(
            fake_host,
            partition_db,
            {
                LLAMA_ID: _stopped(),
                VLLM_ID: FakeUnit(fail_on_start=True),
                ORPHAN_ID: _stopped(),
            },
        )
        await partition_db.set_current_mode(LEGACY_MODE)

        result = await manager.switch_mode(TARGET_MODE)

        assert result.success is False
        # DB current_mode is untouched on failure.
        assert await partition_db.get_current_mode() == LEGACY_MODE
        # llm-router started first (sorted order), then rolled back via stop.
        assert LLAMA_UNIT in fake_host.started_units()
        assert LLAMA_UNIT in fake_host.stopped_units()
