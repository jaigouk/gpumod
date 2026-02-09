"""Tests for gpumod.services.registry — ServiceRegistry."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from gpumod.db import Database
from gpumod.models import DriverType, Service, ServiceState, ServiceStatus, SleepMode
from gpumod.services.base import ServiceDriver
from gpumod.services.drivers.fastapi import FastAPIDriver
from gpumod.services.drivers.llamacpp import LlamaCppDriver
from gpumod.services.drivers.vllm import VLLMDriver
from gpumod.services.registry import ServiceRegistry

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(
    id: str = "svc-1",
    name: str = "Test Service",
    driver: DriverType = DriverType.VLLM,
    port: int = 8000,
    vram_mb: int = 2500,
    sleep_mode: SleepMode = SleepMode.NONE,
    depends_on: list[str] | None = None,
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
        depends_on=depends_on or [],
        startup_timeout=60,
        extra_config={},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def populated_db(tmp_path: Path) -> Database:
    """Create a Database pre-populated with one service per driver type."""
    db = Database(tmp_path / "registry_test.db")
    await db.connect()

    vllm_svc = _make_service(
        id="vllm-embed",
        name="VLLM Embedding",
        driver=DriverType.VLLM,
        port=8001,
        vram_mb=2500,
    )
    llamacpp_svc = _make_service(
        id="qwen3-coder",
        name="Qwen3 Coder",
        driver=DriverType.LLAMACPP,
        port=7070,
        vram_mb=19000,
    )
    fastapi_svc = _make_service(
        id="qwen-asr",
        name="Qwen ASR",
        driver=DriverType.FASTAPI,
        port=9000,
        vram_mb=3000,
    )

    for svc in (vllm_svc, llamacpp_svc, fastapi_svc):
        await db.insert_service(svc)

    yield db  # type: ignore[misc]

    await db.close()


@pytest.fixture
def registry(populated_db: Database) -> ServiceRegistry:
    """Create a ServiceRegistry backed by the populated DB."""
    return ServiceRegistry(populated_db)


# ---------------------------------------------------------------------------
# list_all
# ---------------------------------------------------------------------------


class TestListAll:
    """Tests for ServiceRegistry.list_all()."""

    async def test_list_all_returns_all_services(self, registry: ServiceRegistry) -> None:
        """list_all() should return all services from the DB."""
        services = await registry.list_all()
        assert len(services) == 3
        ids = {s.id for s in services}
        assert ids == {"vllm-embed", "qwen3-coder", "qwen-asr"}


# ---------------------------------------------------------------------------
# list_running
# ---------------------------------------------------------------------------


class TestListRunning:
    """Tests for ServiceRegistry.list_running()."""

    async def test_list_running_filters_to_running_and_sleeping(
        self, registry: ServiceRegistry
    ) -> None:
        """list_running() should return only services in RUNNING or SLEEPING state."""
        # Mock drivers to return specific states
        vllm_driver = AsyncMock(spec=VLLMDriver)
        vllm_driver.status.return_value = ServiceStatus(state=ServiceState.RUNNING)
        llamacpp_driver = AsyncMock(spec=LlamaCppDriver)
        llamacpp_driver.status.return_value = ServiceStatus(state=ServiceState.SLEEPING)
        fastapi_driver = AsyncMock(spec=FastAPIDriver)
        fastapi_driver.status.return_value = ServiceStatus(state=ServiceState.STOPPED)

        registry._drivers = {
            DriverType.VLLM: vllm_driver,
            DriverType.LLAMACPP: llamacpp_driver,
            DriverType.FASTAPI: fastapi_driver,
        }

        running = await registry.list_running()

        assert len(running) == 2
        ids = {s.id for s in running}
        assert ids == {"vllm-embed", "qwen3-coder"}

    async def test_list_running_empty_when_all_stopped(self, registry: ServiceRegistry) -> None:
        """list_running() should return empty list when all services are stopped."""
        stopped_status = ServiceStatus(state=ServiceState.STOPPED)
        mock_driver = AsyncMock(spec=ServiceDriver)
        mock_driver.status.return_value = stopped_status

        registry._drivers = {
            DriverType.VLLM: mock_driver,
            DriverType.LLAMACPP: mock_driver,
            DriverType.FASTAPI: mock_driver,
        }

        running = await registry.list_running()
        assert running == []


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    """Tests for ServiceRegistry.get()."""

    async def test_get_returns_correct_service(self, registry: ServiceRegistry) -> None:
        """get() should return the service matching the given ID."""
        svc = await registry.get("vllm-embed")
        assert svc.id == "vllm-embed"
        assert svc.name == "VLLM Embedding"
        assert svc.driver == DriverType.VLLM

    async def test_get_raises_key_error_for_unknown(self, registry: ServiceRegistry) -> None:
        """get() should raise KeyError for a service ID not in the DB."""
        with pytest.raises(KeyError, match="nonexistent"):
            await registry.get("nonexistent")


# ---------------------------------------------------------------------------
# get_driver
# ---------------------------------------------------------------------------


class TestGetDriver:
    """Tests for ServiceRegistry.get_driver()."""

    def test_get_driver_returns_vllm_for_vllm_type(self, registry: ServiceRegistry) -> None:
        """get_driver() should return VLLMDriver for DriverType.VLLM."""
        driver = registry.get_driver(DriverType.VLLM)
        assert isinstance(driver, VLLMDriver)

    def test_get_driver_returns_llamacpp_for_llamacpp_type(
        self, registry: ServiceRegistry
    ) -> None:
        """get_driver() should return LlamaCppDriver for DriverType.LLAMACPP."""
        driver = registry.get_driver(DriverType.LLAMACPP)
        assert isinstance(driver, LlamaCppDriver)

    def test_get_driver_returns_fastapi_for_fastapi_type(self, registry: ServiceRegistry) -> None:
        """get_driver() should return FastAPIDriver for DriverType.FASTAPI."""
        driver = registry.get_driver(DriverType.FASTAPI)
        assert isinstance(driver, FastAPIDriver)

    def test_get_driver_raises_value_error_for_unknown_type(
        self, registry: ServiceRegistry
    ) -> None:
        """get_driver() should raise ValueError for an unmapped driver type."""
        # Remove a driver to simulate an unmapped type
        registry._drivers.pop(DriverType.DOCKER)
        with pytest.raises(ValueError, match="docker"):
            registry.get_driver(DriverType.DOCKER)


# ---------------------------------------------------------------------------
# get_dependents
# ---------------------------------------------------------------------------


class TestGetDependents:
    """Tests for ServiceRegistry.get_dependents()."""

    async def test_get_dependents_finds_dependent_services(self, populated_db: Database) -> None:
        """get_dependents() should return services whose depends_on includes the given ID."""
        # Add a service that depends on vllm-embed
        dependent_svc = _make_service(
            id="chat-svc",
            name="Chat Service",
            driver=DriverType.VLLM,
            port=8002,
            depends_on=["vllm-embed"],
        )
        await populated_db.insert_service(dependent_svc)

        registry = ServiceRegistry(populated_db)
        dependents = await registry.get_dependents("vllm-embed")

        assert len(dependents) == 1
        assert dependents[0].id == "chat-svc"

    async def test_get_dependents_returns_empty_for_no_dependents(
        self, registry: ServiceRegistry
    ) -> None:
        """get_dependents() should return empty list when nothing depends on the service."""
        dependents = await registry.get_dependents("vllm-embed")
        assert dependents == []


# ---------------------------------------------------------------------------
# register_service / unregister_service
# ---------------------------------------------------------------------------


class TestRegisterUnregister:
    """Tests for register_service() and unregister_service()."""

    async def test_register_service_adds_to_db(self, registry: ServiceRegistry) -> None:
        """register_service() should add the service to the DB."""
        new_svc = _make_service(
            id="new-svc",
            name="New Service",
            driver=DriverType.VLLM,
            port=8080,
        )
        await registry.register_service(new_svc)

        # Should now be retrievable
        svc = await registry.get("new-svc")
        assert svc.id == "new-svc"
        assert svc.name == "New Service"

    async def test_unregister_service_removes_from_db(self, registry: ServiceRegistry) -> None:
        """unregister_service() should remove the service from the DB."""
        await registry.unregister_service("vllm-embed")

        with pytest.raises(KeyError):
            await registry.get("vllm-embed")


# ---------------------------------------------------------------------------
# register_driver
# ---------------------------------------------------------------------------


class TestRegisterDriver:
    """Tests for register_driver()."""

    def test_register_driver_adds_custom_driver(self, registry: ServiceRegistry) -> None:
        """register_driver() should allow adding custom driver type mappings."""

        class CustomDriver(ServiceDriver):
            async def start(self, service: Service) -> None: ...
            async def stop(self, service: Service) -> None: ...
            async def status(self, service: Service) -> ServiceStatus: ...
            async def health_check(self, service: Service) -> bool: ...

        custom = CustomDriver()
        registry.register_driver(DriverType.DOCKER, custom)

        driver = registry.get_driver(DriverType.DOCKER)
        assert isinstance(driver, CustomDriver)
