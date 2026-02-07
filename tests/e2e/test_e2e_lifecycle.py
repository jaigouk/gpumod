"""E2E tests for service lifecycle, mode switching, and VRAM simulation.

These tests use real SQLite databases and exercise the full service
layer stack. GPU-dependent tests are marked with @pytest.mark.gpu_required
and skip gracefully on CPU-only machines.

CI configuration:
  CPU-only:  pytest tests/ -m "not gpu_required and not docker_required"
  GPU CI:    pytest tests/  (runs everything)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from gpumod.models import DriverType, GPUInfo, Service, ServiceState
from gpumod.registry import ModelRegistry
from gpumod.services.lifecycle import LifecycleManager
from gpumod.services.manager import ServiceManager
from gpumod.services.registry import ServiceRegistry
from gpumod.services.sleep import SleepController
from gpumod.services.vram import VRAMTracker
from gpumod.simulation import SimulationEngine

if TYPE_CHECKING:
    from gpumod.db import Database


# ---------------------------------------------------------------------------
# Mode switch lifecycle (no real GPU needed)
# ---------------------------------------------------------------------------


class TestModeSwitchLifecycle:
    """E2E: Mode switch exercises registry, lifecycle, and DB together."""

    async def test_switch_mode_nonexistent_mode_fails(self, e2e_db: Database) -> None:
        """Switching to a nonexistent mode returns failure."""
        registry = ServiceRegistry(e2e_db)
        lifecycle = LifecycleManager(registry)
        vram = VRAMTracker()
        sleep = SleepController(registry)
        manager = ServiceManager(
            db=e2e_db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep,
        )

        result = await manager.switch_mode("nonexistent-mode")
        assert not result.success

    async def test_list_services_returns_all(self, e2e_db: Database) -> None:
        """All inserted services are retrievable."""
        services = await e2e_db.list_services()
        assert len(services) == 3
        ids = {s.id for s in services}
        assert ids == {"vllm-chat", "vllm-embed", "fastapi-app"}

    async def test_list_modes_returns_all(self, e2e_db: Database) -> None:
        """All inserted modes are retrievable."""
        modes = await e2e_db.list_modes()
        assert len(modes) == 2
        ids = {m.id for m in modes}
        assert ids == {"chat", "embed"}

    async def test_get_mode_services_returns_list(self, e2e_db: Database) -> None:
        """Mode services are retrievable via get_mode_services."""
        services = await e2e_db.get_mode_services("chat")
        ids = {s.id for s in services}
        assert "vllm-chat" in ids
        assert "fastapi-app" in ids

    async def test_get_service_returns_details(self, e2e_db: Database) -> None:
        """Getting a service returns full details."""
        svc = await e2e_db.get_service("vllm-chat")
        assert svc is not None
        assert svc.name == "vLLM Chat"
        assert svc.driver == DriverType.VLLM
        assert svc.vram_mb == 8000


# ---------------------------------------------------------------------------
# VRAM simulation (no real GPU needed)
# ---------------------------------------------------------------------------


class TestVRAMSimulation:
    """E2E: VRAM simulation with real DB and mocked GPU info."""

    async def test_simulate_mode_returns_result(self, e2e_db: Database) -> None:
        """Simulation returns a result with VRAM calculations."""
        mock_vram = AsyncMock(spec=VRAMTracker)
        mock_vram.get_gpu_info = AsyncMock(
            return_value=GPUInfo(name="RTX 4090", vram_total_mb=24576)
        )
        mock_vram.estimate_service_vram = AsyncMock(side_effect=lambda svc: svc.vram_mb)

        model_reg = ModelRegistry(e2e_db)
        engine = SimulationEngine(db=e2e_db, vram=mock_vram, model_registry=model_reg)

        result = await engine.simulate_mode("chat")

        assert result.gpu_total_mb == 24576
        assert result.proposed_usage_mb > 0
        assert len(result.services) > 0

    async def test_simulate_mode_nonexistent_raises(self, e2e_db: Database) -> None:
        """Simulating a nonexistent mode raises ValueError."""
        mock_vram = AsyncMock(spec=VRAMTracker)
        mock_vram.get_gpu_info = AsyncMock(
            return_value=GPUInfo(name="RTX 4090", vram_total_mb=24576)
        )

        model_reg = ModelRegistry(e2e_db)
        engine = SimulationEngine(db=e2e_db, vram=mock_vram, model_registry=model_reg)

        with pytest.raises(ValueError, match="not found"):
            await engine.simulate_mode("nonexistent-mode")

    async def test_simulate_mode_fits_check(self, e2e_db: Database) -> None:
        """Simulation correctly determines if services fit in VRAM."""
        mock_vram = AsyncMock(spec=VRAMTracker)
        mock_vram.get_gpu_info = AsyncMock(
            return_value=GPUInfo(name="RTX 4090", vram_total_mb=24576)
        )
        mock_vram.estimate_service_vram = AsyncMock(side_effect=lambda svc: svc.vram_mb)

        model_reg = ModelRegistry(e2e_db)
        engine = SimulationEngine(db=e2e_db, vram=mock_vram, model_registry=model_reg)

        result = await engine.simulate_mode("chat")

        # chat mode: vllm-chat (8000) + fastapi-app (1000) = 9000 < 24576
        assert result.fits


# ---------------------------------------------------------------------------
# Service status (no real systemd — exercises registry and DB)
# ---------------------------------------------------------------------------


class TestServiceStatus:
    """E2E: Service status through the full stack."""

    async def test_get_status_returns_system_status(self, e2e_db: Database) -> None:
        """get_status returns a full SystemStatus with all services."""
        registry = ServiceRegistry(e2e_db)
        lifecycle = LifecycleManager(registry)
        vram = VRAMTracker()
        sleep = SleepController(registry)
        manager = ServiceManager(
            db=e2e_db,
            registry=registry,
            lifecycle=lifecycle,
            vram=vram,
            sleep=sleep,
        )

        status = await manager.get_status()
        assert status is not None
        assert isinstance(status.services, list)
        assert len(status.services) == 3

    async def test_service_registry_lookup(self, e2e_db: Database) -> None:
        """ServiceRegistry can look up services by ID."""
        registry = ServiceRegistry(e2e_db)
        svc = await registry.get("vllm-chat")
        assert svc.name == "vLLM Chat"
        assert svc.driver == DriverType.VLLM

    async def test_service_registry_driver_lookup(self, e2e_db: Database) -> None:
        """ServiceRegistry returns the correct driver for each type."""
        registry = ServiceRegistry(e2e_db)
        for driver_type in DriverType:
            driver = registry.get_driver(driver_type)
            assert driver is not None


# ---------------------------------------------------------------------------
# GPU-dependent tests (skipped on CPU-only machines)
# ---------------------------------------------------------------------------


@pytest.mark.gpu_required
class TestGPUDetection:
    """E2E: Real GPU detection via nvidia-smi."""

    async def test_vram_tracker_detects_real_gpu(self) -> None:
        """VRAMTracker returns real GPU info on GPU machines."""
        vram = VRAMTracker()
        gpu_info = await vram.get_gpu_info()
        assert gpu_info is not None
        assert gpu_info.vram_total_mb > 0
        assert len(gpu_info.name) > 0


# ---------------------------------------------------------------------------
# Docker-dependent tests (skipped when Docker unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.docker_required
class TestDockerIntegration:
    """E2E: Docker driver with real Docker daemon."""

    async def test_docker_driver_status_for_missing_container(self) -> None:
        """DockerDriver reports stopped for a non-existent container."""
        from gpumod.services.drivers.docker import DockerDriver

        driver = DockerDriver()
        svc = Service(
            id="e2e-test-container",
            name="E2E Test",
            driver=DriverType.DOCKER,
            port=8888,
            vram_mb=0,
            extra_config={"image": "alpine:3.19"},
        )

        status = await driver.status(svc)
        assert status.state == ServiceState.STOPPED


# ---------------------------------------------------------------------------
# Cleanup verification
# ---------------------------------------------------------------------------


class TestE2ECleanup:
    """E2E: Verify test fixtures clean up properly."""

    async def test_db_connection_usable(self, e2e_db: Database) -> None:
        """DB connection is usable during test."""
        services = await e2e_db.list_services()
        assert len(services) > 0

    async def test_temp_db_is_real_file(self, e2e_db: Database) -> None:
        """E2E database is a real SQLite file, not in-memory."""
        assert e2e_db._db_path is not None
        assert e2e_db._db_path.suffix == ".db"
