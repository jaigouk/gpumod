"""Tests for gpumod.services.sleep — SleepController and SleepError."""

from __future__ import annotations

from unittest.mock import AsyncMock, PropertyMock

import pytest

from gpumod.models import DriverType, Service, ServiceState, ServiceStatus, SleepMode
from gpumod.services.base import ServiceDriver
from gpumod.services.sleep import SleepController, SleepError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(
    id: str,
    name: str,
    driver: DriverType = DriverType.VLLM,
    vram_mb: int = 2500,
    sleep_mode: SleepMode = SleepMode.L1,
) -> Service:
    return Service(
        id=id,
        name=name,
        driver=driver,
        port=8000,
        vram_mb=vram_mb,
        sleep_mode=sleep_mode,
        health_endpoint="/health",
        model_id="org/model",
        unit_name=f"{id}.service",
        depends_on=[],
        startup_timeout=60,
        extra_config={},
    )


def _make_driver(supports_sleep: bool = True) -> AsyncMock:
    """Create a mock ServiceDriver with configurable sleep support."""
    driver = AsyncMock(spec=ServiceDriver)
    type(driver).supports_sleep = PropertyMock(return_value=supports_sleep)
    return driver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SVC_VLLM = _make_service("svc-vllm", "VLLM Service", DriverType.VLLM, 4000, SleepMode.L1)
SVC_LLAMACPP = _make_service(
    "svc-llamacpp", "LlamaCpp Service", DriverType.LLAMACPP, 19000, SleepMode.ROUTER
)
SVC_FASTAPI = _make_service(
    "svc-fastapi", "FastAPI Service", DriverType.FASTAPI, 3000, SleepMode.NONE
)


@pytest.fixture
def mock_registry() -> AsyncMock:
    """Build a mock ServiceRegistry with 3 services.

    - svc-vllm:     supports_sleep=True,  state=RUNNING
    - svc-llamacpp: supports_sleep=True,  state=SLEEPING
    - svc-fastapi:  supports_sleep=False, state=RUNNING
    """
    registry = AsyncMock()

    # -- services -------------------------------------------------------
    services = {
        "svc-vllm": SVC_VLLM,
        "svc-llamacpp": SVC_LLAMACPP,
        "svc-fastapi": SVC_FASTAPI,
    }

    async def _get(service_id: str) -> Service:
        if service_id not in services:
            raise KeyError(f"Service not found: {service_id!r}")
        return services[service_id]

    registry.get = AsyncMock(side_effect=_get)
    registry.list_all = AsyncMock(return_value=list(services.values()))

    # -- drivers --------------------------------------------------------
    vllm_driver = _make_driver(supports_sleep=True)
    vllm_driver.status = AsyncMock(return_value=ServiceStatus(state=ServiceState.RUNNING))

    llamacpp_driver = _make_driver(supports_sleep=True)
    llamacpp_driver.status = AsyncMock(return_value=ServiceStatus(state=ServiceState.SLEEPING))

    fastapi_driver = _make_driver(supports_sleep=False)
    fastapi_driver.status = AsyncMock(return_value=ServiceStatus(state=ServiceState.RUNNING))

    driver_map = {
        DriverType.VLLM: vllm_driver,
        DriverType.LLAMACPP: llamacpp_driver,
        DriverType.FASTAPI: fastapi_driver,
    }
    registry.get_driver = lambda dt: driver_map[dt]

    return registry


@pytest.fixture
def controller(mock_registry: AsyncMock) -> SleepController:
    return SleepController(mock_registry)


# ---------------------------------------------------------------------------
# sleep() tests
# ---------------------------------------------------------------------------


class TestSleep:
    """Tests for SleepController.sleep()."""

    async def test_sleep_calls_driver_sleep_with_correct_level(
        self, controller: SleepController, mock_registry: AsyncMock
    ) -> None:
        """sleep() should call driver.sleep(service, level) for a RUNNING service."""
        await controller.sleep("svc-vllm", level="l2")

        driver = mock_registry.get_driver(DriverType.VLLM)
        driver.sleep.assert_awaited_once_with(SVC_VLLM, "l2")

    async def test_sleep_is_noop_when_already_sleeping(
        self, controller: SleepController, mock_registry: AsyncMock
    ) -> None:
        """sleep() should be a no-op (idempotent) when service is already SLEEPING."""
        await controller.sleep("svc-llamacpp")

        driver = mock_registry.get_driver(DriverType.LLAMACPP)
        driver.sleep.assert_not_awaited()

    async def test_sleep_raises_for_non_running_service(
        self, controller: SleepController, mock_registry: AsyncMock
    ) -> None:
        """sleep() should raise SleepError when service state is not RUNNING (e.g. STOPPED)."""
        # Override vllm driver to return STOPPED
        driver = mock_registry.get_driver(DriverType.VLLM)
        driver.status.return_value = ServiceStatus(state=ServiceState.STOPPED)

        with pytest.raises(SleepError):
            await controller.sleep("svc-vllm")

    async def test_sleep_raises_for_unsupported_driver(
        self, controller: SleepController, mock_registry: AsyncMock
    ) -> None:
        """sleep() should raise SleepError when driver doesn't support sleep."""
        with pytest.raises(SleepError):
            await controller.sleep("svc-fastapi")

    async def test_sleep_raises_key_error_for_nonexistent_service(
        self, controller: SleepController
    ) -> None:
        """sleep() should raise KeyError for a service that doesn't exist."""
        with pytest.raises(KeyError):
            await controller.sleep("svc-nonexistent")


# ---------------------------------------------------------------------------
# wake() tests
# ---------------------------------------------------------------------------


class TestWake:
    """Tests for SleepController.wake()."""

    async def test_wake_calls_driver_wake(
        self, controller: SleepController, mock_registry: AsyncMock
    ) -> None:
        """wake() should call driver.wake(service) for a SLEEPING service."""
        await controller.wake("svc-llamacpp")

        driver = mock_registry.get_driver(DriverType.LLAMACPP)
        driver.wake.assert_awaited_once_with(SVC_LLAMACPP)

    async def test_wake_raises_for_non_sleeping_service(self, controller: SleepController) -> None:
        """wake() should raise SleepError when service is not in SLEEPING state."""
        with pytest.raises(SleepError):
            await controller.wake("svc-vllm")

    async def test_wake_raises_for_unsupported_driver(
        self, controller: SleepController, mock_registry: AsyncMock
    ) -> None:
        """wake() should raise SleepError when driver doesn't support sleep/wake."""
        # Make fastapi appear SLEEPING so we get past the state check
        fastapi_driver = mock_registry.get_driver(DriverType.FASTAPI)
        fastapi_driver.status.return_value = ServiceStatus(state=ServiceState.SLEEPING)

        with pytest.raises(SleepError):
            await controller.wake("svc-fastapi")


# ---------------------------------------------------------------------------
# get_sleepable_services() tests
# ---------------------------------------------------------------------------


class TestGetSleepableServices:
    """Tests for SleepController.get_sleepable_services()."""

    async def test_returns_only_running_with_supports_sleep(
        self, controller: SleepController
    ) -> None:
        """get_sleepable_services() returns only RUNNING services whose driver supports sleep."""
        result = await controller.get_sleepable_services()

        ids = [svc.id for svc in result]
        assert ids == ["svc-vllm"]

    async def test_excludes_already_sleeping_services(self, controller: SleepController) -> None:
        """get_sleepable_services() excludes services already in SLEEPING state."""
        result = await controller.get_sleepable_services()

        ids = [svc.id for svc in result]
        assert "svc-llamacpp" not in ids


# ---------------------------------------------------------------------------
# sleep_all() tests
# ---------------------------------------------------------------------------


class TestSleepAll:
    """Tests for SleepController.sleep_all()."""

    async def test_sleep_all_sleeps_running_sleepable_services(
        self, controller: SleepController, mock_registry: AsyncMock
    ) -> None:
        """sleep_all() sleeps all running sleepable services and returns their IDs."""
        result = await controller.sleep_all(level="l1")

        assert result == ["svc-vllm"]
        driver = mock_registry.get_driver(DriverType.VLLM)
        driver.sleep.assert_awaited_once_with(SVC_VLLM, "l1")

    async def test_sleep_all_skips_already_sleeping(
        self, controller: SleepController, mock_registry: AsyncMock
    ) -> None:
        """sleep_all() skips already-sleeping services."""
        await controller.sleep_all()

        llamacpp_driver = mock_registry.get_driver(DriverType.LLAMACPP)
        llamacpp_driver.sleep.assert_not_awaited()


# ---------------------------------------------------------------------------
# wake_all() tests
# ---------------------------------------------------------------------------


class TestWakeAll:
    """Tests for SleepController.wake_all()."""

    async def test_wake_all_wakes_all_sleeping_services(
        self, controller: SleepController, mock_registry: AsyncMock
    ) -> None:
        """wake_all() wakes all sleeping services and returns their IDs."""
        result = await controller.wake_all()

        assert result == ["svc-llamacpp"]
        llamacpp_driver = mock_registry.get_driver(DriverType.LLAMACPP)
        llamacpp_driver.wake.assert_awaited_once_with(SVC_LLAMACPP)

    async def test_wake_all_skips_non_sleeping_services(
        self, controller: SleepController, mock_registry: AsyncMock
    ) -> None:
        """wake_all() skips services that are not in SLEEPING state."""
        await controller.wake_all()

        vllm_driver = mock_registry.get_driver(DriverType.VLLM)
        vllm_driver.wake.assert_not_awaited()


# ---------------------------------------------------------------------------
# SleepError tests
# ---------------------------------------------------------------------------


class TestSleepError:
    """Tests for SleepError message formatting."""

    def test_error_message_includes_service_id_and_reason(self) -> None:
        """SleepError message should include both service_id and reason."""
        err = SleepError("svc-test", "not in RUNNING state")

        assert "svc-test" in str(err)
        assert "not in RUNNING state" in str(err)
