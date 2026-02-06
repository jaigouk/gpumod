"""Comprehensive tests for gpumod.services.base ServiceDriver ABC."""

from __future__ import annotations

import pytest

from gpumod.models import DriverType, Service, ServiceState, ServiceStatus
from gpumod.services.base import ServiceDriver


def _make_service(service_id: str = "test-svc") -> Service:
    """Create a minimal Service for testing."""
    return Service(
        id=service_id,
        name="Test Service",
        driver=DriverType.VLLM,
        vram_mb=4096,
    )


class TestServiceDriverABC:
    """ServiceDriver cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            ServiceDriver()  # type: ignore[abstract]

    def test_subclass_missing_start_raises(self) -> None:
        class Incomplete(ServiceDriver):
            async def stop(self, service: Service) -> None: ...
            async def status(self, service: Service) -> ServiceStatus: ...
            async def health_check(self, service: Service) -> bool: ...

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_missing_stop_raises(self) -> None:
        class Incomplete(ServiceDriver):
            async def start(self, service: Service) -> None: ...
            async def status(self, service: Service) -> ServiceStatus: ...
            async def health_check(self, service: Service) -> bool: ...

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_missing_status_raises(self) -> None:
        class Incomplete(ServiceDriver):
            async def start(self, service: Service) -> None: ...
            async def stop(self, service: Service) -> None: ...
            async def health_check(self, service: Service) -> bool: ...

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_missing_health_check_raises(self) -> None:
        class Incomplete(ServiceDriver):
            async def start(self, service: Service) -> None: ...
            async def stop(self, service: Service) -> None: ...
            async def status(self, service: Service) -> ServiceStatus: ...

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


class TestMinimalConcreteSubclass:
    """A concrete subclass implementing all 4 abstract methods can be instantiated."""

    def _make_driver(self) -> ServiceDriver:
        class MinimalDriver(ServiceDriver):
            async def start(self, service: Service) -> None: ...
            async def stop(self, service: Service) -> None: ...
            async def status(self, service: Service) -> ServiceStatus:
                return ServiceStatus(state=ServiceState.STOPPED)

            async def health_check(self, service: Service) -> bool:
                return True

        return MinimalDriver()

    def test_can_be_instantiated(self) -> None:
        driver = self._make_driver()
        assert isinstance(driver, ServiceDriver)


class TestDefaultSleep:
    """Default sleep() raises NotImplementedError with driver class name."""

    def _make_driver(self) -> ServiceDriver:
        class NoSleepDriver(ServiceDriver):
            async def start(self, service: Service) -> None: ...
            async def stop(self, service: Service) -> None: ...
            async def status(self, service: Service) -> ServiceStatus:
                return ServiceStatus(state=ServiceState.STOPPED)

            async def health_check(self, service: Service) -> bool:
                return True

        return NoSleepDriver()

    async def test_sleep_raises_not_implemented(self) -> None:
        driver = self._make_driver()
        svc = _make_service()
        with pytest.raises(NotImplementedError, match="NoSleepDriver"):
            await driver.sleep(svc)

    async def test_sleep_with_level_arg(self) -> None:
        driver = self._make_driver()
        svc = _make_service()
        with pytest.raises(NotImplementedError, match="does not support sleep"):
            await driver.sleep(svc, level="l2")


class TestDefaultWake:
    """Default wake() raises NotImplementedError with driver class name."""

    def _make_driver(self) -> ServiceDriver:
        class NoWakeDriver(ServiceDriver):
            async def start(self, service: Service) -> None: ...
            async def stop(self, service: Service) -> None: ...
            async def status(self, service: Service) -> ServiceStatus:
                return ServiceStatus(state=ServiceState.STOPPED)

            async def health_check(self, service: Service) -> bool:
                return True

        return NoWakeDriver()

    async def test_wake_raises_not_implemented(self) -> None:
        driver = self._make_driver()
        svc = _make_service()
        with pytest.raises(NotImplementedError, match="NoWakeDriver"):
            await driver.wake(svc)

    async def test_wake_error_message_contains_does_not_support(self) -> None:
        driver = self._make_driver()
        svc = _make_service()
        with pytest.raises(NotImplementedError, match="does not support wake"):
            await driver.wake(svc)


class TestSupportsSleepProperty:
    """Default supports_sleep returns False; subclass can override to True."""

    def test_default_supports_sleep_is_false(self) -> None:
        class BasicDriver(ServiceDriver):
            async def start(self, service: Service) -> None: ...
            async def stop(self, service: Service) -> None: ...
            async def status(self, service: Service) -> ServiceStatus:
                return ServiceStatus(state=ServiceState.STOPPED)

            async def health_check(self, service: Service) -> bool:
                return True

        driver = BasicDriver()
        assert driver.supports_sleep is False

    def test_subclass_can_override_to_true(self) -> None:
        class SleepyDriver(ServiceDriver):
            async def start(self, service: Service) -> None: ...
            async def stop(self, service: Service) -> None: ...
            async def status(self, service: Service) -> ServiceStatus:
                return ServiceStatus(state=ServiceState.STOPPED)

            async def health_check(self, service: Service) -> bool:
                return True

            @property
            def supports_sleep(self) -> bool:
                return True

        driver = SleepyDriver()
        assert driver.supports_sleep is True
