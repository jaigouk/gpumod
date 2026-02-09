"""SleepController — manages sleep/wake operations across all services."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpumod.models import ServiceState

if TYPE_CHECKING:
    from gpumod.models import Service
    from gpumod.services.registry import ServiceRegistry


class SleepError(Exception):
    """Raised when a sleep/wake operation fails.

    Attributes
    ----------
    service_id:
        The ID of the service that caused the error.
    reason:
        A human-readable explanation of why the operation failed.
    """

    def __init__(self, service_id: str, reason: str) -> None:
        self.service_id = service_id
        self.reason = reason
        super().__init__(f"[{service_id}] {reason}")


class SleepController:
    """Manages sleep/wake operations across all services.

    Parameters
    ----------
    registry:
        The ServiceRegistry used to look up services and their drivers.
    """

    def __init__(self, registry: ServiceRegistry) -> None:
        self._registry = registry

    async def sleep(self, service_id: str, level: int = 1) -> None:
        """Put a service to sleep.

        The operation is **idempotent**: calling sleep on an already-sleeping
        service is a silent no-op.

        Parameters
        ----------
        service_id:
            The ID of the service to sleep.
        level:
            The sleep level (1 or 2). See driver docs for level semantics.

        Raises
        ------
        KeyError
            If *service_id* does not exist in the registry.
        SleepError
            If the service is not in RUNNING state (and not already SLEEPING),
            or if its driver does not support sleep.
        """
        service = await self._registry.get(service_id)
        driver = self._registry.get_driver(service.driver)
        status = await driver.status(service)

        # Idempotent: already sleeping is a no-op
        if status.state == ServiceState.SLEEPING:
            return

        if not driver.supports_sleep:
            raise SleepError(service_id, "driver does not support sleep")

        if status.state != ServiceState.RUNNING:
            raise SleepError(service_id, f"service is {status.state}, must be RUNNING")

        await driver.sleep(service, level)

    async def wake(self, service_id: str) -> None:
        """Wake a sleeping service.

        Parameters
        ----------
        service_id:
            The ID of the service to wake.

        Raises
        ------
        KeyError
            If *service_id* does not exist in the registry.
        SleepError
            If the service is not in SLEEPING state, or if the driver does
            not support sleep/wake.
        """
        service = await self._registry.get(service_id)
        driver = self._registry.get_driver(service.driver)
        status = await driver.status(service)

        if not driver.supports_sleep:
            raise SleepError(service_id, "driver does not support sleep")

        if status.state != ServiceState.SLEEPING:
            raise SleepError(service_id, f"service is {status.state}, must be SLEEPING")

        await driver.wake(service)

    async def get_sleepable_services(self) -> list[Service]:
        """Return RUNNING services whose driver supports sleep.

        Already-sleeping services are excluded.
        """
        all_services = await self._registry.list_all()
        sleepable: list[Service] = []

        for service in all_services:
            driver = self._registry.get_driver(service.driver)
            if not driver.supports_sleep:
                continue
            status = await driver.status(service)
            if status.state == ServiceState.RUNNING:
                sleepable.append(service)

        return sleepable

    async def sleep_all(self, level: int = 1) -> list[str]:
        """Sleep all running sleepable services.

        Returns the list of service IDs that were put to sleep.
        """
        sleepable = await self.get_sleepable_services()
        slept_ids: list[str] = []

        for service in sleepable:
            driver = self._registry.get_driver(service.driver)
            await driver.sleep(service, level)
            slept_ids.append(service.id)

        return slept_ids

    async def wake_all(self) -> list[str]:
        """Wake all sleeping services whose driver supports sleep.

        Returns the list of service IDs that were woken up.
        """
        all_services = await self._registry.list_all()
        woken_ids: list[str] = []

        for service in all_services:
            driver = self._registry.get_driver(service.driver)
            if not driver.supports_sleep:
                continue
            status = await driver.status(service)
            if status.state != ServiceState.SLEEPING:
                continue
            await driver.wake(service)
            woken_ids.append(service.id)

        return woken_ids
