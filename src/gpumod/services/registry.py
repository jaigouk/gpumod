"""ServiceRegistry — discovers, tracks, and maps services to their drivers.

Bridges the Database layer and the driver instances, providing a unified
interface for querying service state and managing driver mappings.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from gpumod.models import DriverType, ServiceState
from gpumod.services.drivers.docker import DockerDriver
from gpumod.services.drivers.fastapi import FastAPIDriver
from gpumod.services.drivers.llamacpp import LlamaCppDriver
from gpumod.services.drivers.vllm import VLLMDriver

if TYPE_CHECKING:
    from gpumod.db import Database
    from gpumod.models import Service
    from gpumod.services.base import ServiceDriver


class ServiceRegistry:
    """Discovers, tracks, and maps services to their drivers.

    Parameters
    ----------
    db:
        The Database instance for service persistence.
    custom_drivers:
        Optional mapping of additional DriverType to ServiceDriver instances.
    """

    def __init__(
        self,
        db: Database,
        custom_drivers: dict[DriverType, ServiceDriver] | None = None,
    ) -> None:
        self._db = db
        self._drivers: dict[DriverType, ServiceDriver] = {
            DriverType.VLLM: VLLMDriver(),
            DriverType.LLAMACPP: LlamaCppDriver(),
            DriverType.FASTAPI: FastAPIDriver(),
            DriverType.DOCKER: DockerDriver(),
        }
        if custom_drivers:
            self._drivers.update(custom_drivers)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    async def list_all(self) -> list[Service]:
        """Return all services from the database."""
        return await self._db.list_services()

    async def list_running(self) -> list[Service]:
        """Return only services whose current state is RUNNING or SLEEPING.

        Checks status via the appropriate driver for each service concurrently
        using asyncio.gather.
        """
        services = await self._db.list_services()
        if not services:
            return []

        async def _check(svc: Service) -> tuple[Service, ServiceState]:
            driver = self._drivers.get(svc.driver)
            if driver is None:
                return svc, ServiceState.UNKNOWN
            status = await driver.status(svc)
            return svc, status.state

        results = await asyncio.gather(*[_check(svc) for svc in services])

        return [
            svc for svc, state in results if state in (ServiceState.RUNNING, ServiceState.SLEEPING)
        ]

    async def get(self, service_id: str) -> Service:
        """Return a service by ID.

        Raises
        ------
        KeyError
            If the service ID is not found in the database.
        """
        svc = await self._db.get_service(service_id)
        if svc is None:
            msg = f"Service not found: {service_id!r}"
            raise KeyError(msg)
        return svc

    def get_driver(self, driver_type: DriverType) -> ServiceDriver:
        """Return the driver instance for a given DriverType.

        Raises
        ------
        ValueError
            If no driver is registered for the given type.
        """
        driver = self._drivers.get(driver_type)
        if driver is None:
            msg = f"No driver registered for type: {driver_type!r}"
            raise ValueError(msg)
        return driver

    async def get_dependents(self, service_id: str) -> list[Service]:
        """Return services whose depends_on includes *service_id*."""
        all_services = await self._db.list_services()
        return [svc for svc in all_services if service_id in svc.depends_on]

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    async def register_service(self, service: Service) -> None:
        """Add a service to the database."""
        await self._db.insert_service(service)

    async def unregister_service(self, service_id: str) -> None:
        """Remove a service from the database."""
        await self._db.delete_service(service_id)

    def register_driver(self, driver_type: DriverType, driver: ServiceDriver) -> None:
        """Register a custom driver for the given type."""
        self._drivers[driver_type] = driver
