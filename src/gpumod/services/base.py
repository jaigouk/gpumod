"""Abstract base class for service drivers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpumod.models import Service, ServiceStatus


class ServiceDriver(ABC):
    """Abstract base class that all service drivers must implement.

    Concrete drivers (VLLMDriver, LlamaCppDriver, etc.) must implement
    the four abstract methods: start, stop, status, and health_check.

    Sleep/wake support is optional; the default implementations raise
    NotImplementedError with a descriptive message.
    """

    @abstractmethod
    async def start(self, service: Service) -> None:
        """Start the service."""
        ...

    @abstractmethod
    async def stop(self, service: Service) -> None:
        """Stop the service."""
        ...

    @abstractmethod
    async def status(self, service: Service) -> ServiceStatus:
        """Get the current status of the service."""
        ...

    @abstractmethod
    async def health_check(self, service: Service) -> bool:
        """Check if the service is healthy."""
        ...

    async def sleep(self, service: Service, level: int = 1) -> None:
        """Put the service to sleep. Override in drivers that support sleep.

        Parameters
        ----------
        service:
            The service to put to sleep.
        level:
            Sleep level (1 or 2). Level semantics are driver-specific.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support sleep")

    async def wake(self, service: Service) -> None:
        """Wake the service from sleep. Override in drivers that support wake."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support wake")

    @property
    def supports_sleep(self) -> bool:
        """Whether this driver supports sleep/wake operations."""
        return False
