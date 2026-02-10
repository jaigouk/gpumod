"""System information collector for model discovery.

Collects GPU, RAM, and swap information dynamically at runtime.
Reuses existing VRAMTracker infrastructure for NVIDIA GPU queries.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from gpumod.services.vram import NvidiaSmiError, VRAMTracker

logger = logging.getLogger(__name__)


class NvidiaSmiUnavailableError(Exception):
    """Raised when nvidia-smi is unavailable or times out."""


@dataclass(frozen=True)
class SystemInfo:
    """Immutable snapshot of current system state.

    All values are detected at runtime, never hardcoded.

    Attributes:
        gpu_total_mb: Total GPU VRAM in megabytes.
        gpu_used_mb: Currently used GPU VRAM in megabytes.
        gpu_available_mb: Available GPU VRAM (total - used).
        gpu_name: GPU model name (e.g., "NVIDIA RTX 4090").
        ram_total_mb: Total system RAM in megabytes.
        ram_available_mb: Available system RAM in megabytes.
        swap_available_mb: Available swap space in megabytes.
        current_mode: Currently active gpumod mode, or None.
        running_services: Tuple of running gpumod service IDs.
    """

    gpu_total_mb: int
    gpu_used_mb: int
    gpu_available_mb: int
    gpu_name: str
    ram_total_mb: int
    ram_available_mb: int
    swap_available_mb: int
    current_mode: str | None
    running_services: tuple[str, ...]


class SystemInfoCollector:
    """Collects system information for model discovery.

    Uses VRAMTracker for GPU info (reusing existing infrastructure)
    and reads /proc/meminfo for RAM/swap on Linux.

    Example:
        >>> collector = SystemInfoCollector()
        >>> info = await collector.get_system_info()
        >>> print(f"GPU: {info.gpu_name}, Available: {info.gpu_available_mb} MB")
    """

    def __init__(
        self,
        *,
        nvidia_smi_timeout: float = 5.0,
        vram_tracker: VRAMTracker | None = None,
    ) -> None:
        """Initialize the collector.

        Args:
            nvidia_smi_timeout: Timeout for nvidia-smi calls in seconds.
            vram_tracker: Optional VRAMTracker instance (for testing).
        """
        self._nvidia_smi_timeout = nvidia_smi_timeout
        self._vram_tracker = vram_tracker

    async def get_system_info(self) -> SystemInfo:
        """Collect current system information.

        Returns:
            SystemInfo with current GPU, RAM, and service state.

        Raises:
            NvidiaSmiUnavailableError: If nvidia-smi is missing or times out.
        """
        # Get or create VRAMTracker
        vram_tracker = self._vram_tracker or VRAMTracker()

        # Query GPU info with timeout
        try:
            gpu_info, vram_usage = await asyncio.wait_for(
                asyncio.gather(
                    vram_tracker.get_gpu_info(),
                    vram_tracker.get_usage(),
                ),
                timeout=self._nvidia_smi_timeout,
            )
        except TimeoutError as exc:
            msg = f"nvidia-smi timeout after {self._nvidia_smi_timeout}s"
            raise NvidiaSmiUnavailableError(msg) from exc
        except NvidiaSmiError as exc:
            raise NvidiaSmiUnavailableError(str(exc)) from exc

        # Parse /proc/meminfo for RAM and swap
        ram_total_mb, ram_available_mb, swap_available_mb = self._parse_meminfo()

        # Get current mode and running services
        current_mode = await self._get_current_mode()
        running_services = await self._get_running_services()

        return SystemInfo(
            gpu_total_mb=gpu_info.vram_total_mb,
            gpu_used_mb=vram_usage.used_mb,
            gpu_available_mb=vram_usage.free_mb,
            gpu_name=gpu_info.name,
            ram_total_mb=ram_total_mb,
            ram_available_mb=ram_available_mb,
            swap_available_mb=swap_available_mb,
            current_mode=current_mode,
            running_services=tuple(running_services),
        )

    def _parse_meminfo(self) -> tuple[int, int, int]:
        """Parse /proc/meminfo for RAM and swap values.

        Returns:
            Tuple of (ram_total_mb, ram_available_mb, swap_available_mb).
        """
        meminfo_path = Path("/proc/meminfo")
        if not meminfo_path.exists():
            logger.warning("/proc/meminfo not found, returning zeros for RAM/swap")
            return (0, 0, 0)

        values: dict[str, int] = {}
        try:
            with meminfo_path.open() as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        # Values in /proc/meminfo are in kB
                        try:
                            values[key] = int(parts[1])
                        except ValueError:
                            continue
        except OSError as exc:
            logger.warning("Failed to read /proc/meminfo: %s", exc)
            return (0, 0, 0)

        # Convert from kB to MB
        ram_total_mb = values.get("MemTotal", 0) // 1024
        ram_available_mb = values.get("MemAvailable", 0) // 1024
        swap_free_mb = values.get("SwapFree", 0) // 1024

        return (ram_total_mb, ram_available_mb, swap_free_mb)

    async def _get_current_mode(self) -> str | None:
        """Get the currently active gpumod mode.

        Returns:
            Mode name or None if no mode is active.
        """
        # Import here to avoid circular imports
        try:
            from gpumod.config import get_settings
            from gpumod.db import Database

            settings = get_settings()
            if not settings.db_path.exists():
                return None

            db = Database(settings.db_path)
            await db.connect()
            try:
                return await db.get_setting("current_mode")
            finally:
                await db.close()
        except Exception as exc:
            logger.debug("Failed to get current mode: %s", exc)
            return None

    async def _get_running_services(self) -> list[str]:
        """Get list of running gpumod service IDs.

        Returns:
            List of service IDs that are currently running.
        """
        try:
            from gpumod.config import get_settings
            from gpumod.db import Database
            from gpumod.models import ServiceState
            from gpumod.services.registry import ServiceRegistry

            settings = get_settings()
            if not settings.db_path.exists():
                return []

            db = Database(settings.db_path)
            await db.connect()
            try:
                registry = ServiceRegistry(db)
                services = await registry.list_services()
                running = []
                for svc in services:
                    state = await registry.get_service_state(svc.id)
                    if state in (ServiceState.RUNNING, ServiceState.STARTING):
                        running.append(svc.id)
                return running
            finally:
                await db.close()
        except Exception as exc:
            logger.debug("Failed to get running services: %s", exc)
            return []


# Helper functions for external use (can be mocked in tests)
async def get_current_mode() -> str | None:
    """Get current gpumod mode (convenience function)."""
    collector = SystemInfoCollector()
    return await collector._get_current_mode()


async def get_running_services() -> list[str]:
    """Get running gpumod services (convenience function)."""
    collector = SystemInfoCollector()
    return await collector._get_running_services()
