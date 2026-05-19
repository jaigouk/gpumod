"""System RAM tracking via /proc/meminfo.

Provides :class:`RAMTracker` for querying available system memory and
:class:`InsufficientRAMError` for safeguard rejections.

Why this exists
---------------

GPU services backed by llama.cpp, vLLM, or PyTorch use ``cudaHostAlloc``
(pinned, page-locked memory) for CPU↔GPU transfers. Pinned memory must be
physically contiguous and non-swappable, so it cannot be satisfied from
reclaimable buff/cache that ``MemAvailable`` includes.

When a CUDA pinned allocation is attempted shortly after a large workload
teardown, ``MemAvailable`` may report plenty of headroom but the kernel
cannot satisfy the contiguous allocation request and the NVIDIA driver
hangs. The result is a silent freeze — no OOM kill, no kernel panic.

This tracker is the data source for the pre-flight check in
:meth:`ServiceManager.switch_mode` and :meth:`ServiceManager.start_service`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from gpumod.models import RAMUsage

logger = logging.getLogger(__name__)

_MEMINFO_PATH = Path("/proc/meminfo")
_KB_PER_MB = 1024


class InsufficientRAMError(Exception):
    """Raised when starting a service would risk a memory-pressure freeze.

    Attributes
    ----------
    required_mb:
        Estimated RAM required for the incoming service(s) (MB).
    available_mb:
        Current MemAvailable from /proc/meminfo (MB).
    """

    def __init__(self, required_mb: int, available_mb: int, message: str | None = None) -> None:
        self.required_mb = required_mb
        self.available_mb = available_mb
        if message is None:
            message = (
                f"Insufficient RAM: need {required_mb}MB available "
                f"but only {available_mb}MB free. Refusing to start to "
                "avoid CUDA pinned-memory kernel freeze."
            )
        super().__init__(message)


class RAMTracker:
    """Query system RAM usage from ``/proc/meminfo``.

    The kernel exposes MemTotal/MemAvailable/MemFree in kilobytes. We
    convert to megabytes to match the rest of gpumod's units.
    """

    def __init__(self, meminfo_path: Path = _MEMINFO_PATH) -> None:
        self._path = meminfo_path

    async def get_usage(self) -> RAMUsage:
        """Read /proc/meminfo and return MemTotal/MemAvailable/MemFree in MB."""
        # /proc/meminfo is small (~1 KB) and sync read is faster than asyncio
        # subprocess overhead. Wrap in to_thread to stay non-blocking.
        text = await asyncio.to_thread(self._path.read_text)
        fields: dict[str, int] = {}
        for line in text.splitlines():
            key, _, rest = line.partition(":")
            parts = rest.strip().split()
            if not parts:
                continue
            try:
                fields[key] = int(parts[0])
            except ValueError:
                continue

        return RAMUsage(
            total_mb=fields.get("MemTotal", 0) // _KB_PER_MB,
            available_mb=fields.get("MemAvailable", 0) // _KB_PER_MB,
            free_mb=fields.get("MemFree", 0) // _KB_PER_MB,
        )

    async def wait_for_ram_release(
        self,
        required_mb: int,
        timeout_s: float = 60.0,
        poll_interval_s: float = 0.5,
        safety_margin_mb: int = 1024,
    ) -> bool:
        """Poll MemAvailable until it meets ``required_mb + safety_margin_mb``.

        Used between stopping outgoing services and starting incoming ones,
        to give the kernel time to consolidate freed pages.

        Returns
        -------
        bool
            True if RAM threshold was met within the timeout, False otherwise.
        """
        threshold_mb = required_mb + safety_margin_mb
        start = time.monotonic()

        while True:
            usage = await self.get_usage()
            if usage.available_mb >= threshold_mb:
                elapsed = time.monotonic() - start
                if elapsed > 0.1:
                    logger.info(
                        "RAM released: %d MB available >= %d MB required (waited %.1fs)",
                        usage.available_mb,
                        threshold_mb,
                        elapsed,
                    )
                return True

            elapsed = time.monotonic() - start
            if elapsed >= timeout_s:
                logger.warning(
                    "RAM wait timeout: %d MB available < %d MB required after %.1fs",
                    usage.available_mb,
                    threshold_mb,
                    elapsed,
                )
                return False

            logger.debug(
                "Waiting for RAM: %d MB available, need %d MB (%.1fs elapsed)",
                usage.available_mb,
                threshold_mb,
                elapsed,
            )
            await asyncio.sleep(poll_interval_s)
