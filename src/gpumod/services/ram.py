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

# RAM safeguard tunables. A large-VRAM llama.cpp/vllm service typically pulls
# RAM proportional to its VRAM footprint (model weights staged + pinned
# transfer buffers + Python heap). The defaults reject the dangerous case
# where MemAvailable looks fine but pages are too fragmented for a
# contiguous CUDA pinned allocation.
#
# Source-of-truth lives here so both ServiceManager (Python API) and the
# `gpumod preflight ram` CLI subcommand share one implementation.
RAM_HEADROOM_RATIO = 0.3  # require avail >= sum(vram) * ratio + absolute
RAM_ABSOLUTE_HEADROOM_MB = 5_000
RAM_HARD_FLOOR_MB = 6_000  # never start a service if avail < this


def required_ram_mb(incoming_vram_mb: int) -> int:
    """Estimate RAM needed to safely start services with given VRAM."""
    return int(incoming_vram_mb * RAM_HEADROOM_RATIO) + RAM_ABSOLUTE_HEADROOM_MB


async def check_ram_safeguard(
    incoming_vram_mb: int,
    ram: RAMTracker,
    *,
    hard_floor_mb: int | None = None,
) -> str | None:
    """Return None if safe to start, else a single-line error message.

    Two-tier check:
      1. Hard floor — refuse if MemAvailable < ``hard_floor_mb``
         (defaults to ``RAM_HARD_FLOOR_MB``) regardless of service size.
      2. Headroom check — refuse if MemAvailable < required estimate
         for the incoming workload (``vram * RATIO + ABSOLUTE``).

    Parameters
    ----------
    incoming_vram_mb:
        Sum of declared ``vram_mb`` for the services about to start.
        Pass 0 to check only the hard floor (e.g. when starting nothing
        directly but still validating the kernel state).
    ram:
        :class:`RAMTracker` instance — typically reads /proc/meminfo.
        Injectable for testability.
    hard_floor_mb:
        Optional override of ``RAM_HARD_FLOOR_MB`` (e.g. ``gpumod
        preflight ram --hard-floor 8000``). When ``None``, the module
        default applies. The override IS the floor — a lower value
        does loosen the check.
    """
    floor = RAM_HARD_FLOOR_MB if hard_floor_mb is None else hard_floor_mb
    usage = await ram.get_usage()
    if usage.available_mb < floor:
        return (
            f"RAM safeguard: only {usage.available_mb} MB available, "
            f"hard floor is {floor} MB. Refusing to start "
            "to avoid CUDA pinned-memory freeze. Stop other workloads "
            "and retry."
        )

    if incoming_vram_mb > 0:
        required = required_ram_mb(incoming_vram_mb)
        if usage.available_mb < required:
            return (
                f"RAM safeguard: incoming services need ~{required} MB "
                f"available ({incoming_vram_mb} MB VRAM * "
                f"{RAM_HEADROOM_RATIO} + {RAM_ABSOLUTE_HEADROOM_MB} MB "
                f"headroom), but only {usage.available_mb} MB free. "
                "Refusing to start to avoid CUDA pinned-memory freeze."
            )
    return None


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

    def __init__(self, meminfo_path: Path | None = None) -> None:
        # Lazily resolve the module-level default so tests that monkey-patch
        # `_MEMINFO_PATH` at runtime take effect for new instances. Binding
        # the default at definition time would make patching silently no-op.
        self._path = meminfo_path if meminfo_path is not None else _MEMINFO_PATH

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
