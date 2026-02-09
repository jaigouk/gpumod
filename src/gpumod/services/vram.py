"""VRAM tracking via nvidia-smi and pynvml.

Provides :class:`VRAMTracker` for querying GPU info, VRAM usage,
and per-process memory consumption.  Uses pynvml when available for
faster polling, with nvidia-smi subprocess fallback.
"""

from __future__ import annotations

import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

from gpumod.models import GPUInfo, VRAMUsage

# Try to import pynvml for faster GPU memory queries
try:
    import pynvml  # type: ignore[import-untyped]

    _PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
    _PYNVML_AVAILABLE = False

if TYPE_CHECKING:
    from gpumod.models import Service

logger = logging.getLogger(__name__)


class NvidiaSmiError(Exception):
    """Raised when nvidia-smi is missing or returns a non-zero exit code."""


class VRAMTracker:
    """Query GPU information and VRAM usage via nvidia-smi or pynvml.

    Uses pynvml when available for faster GPU memory polling (no subprocess
    overhead). Falls back to nvidia-smi subprocess calls when pynvml is
    unavailable.
    """

    def __init__(self) -> None:
        self._pynvml_available = _PYNVML_AVAILABLE
        self._pynvml_initialized = False

    def _ensure_pynvml_init(self) -> None:
        """Initialize pynvml if available and not already initialized."""
        if self._pynvml_available and not self._pynvml_initialized:
            try:
                pynvml.nvmlInit()
                self._pynvml_initialized = True
            except Exception:
                logger.warning("pynvml init failed, falling back to nvidia-smi")
                self._pynvml_available = False

    async def _run_nvidia_smi(self, args: list[str]) -> str:
        """Execute nvidia-smi with *args* and return stdout.

        Raises
        ------
        NvidiaSmiError
            If nvidia-smi cannot be found or exits with a non-zero code.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise NvidiaSmiError(str(exc)) from exc

        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode() if stdout_bytes else ""
        stderr = stderr_bytes.decode() if stderr_bytes else ""

        if proc.returncode != 0:
            raise NvidiaSmiError(stderr)

        return stdout

    async def get_gpu_info(self) -> GPUInfo:
        """Query GPU name, total VRAM, and driver version via CSV output."""
        raw = await self._run_nvidia_smi(
            [
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        parts = [p.strip() for p in raw.strip().split(",")]
        return GPUInfo(
            name=parts[0],
            vram_total_mb=int(parts[1]),
            driver=parts[2],
        )

    async def get_usage(self) -> VRAMUsage:
        """Query current VRAM used/free via CSV output."""
        raw = await self._run_nvidia_smi(
            [
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ]
        )
        parts = [p.strip() for p in raw.strip().split(",")]
        used = int(parts[0])
        free = int(parts[1])
        return VRAMUsage(
            total_mb=used + free,
            used_mb=used,
            free_mb=free,
        )

    async def get_process_vram(self) -> dict[int, int]:
        """Return a mapping of PID to VRAM usage (MiB) from XML output."""
        raw = await self._run_nvidia_smi(["-q", "-x"])

        root = ET.fromstring(raw)  # noqa: S314 -- parsing trusted nvidia-smi output
        result: dict[int, int] = {}

        for proc_info in root.iter("process_info"):
            pid_el = proc_info.find("pid")
            mem_el = proc_info.find("used_memory")
            if pid_el is not None and mem_el is not None:
                pid_text = pid_el.text
                mem_text = mem_el.text
                if pid_text is None or mem_text is None:
                    continue
                pid = int(pid_text.strip())
                # Format: "2500 MiB"
                mem_str = mem_text.strip()
                mem_mb = int(mem_str.split()[0])
                result[pid] = mem_mb

        return result

    async def estimate_service_vram(self, service: Service) -> int:
        """Return the estimated VRAM for a service (from its config)."""
        return service.vram_mb

    # ------------------------------------------------------------------
    # VRAM release waiting (for mode switching)
    # ------------------------------------------------------------------

    async def _get_free_vram_mb(self, device_id: int = 0) -> int:
        """Get free VRAM in MB using pynvml or nvidia-smi fallback.

        Parameters
        ----------
        device_id:
            GPU device index (default 0 for single-GPU systems).

        Returns
        -------
        int
            Free VRAM in megabytes.
        """
        if self._pynvml_available:
            self._ensure_pynvml_init()
            if self._pynvml_initialized:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    # pynvml returns bytes, convert to MB
                    return int(mem_info.free / (1024 * 1024))
                except Exception:
                    logger.warning("pynvml query failed, falling back to nvidia-smi")
                    self._pynvml_available = False

        # Fallback to nvidia-smi
        usage = await self.get_usage()
        return usage.free_mb

    async def wait_for_vram_release(
        self,
        required_mb: int,
        timeout_s: float = 60.0,
        poll_interval_s: float = 0.5,
        safety_margin_mb: int = 512,
        device_id: int = 0,
    ) -> bool:
        """Wait for VRAM to be released before starting a new GPU service.

        Polls GPU memory until free VRAM exceeds the required amount plus
        a safety margin, or until timeout expires.

        Parameters
        ----------
        required_mb:
            Minimum free VRAM required in megabytes.
        timeout_s:
            Maximum time to wait in seconds (default 60s).
        poll_interval_s:
            Time between polls in seconds (default 0.5s).
        safety_margin_mb:
            Extra margin above required_mb (default 512MB).
        device_id:
            GPU device index (default 0).

        Returns
        -------
        bool
            True if VRAM was released within timeout, False otherwise.
        """
        threshold_mb = required_mb + safety_margin_mb
        start_time = time.monotonic()

        while True:
            free_mb = await self._get_free_vram_mb(device_id)

            if free_mb >= threshold_mb:
                elapsed = time.monotonic() - start_time
                if elapsed > 0.1:  # Only log if we actually waited
                    logger.info(
                        "VRAM released: %d MB free >= %d MB required (waited %.1fs)",
                        free_mb,
                        threshold_mb,
                        elapsed,
                    )
                return True

            elapsed = time.monotonic() - start_time
            if elapsed >= timeout_s:
                logger.warning(
                    "VRAM wait timeout: %d MB free < %d MB required after %.1fs",
                    free_mb,
                    threshold_mb,
                    elapsed,
                )
                return False

            logger.debug(
                "Waiting for VRAM: %d MB free, need %d MB (%.1fs elapsed)",
                free_mb,
                threshold_mb,
                elapsed,
            )
            await asyncio.sleep(poll_interval_s)
