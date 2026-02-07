"""VRAM tracking via nvidia-smi.

Provides :class:`VRAMTracker` for querying GPU info, VRAM usage,
and per-process memory consumption.  All subprocess calls use
``asyncio.create_subprocess_exec`` (never ``shell=True``).
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

from gpumod.models import GPUInfo, VRAMUsage

if TYPE_CHECKING:
    from gpumod.models import Service


class NvidiaSmiError(Exception):
    """Raised when nvidia-smi is missing or returns a non-zero exit code."""


class VRAMTracker:
    """Query GPU information and VRAM usage via nvidia-smi."""

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
