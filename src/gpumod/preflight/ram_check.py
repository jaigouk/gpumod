"""RAM preflight check for gpumod services (gpumod-bfx).

Validates that sufficient system RAM is available before starting a service.
Prevents OOM freezes from mmap, CPU offloading, or KV cache allocation.

Reads MemAvailable from /proc/meminfo directly (no dependency on
SystemInfoCollector) with injectable path for testability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from gpumod.preflight.base import CheckResult

if TYPE_CHECKING:
    from gpumod.models import Service

logger = logging.getLogger(__name__)

DEFAULT_MIN_FREE_MB = 1024
DEFAULT_WARN_FREE_MB = 4096


class RAMCheck:
    """Preflight check that validates system RAM availability.

    Prevents service start when system RAM is critically low,
    avoiding OOM freezes from mmap, CPU offloading, or KV cache.

    Usage:
        check = RAMCheck()
        result = await check.check(service)
    """

    def __init__(
        self,
        min_free_mb: int = DEFAULT_MIN_FREE_MB,
        warn_free_mb: int = DEFAULT_WARN_FREE_MB,
        meminfo_path: Path = Path("/proc/meminfo"),
    ) -> None:
        """Initialize RAMCheck.

        Parameters
        ----------
        min_free_mb:
            Error threshold in MB (default 2048).
        warn_free_mb:
            Warning threshold in MB (default 4096).
        meminfo_path:
            Path to meminfo file (injectable for testing).
        """
        self._min_free_mb = min_free_mb
        self._warn_free_mb = warn_free_mb
        self._meminfo_path = meminfo_path

    @property
    def name(self) -> str:
        """Return check name."""
        return "ram"

    async def check(self, service: Service) -> CheckResult:
        """Check if system RAM is sufficient for service startup.

        Parameters
        ----------
        service:
            The service to validate.

        Returns
        -------
        CheckResult:
            Pass/warn/error based on available RAM.
        """
        mem_available_mb, mem_total_mb = self._read_meminfo()

        # Could not read meminfo — warn but don't block
        if mem_available_mb is None:
            return CheckResult(
                passed=True,
                severity="warning",
                message="RAM check skipped: unable to read /proc/meminfo",
            )

        # Critical: below minimum threshold
        if mem_available_mb < self._min_free_mb:
            return CheckResult(
                passed=False,
                severity="error",
                message=(
                    f"System RAM critically low: {mem_available_mb} MB available "
                    f"(minimum: {self._min_free_mb} MB)"
                ),
                remediation=(
                    f"MemAvailable: {mem_available_mb} MB "
                    f"(of {mem_total_mb} MB total). "
                    f"Close other applications, reduce model size, "
                    f"or add swap space."
                ),
            )

        # Warning: below warn threshold but above minimum
        if mem_available_mb < self._warn_free_mb:
            return CheckResult(
                passed=True,
                severity="warning",
                message=(
                    f"System RAM low: {mem_available_mb} MB available "
                    f"(warning threshold: {self._warn_free_mb} MB)"
                ),
            )

        # OK
        return CheckResult(
            passed=True,
            severity="info",
            message=f"RAM OK: {mem_available_mb} MB available",
        )

    def _read_meminfo(self) -> tuple[int | None, int | None]:
        """Read MemAvailable and MemTotal from /proc/meminfo.

        Returns
        -------
        tuple[int | None, int | None]:
            (mem_available_mb, mem_total_mb) or (None, None) on failure.
        """
        if not self._meminfo_path.exists():
            logger.warning("meminfo not found at %s", self._meminfo_path)
            return None, None

        values: dict[str, int] = {}
        try:
            with self._meminfo_path.open() as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        try:
                            values[key] = int(parts[1])
                        except ValueError:
                            continue
        except OSError as exc:
            logger.warning("Failed to read meminfo: %s", exc)
            return None, None

        mem_available_kb = values.get("MemAvailable")
        if mem_available_kb is None:
            logger.warning("MemAvailable not found in %s", self._meminfo_path)
            return None, None

        mem_total_kb = values.get("MemTotal", 0)
        return mem_available_kb // 1024, mem_total_kb // 1024
