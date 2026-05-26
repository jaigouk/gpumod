"""RAM preflight check for gpumod services (gpumod-bfx, extended gpumod-lgt).

Validates that sufficient system RAM is available before starting a service.
Prevents OOM from mmap page-cache pressure during GGUF model loading.

gpumod-lgt: extended to model-file-size aware checking. Reading an 18 GB
GGUF via llama-server mmap loads its pages into the host page cache; if
MemAvailable cannot accommodate (file_size * overhead_factor + min_free),
starting the service risks OOM.

gpumod-ki89: overhead factor relaxed from 1.1 to 0.9. The 1.1x headroom
guarded against cudaHostAlloc contiguous-page hangs; gpumod-56md made
GGML_CUDA_NO_PINNED=1 the default for all llamacpp services, eliminating
that freeze class entirely. Factor < 1.0 is valid because mmap'd pages
are demand-paged and reclaimable during GPU transfer — the full model
file does not need to reside in RAM simultaneously.
See llamacpp.service.j2 line 13.

Reads MemAvailable from /proc/meminfo directly (no dependency on
SystemInfoCollector) with injectable path for testability.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpumod.preflight.base import CheckResult

if TYPE_CHECKING:
    from gpumod.models import Service

logger = logging.getLogger(__name__)

DEFAULT_MIN_FREE_MB = 1024
DEFAULT_WARN_FREE_MB = 4096
# SAFETY: factor < 1.1 is only safe because llamacpp.service.j2 sets
# GGML_CUDA_NO_PINNED=1, bypassing cudaHostAlloc (gpumod-56md).
# If that env var is removed, raise this back to >= 1.1.
# See gpumod-56md, gpumod-ki89, gpumod-x7rv.
DEFAULT_MMAP_OVERHEAD_FACTOR = 0.9


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
        mmap_overhead_factor: float = DEFAULT_MMAP_OVERHEAD_FACTOR,
    ) -> None:
        """Initialize RAMCheck.

        Parameters
        ----------
        min_free_mb:
            Error threshold in MB (default 1024).
        warn_free_mb:
            Warning threshold in MB (default 4096).
        meminfo_path:
            Path to meminfo file (injectable for testing).
        mmap_overhead_factor:
            Multiplier applied to the service's model file size when the
            model_path is known. Required RAM = file_size * factor + min_free.
            Default 0.9 (gpumod-ki89: relaxed from 1.1 because
            GGML_CUDA_NO_PINNED=1 is the default in llamacpp.service.j2 —
            see module docstring).
        """
        self._min_free_mb = min_free_mb
        self._warn_free_mb = warn_free_mb
        self._meminfo_path = meminfo_path
        self._mmap_overhead_factor = mmap_overhead_factor

    @property
    def name(self) -> str:
        """Return check name."""
        return "ram"

    async def check(self, service: Service) -> CheckResult:
        """Check if system RAM is sufficient for service startup.

        gpumod-lgt: when the service has a resolvable ``model_path``, this
        check requires MemAvailable to cover ``file_size * factor + min_free``
        — a much stricter bound than threshold-only. Services without a
        known model file fall back to threshold semantics.

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

        # gpumod-lgt: prefer model-file-size aware check when the file
        # exists. Falls through to threshold check on missing path / file.
        model_path, model_size_mb = self._get_model_file_size_mb(service)
        if model_size_mb is not None:
            required_mb = int(model_size_mb * self._mmap_overhead_factor)
            required_with_floor = required_mb + self._min_free_mb
            if mem_available_mb < required_with_floor:
                deficit = required_with_floor - mem_available_mb
                model_name = Path(model_path).name if model_path else "model file"
                return CheckResult(
                    passed=False,
                    severity="error",
                    message=(
                        f"System RAM insufficient to mmap {model_size_mb} MB "
                        f"model: {mem_available_mb} MB available, "
                        f"{required_with_floor} MB required "
                        f"({model_size_mb} * {self._mmap_overhead_factor} "
                        f"overhead + {self._min_free_mb} MB floor) "
                        f"— short by {deficit} MB"
                    ),
                    remediation=(
                        f"Model file: {model_name} ({model_size_mb} MB)\n"
                        f"  MemAvailable: {mem_available_mb} MB "
                        f"(of {mem_total_mb} MB total)\n"
                        f"  Required: {required_with_floor} MB "
                        f"(mmap overhead factor {self._mmap_overhead_factor})\n"
                        f"  Deficit: {deficit} MB\n"
                        f"\n"
                        f"Suggestions:\n"
                        f"  1. Stop other gpumod services to free RAM "
                        f"(see `gpumod service list`)\n"
                        f"  2. Close memory-hungry desktop apps "
                        f"(browser, IDE, code-server)\n"
                        f"  3. Use a smaller model quantization\n"
                        f"  4. See "
                        f"docs/research/20260525_oom_protection_findings/FINDINGS.md "
                        f"for the broader pinned-memory freeze class"
                    ),
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

    def _get_model_file_size_mb(self, service: Service) -> tuple[str | None, int | None]:
        """Resolve ``service.extra_config['unit_vars']['model_path']`` and
        return (resolved_path, file_size_mb).

        Returns ``(None, None)`` when the model_path is unset, unresolvable,
        or the file does not yet exist (e.g. before download).
        """
        try:
            unit_vars: Any = service.extra_config.get("unit_vars", {})
        except AttributeError:
            return None, None
        if not isinstance(unit_vars, dict):
            return None, None
        raw_path = unit_vars.get("model_path")
        if not raw_path:
            return None, None
        try:
            resolved = os.path.expandvars(os.path.expanduser(str(raw_path)))
            stat_result = Path(resolved).stat()
        except (OSError, ValueError):
            return None, None
        size_mb = stat_result.st_size // (1024 * 1024)
        return resolved, size_mb

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
