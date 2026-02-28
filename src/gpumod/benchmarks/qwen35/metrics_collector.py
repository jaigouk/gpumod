"""Default MetricsCollector implementation for benchmark measurements."""

from __future__ import annotations

from typing import Any

from .performance_metrics import measure_tps, measure_vram


class DefaultMetricsCollector:
    """Default implementation of MetricsCollector protocol.

    Collects TPS, VRAM, and latency metrics during benchmark runs.
    """

    def __init__(self) -> None:
        self._generations: list[dict[str, float]] = []
        self._vram_idle: int | None = None
        self._vram_peak: int | None = None

    def measure_vram(self) -> int | None:
        """Measure current VRAM usage in MB.

        First call sets idle VRAM, subsequent calls update peak.
        """
        vram = measure_vram()
        if vram is None:
            return None

        if self._vram_idle is None:
            self._vram_idle = vram
        elif self._vram_peak is None or vram > self._vram_peak:
            self._vram_peak = vram

        return vram

    def record_generation(
        self,
        tokens: int,
        duration_seconds: float,
        ttft_seconds: float,
    ) -> None:
        """Record metrics for a single generation.

        Args:
            tokens: Number of tokens generated
            duration_seconds: Total generation time in seconds
            ttft_seconds: Time to first token in seconds
        """
        if duration_seconds <= 0:
            return

        tps = measure_tps(tokens, duration_seconds)

        self._generations.append({
            "tokens": tokens,
            "tps": tps,
            "ttft_ms": ttft_seconds * 1000,
            "total_ms": duration_seconds * 1000,
        })

    def get_iteration_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics for the iteration and reset.

        Returns:
            Dict with mean_tps, mean_ttft_ms, mean_total_ms, vram_idle_mb, vram_peak_mb
        """
        if not self._generations:
            result: dict[str, Any] = {}
            self._reset()
            return result

        # Calculate means
        tps_values = [g["tps"] for g in self._generations]
        ttft_values = [g["ttft_ms"] for g in self._generations]
        total_values = [g["total_ms"] for g in self._generations]

        result = {
            "mean_tps": sum(tps_values) / len(tps_values),
            "mean_ttft_ms": sum(ttft_values) / len(ttft_values),
            "mean_total_ms": sum(total_values) / len(total_values),
        }

        if self._vram_idle is not None:
            result["vram_idle_mb"] = self._vram_idle
        if self._vram_peak is not None:
            result["vram_peak_mb"] = self._vram_peak

        self._reset()
        return result

    def _reset(self) -> None:
        """Reset state for next iteration."""
        self._generations = []
        self._vram_idle = None
        self._vram_peak = None
