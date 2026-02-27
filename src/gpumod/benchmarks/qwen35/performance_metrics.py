"""Performance metrics for benchmark measurement."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Performance metrics for a benchmark run."""

    tps: float  # Tokens per second
    vram_mb: int | None  # VRAM usage in MB (None if unavailable)
    ttft_ms: float  # Time to first token in milliseconds
    total_ms: float  # Total generation time in milliseconds


def measure_tps(tokens: int, duration_seconds: float) -> float:
    """Calculate tokens per second.

    Args:
        tokens: Number of tokens generated
        duration_seconds: Generation duration in seconds

    Returns:
        Tokens per second

    Raises:
        ValueError: If duration is not positive
    """
    if duration_seconds <= 0:
        msg = "Duration must be positive"
        raise ValueError(msg)
    return tokens / duration_seconds


def measure_latency(
    start_time: float,
    first_token_time: float,
    end_time: float,
) -> dict[str, float]:
    """Calculate latency metrics.

    Args:
        start_time: Timestamp when generation started (seconds)
        first_token_time: Timestamp when first token arrived (seconds)
        end_time: Timestamp when generation completed (seconds)

    Returns:
        Dict with ttft_ms and total_ms
    """
    ttft_ms = (first_token_time - start_time) * 1000
    total_ms = (end_time - start_time) * 1000
    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
    }


def measure_vram(gpu_index: int = 0) -> int | None:
    """Measure current VRAM usage via nvidia-smi.

    Args:
        gpu_index: GPU index to measure (default 0)

    Returns:
        VRAM usage in MB, or None if nvidia-smi unavailable/error
    """
    try:
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return int(result.stdout.strip())
    except FileNotFoundError:
        return None
    except ValueError:
        return None
