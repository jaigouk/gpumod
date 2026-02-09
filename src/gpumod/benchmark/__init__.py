"""Benchmark utilities for gpumod."""

from gpumod.benchmark.consistency import (
    ConsistencyChecker,
    ConsistencyResult,
    compute_consistency,
    extract_facts,
)
from gpumod.benchmark.mode_switch import (
    ModeSwitchBenchmark,
    ModeSwitchResult,
    TransitionStats,
    generate_comparison_table,
)

__all__ = [
    # Consistency checking
    "ConsistencyChecker",
    "ConsistencyResult",
    "compute_consistency",
    "extract_facts",
    # Mode switching
    "ModeSwitchBenchmark",
    "ModeSwitchResult",
    "TransitionStats",
    "generate_comparison_table",
]
