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
    "ConsistencyChecker",
    "ConsistencyResult",
    "ModeSwitchBenchmark",
    "ModeSwitchResult",
    "TransitionStats",
    "compute_consistency",
    "extract_facts",
    "generate_comparison_table",
]
