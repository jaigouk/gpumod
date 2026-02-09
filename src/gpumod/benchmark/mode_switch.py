"""Mode switch benchmarking utilities.

Provides tools to measure and compare mode switch latency,
particularly for comparing sleep-aware vs restart-based transitions.

Usage:
    from gpumod.benchmark.mode_switch import ModeSwitchBenchmark

    benchmark = ModeSwitchBenchmark(manager=service_manager)
    results = await benchmark.run_benchmark(
        transitions=[("code", "rag"), ("rag", "code")],
        warmup_runs=2,
        measured_runs=5,
    )

    stats = [TransitionStats.from_results(group) for group in grouped_results]
    table = generate_comparison_table(baseline_stats, stats)
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpumod.services.manager import ServiceManager


@dataclass
class ModeSwitchResult:
    """Result of a single mode switch operation.

    Attributes
    ----------
    from_mode:
        The mode we switched from.
    to_mode:
        The mode we switched to.
    total_ms:
        Total time for the switch operation in milliseconds.
    services_slept:
        IDs of services that were put to sleep.
    services_stopped:
        IDs of services that were stopped (not slept).
    services_woken:
        IDs of services that were woken from sleep.
    services_started:
        IDs of services that were started (not woken).
    error:
        Error message if the switch failed.
    """

    from_mode: str
    to_mode: str
    total_ms: float
    services_slept: list[str] = field(default_factory=list)
    services_stopped: list[str] = field(default_factory=list)
    services_woken: list[str] = field(default_factory=list)
    services_started: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def transition(self) -> str:
        """Return formatted transition string."""
        return f"{self.from_mode} → {self.to_mode}"

    @property
    def is_sleep_capable(self) -> bool:
        """Return True if this transition used sleep/wake operations."""
        return bool(self.services_slept or self.services_woken)


@dataclass
class TransitionStats:
    """Aggregated statistics for a transition type.

    Attributes
    ----------
    transition:
        The transition string (e.g., "code → rag").
    runs:
        Number of measured runs.
    min_ms:
        Minimum time in milliseconds.
    max_ms:
        Maximum time in milliseconds.
    mean_ms:
        Mean time in milliseconds.
    median_ms:
        Median time in milliseconds.
    is_sleep_capable:
        Whether this transition used sleep/wake operations.
    """

    transition: str
    runs: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    is_sleep_capable: bool

    @classmethod
    def from_results(cls, results: list[ModeSwitchResult]) -> TransitionStats:
        """Create stats from a list of results for the same transition.

        Parameters
        ----------
        results:
            List of ModeSwitchResult objects for the same transition.

        Returns
        -------
        TransitionStats:
            Aggregated statistics.
        """
        if not results:
            msg = "Cannot compute stats from empty results"
            raise ValueError(msg)

        times = [r.total_ms for r in results]
        first = results[0]

        return cls(
            transition=first.transition,
            runs=len(results),
            min_ms=min(times),
            max_ms=max(times),
            mean_ms=statistics.mean(times),
            median_ms=statistics.median(times),
            is_sleep_capable=any(r.is_sleep_capable for r in results),
        )


class ModeSwitchBenchmark:
    """Benchmark runner for mode switch operations.

    Parameters
    ----------
    manager:
        The ServiceManager to use for mode switching.
    """

    def __init__(self, manager: ServiceManager) -> None:
        self._manager = manager

    async def run_transition(self, from_mode: str, to_mode: str) -> ModeSwitchResult:
        """Run a single mode transition and measure timing.

        Parameters
        ----------
        from_mode:
            The current mode (for labeling; actual current mode from DB).
        to_mode:
            The target mode to switch to.

        Returns
        -------
        ModeSwitchResult:
            The result of the transition including timing.
        """
        start = time.perf_counter()

        mode_result = await self._manager.switch_mode(to_mode)

        elapsed_ms = (time.perf_counter() - start) * 1000

        if not mode_result.success:
            return ModeSwitchResult(
                from_mode=from_mode,
                to_mode=to_mode,
                total_ms=elapsed_ms,
                error=mode_result.errors[0] if mode_result.errors else "Unknown error",
            )

        # ModeResult combines slept+stopped into 'stopped' and woken+started into 'started'
        # We pass through what the manager reports
        return ModeSwitchResult(
            from_mode=from_mode,
            to_mode=to_mode,
            total_ms=elapsed_ms,
            services_slept=[],  # ModeResult doesn't separate these yet
            services_stopped=mode_result.stopped or [],
            services_woken=[],  # ModeResult doesn't separate these yet
            services_started=mode_result.started or [],
        )

    async def run_benchmark(
        self,
        transitions: list[tuple[str, str]],
        warmup_runs: int = 1,
        measured_runs: int = 3,
    ) -> list[ModeSwitchResult]:
        """Run a benchmark with warmup and measured iterations.

        Parameters
        ----------
        transitions:
            List of (from_mode, to_mode) tuples to benchmark.
        warmup_runs:
            Number of warmup iterations to run (results discarded).
        measured_runs:
            Number of measured iterations to run.

        Returns
        -------
        list[ModeSwitchResult]:
            All measured results (warmup discarded).
        """
        results: list[ModeSwitchResult] = []

        for from_mode, to_mode in transitions:
            # Warmup runs (discarded)
            for _ in range(warmup_runs):
                await self.run_transition(from_mode, to_mode)

            # Measured runs
            for _ in range(measured_runs):
                result = await self.run_transition(from_mode, to_mode)
                results.append(result)

        return results


def generate_comparison_table(
    baseline: list[TransitionStats],
    current: list[TransitionStats],
) -> str:
    """Generate a markdown comparison table.

    Parameters
    ----------
    baseline:
        Baseline statistics (e.g., without sleep-aware switching).
    current:
        Current statistics (e.g., with sleep-aware switching).

    Returns
    -------
    str:
        Markdown table comparing baseline and current stats.
    """
    # Build lookup for baseline stats
    baseline_by_transition = {s.transition: s for s in baseline}

    lines = [
        "| Transition | Baseline (ms) | After (ms) | Improvement | Sleep-Capable |",
        "|------------|---------------|------------|-------------|---------------|",
    ]

    for stats in current:
        base = baseline_by_transition.get(stats.transition)

        if base:
            baseline_ms = f"{base.median_ms:.0f}"
            improvement_pct = ((base.median_ms - stats.median_ms) / base.median_ms) * 100
            improvement = f"{improvement_pct:.0f}%"
        else:
            baseline_ms = "N/A"
            improvement = "N/A"

        sleep_capable = "Yes" if stats.is_sleep_capable else "No"
        after_ms = f"{stats.median_ms:.0f}"

        row = (
            f"| {stats.transition} | {baseline_ms} | {after_ms} "
            f"| {improvement} | {sleep_capable} |"
        )
        lines.append(row)

    return "\n".join(lines)
