"""Tests for mode switch benchmarking utilities.

Follows TDD: tests written first (Red), then implementation (Green).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from unittest.mock import AsyncMock

import pytest

from gpumod.benchmark.mode_switch import (
    ModeSwitchBenchmark,
    ModeSwitchResult,
    TransitionStats,
    generate_comparison_table,
)

# ---------------------------------------------------------------------------
# ModeSwitchResult dataclass tests
# ---------------------------------------------------------------------------


class TestModeSwitchResult:
    """Tests for the ModeSwitchResult dataclass."""

    def test_creates_with_required_fields(self) -> None:
        """Result captures transition, timing, and service counts."""
        result = ModeSwitchResult(
            from_mode="code",
            to_mode="rag",
            total_ms=4500.0,
            services_slept=["vllm-devstral"],
            services_stopped=[],
            services_woken=[],
            services_started=["vllm-embed"],
        )

        assert result.from_mode == "code"
        assert result.to_mode == "rag"
        assert result.total_ms == 4500.0
        assert result.services_slept == ["vllm-devstral"]
        assert result.services_started == ["vllm-embed"]

    def test_transition_property(self) -> None:
        """Transition string is formatted correctly."""
        result = ModeSwitchResult(
            from_mode="code",
            to_mode="rag",
            total_ms=1234.5,
            services_slept=[],
            services_stopped=[],
            services_woken=[],
            services_started=[],
        )
        assert result.transition == "code → rag"

    def test_is_sleep_capable_transition(self) -> None:
        """Detects when transition used sleep/wake vs stop/start."""
        # Sleep-capable: used sleep or wake
        sleep_result = ModeSwitchResult(
            from_mode="code",
            to_mode="rag",
            total_ms=3000.0,
            services_slept=["vllm-devstral"],
            services_stopped=[],
            services_woken=[],
            services_started=[],
        )
        assert sleep_result.is_sleep_capable is True

        # Non-sleep: only used stop/start
        restart_result = ModeSwitchResult(
            from_mode="blank",
            to_mode="code",
            total_ms=22000.0,
            services_slept=[],
            services_stopped=[],
            services_woken=[],
            services_started=["vllm-devstral"],
        )
        assert restart_result.is_sleep_capable is False

    def test_serializes_to_dict(self) -> None:
        """Result can be serialized for JSON output."""
        result = ModeSwitchResult(
            from_mode="code",
            to_mode="rag",
            total_ms=4500.0,
            services_slept=["vllm-devstral"],
            services_stopped=[],
            services_woken=["vllm-embed"],
            services_started=[],
        )

        d = asdict(result)
        assert d["from_mode"] == "code"
        assert d["to_mode"] == "rag"
        assert d["total_ms"] == 4500.0

        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert "code" in json_str


# ---------------------------------------------------------------------------
# TransitionStats tests
# ---------------------------------------------------------------------------


class TestTransitionStats:
    """Tests for aggregated transition statistics."""

    def test_computes_stats_from_multiple_runs(self) -> None:
        """Stats aggregate min/max/mean from multiple results."""
        results = [
            ModeSwitchResult(
                from_mode="code",
                to_mode="rag",
                total_ms=3500.0,
                services_slept=["vllm"],
                services_stopped=[],
                services_woken=[],
                services_started=[],
            ),
            ModeSwitchResult(
                from_mode="code",
                to_mode="rag",
                total_ms=4500.0,
                services_slept=["vllm"],
                services_stopped=[],
                services_woken=[],
                services_started=[],
            ),
            ModeSwitchResult(
                from_mode="code",
                to_mode="rag",
                total_ms=4000.0,
                services_slept=["vllm"],
                services_stopped=[],
                services_woken=[],
                services_started=[],
            ),
        ]

        stats = TransitionStats.from_results(results)

        assert stats.transition == "code → rag"
        assert stats.runs == 3
        assert stats.min_ms == 3500.0
        assert stats.max_ms == 4500.0
        assert stats.mean_ms == 4000.0  # (3500 + 4500 + 4000) / 3
        assert stats.is_sleep_capable is True

    def test_computes_median(self) -> None:
        """Stats compute median correctly for odd and even counts."""
        # Odd count
        results_odd = [
            ModeSwitchResult("a", "b", 1000.0, [], [], [], []),
            ModeSwitchResult("a", "b", 2000.0, [], [], [], []),
            ModeSwitchResult("a", "b", 9000.0, [], [], [], []),  # outlier
        ]
        stats = TransitionStats.from_results(results_odd)
        assert stats.median_ms == 2000.0

        # Even count
        results_even = [
            ModeSwitchResult("a", "b", 1000.0, [], [], [], []),
            ModeSwitchResult("a", "b", 2000.0, [], [], [], []),
            ModeSwitchResult("a", "b", 3000.0, [], [], [], []),
            ModeSwitchResult("a", "b", 4000.0, [], [], [], []),
        ]
        stats = TransitionStats.from_results(results_even)
        assert stats.median_ms == 2500.0  # (2000 + 3000) / 2


# ---------------------------------------------------------------------------
# ModeSwitchBenchmark tests
# ---------------------------------------------------------------------------


class TestModeSwitchBenchmark:
    """Tests for the benchmark runner."""

    @pytest.fixture
    def mock_manager(self) -> AsyncMock:
        """Create a mock ServiceManager."""
        manager = AsyncMock()
        manager.switch_mode = AsyncMock()
        return manager

    @pytest.mark.asyncio
    async def test_runs_single_transition(self, mock_manager: AsyncMock) -> None:
        """Benchmark times a single mode transition."""
        # Mock a successful mode switch that returns timing info
        from gpumod.models import ModeResult

        mock_manager.switch_mode.return_value = ModeResult(
            success=True,
            mode_id="rag",
            started=["vllm-embed"],
            stopped=["vllm-devstral"],
        )

        benchmark = ModeSwitchBenchmark(manager=mock_manager)

        result = await benchmark.run_transition("code", "rag")

        assert result.from_mode == "code"
        assert result.to_mode == "rag"
        assert result.total_ms >= 0  # Should capture some time
        mock_manager.switch_mode.assert_called_once_with("rag")

    @pytest.mark.asyncio
    async def test_handles_failed_transition(self, mock_manager: AsyncMock) -> None:
        """Benchmark captures failure info."""
        from gpumod.models import ModeResult

        mock_manager.switch_mode.return_value = ModeResult(
            success=False,
            mode_id="rag",
            errors=["VRAM exceeded"],
        )

        benchmark = ModeSwitchBenchmark(manager=mock_manager)

        result = await benchmark.run_transition("code", "rag")

        assert result.total_ms >= 0
        assert result.error == "VRAM exceeded"

    @pytest.mark.asyncio
    async def test_runs_multiple_warmup_iterations(self, mock_manager: AsyncMock) -> None:
        """Benchmark can run warmup iterations before measurement."""
        from gpumod.models import ModeResult

        mock_manager.switch_mode.return_value = ModeResult(
            success=True,
            mode_id="rag",
            started=[],
            stopped=[],
        )

        benchmark = ModeSwitchBenchmark(manager=mock_manager)

        # Run with 2 warmup + 3 measured runs
        results = await benchmark.run_benchmark(
            transitions=[("code", "rag")],
            warmup_runs=2,
            measured_runs=3,
        )

        # Should have 3 measured results (warmup discarded)
        assert len(results) == 3
        # Total calls = warmup + measured per transition
        assert mock_manager.switch_mode.call_count == 5

    @pytest.mark.asyncio
    async def test_captures_service_actions(self, mock_manager: AsyncMock) -> None:
        """Benchmark captures which services were slept/woken/started/stopped."""
        from gpumod.models import ModeResult

        # Simulate sleep-aware switch
        mock_manager.switch_mode.return_value = ModeResult(
            success=True,
            mode_id="rag",
            started=["vllm-embed", "vllm-devstral"],  # woken + started combined
            stopped=["vllm-code"],  # slept + stopped combined
        )

        benchmark = ModeSwitchBenchmark(manager=mock_manager)

        result = await benchmark.run_transition("code", "rag")

        # ModeResult combines slept/stopped and woken/started
        # The benchmark should pass through what the manager reports
        assert "vllm-embed" in result.services_started or "vllm-embed" in result.services_woken


# ---------------------------------------------------------------------------
# Comparison table generation
# ---------------------------------------------------------------------------


class TestGenerateComparisonTable:
    """Tests for markdown table generation."""

    def test_generates_markdown_table(self) -> None:
        """Generates a markdown comparison table."""
        baseline_stats = [
            TransitionStats(
                transition="code → rag",
                runs=3,
                min_ms=30000.0,
                max_ms=35000.0,
                mean_ms=33000.0,
                median_ms=33000.0,
                is_sleep_capable=False,
            ),
        ]
        current_stats = [
            TransitionStats(
                transition="code → rag",
                runs=3,
                min_ms=3000.0,
                max_ms=5000.0,
                mean_ms=4000.0,
                median_ms=4000.0,
                is_sleep_capable=True,
            ),
        ]

        table = generate_comparison_table(baseline_stats, current_stats)

        # Should be valid markdown
        assert "| Transition |" in table
        assert "code → rag" in table
        assert "Improvement" in table or "improvement" in table.lower()

    def test_calculates_improvement_percentage(self) -> None:
        """Table shows percentage improvement."""
        baseline = [
            TransitionStats("a → b", 3, 10000, 10000, 10000, 10000, False),
        ]
        current = [
            TransitionStats("a → b", 3, 2000, 2000, 2000, 2000, True),
        ]

        table = generate_comparison_table(baseline, current)

        # 10000 → 2000 = 80% improvement
        assert "80%" in table

    def test_handles_missing_baseline(self) -> None:
        """Table handles transitions without baseline data."""
        baseline: list[TransitionStats] = []
        current = [
            TransitionStats("new → mode", 3, 5000, 5000, 5000, 5000, True),
        ]

        table = generate_comparison_table(baseline, current)

        # Should still generate table
        assert "new → mode" in table
        assert "N/A" in table or "-" in table  # No baseline to compare
