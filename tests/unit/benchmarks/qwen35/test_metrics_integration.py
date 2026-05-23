"""Tests for MetricsCollector integration with BenchmarkRunner.

TDD Phase: RED - Write failing tests first.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# MetricsCollector protocol tests
# ---------------------------------------------------------------------------


class TestMetricsCollectorProtocol:
    """Tests for MetricsCollector protocol definition."""

    def test_protocol_exists(self) -> None:
        """MetricsCollector protocol is defined."""
        from gpumod.benchmarks.qwen35.runner import MetricsCollector

        assert MetricsCollector is not None

    def test_protocol_has_measure_vram(self) -> None:
        """MetricsCollector has measure_vram method."""
        from gpumod.benchmarks.qwen35.runner import MetricsCollector

        # Check protocol has the method defined
        assert hasattr(MetricsCollector, "measure_vram")

    def test_protocol_has_record_generation(self) -> None:
        """MetricsCollector has record_generation method."""
        from gpumod.benchmarks.qwen35.runner import MetricsCollector

        assert hasattr(MetricsCollector, "record_generation")

    def test_protocol_has_get_iteration_metrics(self) -> None:
        """MetricsCollector has get_iteration_metrics method."""
        from gpumod.benchmarks.qwen35.runner import MetricsCollector

        assert hasattr(MetricsCollector, "get_iteration_metrics")


# ---------------------------------------------------------------------------
# BenchmarkRunner with MetricsCollector tests
# ---------------------------------------------------------------------------


class TestBenchmarkRunnerWithMetrics:
    """Tests for BenchmarkRunner with MetricsCollector integration."""

    def test_accepts_optional_metrics_collector(self) -> None:
        """BenchmarkRunner accepts optional metrics_collector parameter."""
        from gpumod.benchmarks.qwen35.runner import BenchmarkConfig, BenchmarkRunner

        config = BenchmarkConfig(model_id="test-model", iterations=1)

        # Should work without collector
        runner = BenchmarkRunner(config)
        assert runner.metrics_collector is None

        # Should accept collector
        mock_collector = MagicMock()
        runner_with_metrics = BenchmarkRunner(config, metrics_collector=mock_collector)
        assert runner_with_metrics.metrics_collector is mock_collector

    @pytest.mark.asyncio
    async def test_measures_vram_before_iteration(self) -> None:
        """Runner measures VRAM before each iteration starts."""
        from gpumod.benchmarks.qwen35.runner import BenchmarkConfig, BenchmarkRunner

        config = BenchmarkConfig(model_id="test-model", iterations=1)
        mock_collector = MagicMock()
        mock_collector.measure_vram = MagicMock(return_value=18000)
        mock_collector.record_generation = MagicMock()
        mock_collector.get_iteration_metrics = MagicMock(return_value={})

        runner = BenchmarkRunner(config, metrics_collector=mock_collector)

        # Mock LLM client
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="def test(): pass")
        runner.set_client(mock_client)

        await runner.run()

        # Should have called measure_vram at least once
        mock_collector.measure_vram.assert_called()

    @pytest.mark.asyncio
    async def test_records_generation_metrics(self) -> None:
        """Runner records metrics for each generation."""
        from gpumod.benchmarks.qwen35.runner import BenchmarkConfig, BenchmarkRunner

        config = BenchmarkConfig(model_id="test-model", iterations=1)
        mock_collector = MagicMock()
        mock_collector.measure_vram = MagicMock(return_value=18000)
        mock_collector.record_generation = MagicMock()
        mock_collector.get_iteration_metrics = MagicMock(return_value={})

        runner = BenchmarkRunner(config, metrics_collector=mock_collector)

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="response")
        runner.set_client(mock_client)

        await runner.run()

        # Should have recorded generation metrics for each level
        assert mock_collector.record_generation.call_count >= 1

    @pytest.mark.asyncio
    async def test_works_without_collector(self) -> None:
        """Runner works correctly without metrics collector (graceful degradation)."""
        from gpumod.benchmarks.qwen35.runner import BenchmarkConfig, BenchmarkRunner

        config = BenchmarkConfig(model_id="test-model", iterations=1)

        # No collector provided
        runner = BenchmarkRunner(config)

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="def test(): pass")
        runner.set_client(mock_client)

        # Should not raise
        results = await runner.run()
        assert len(results) == 1


# ---------------------------------------------------------------------------
# BenchmarkResult with metrics tests
# ---------------------------------------------------------------------------


class TestBenchmarkResultWithMetrics:
    """Tests for BenchmarkResult containing performance metrics."""

    def test_result_includes_metrics(self) -> None:
        """BenchmarkResult can include iteration_metrics field."""
        from gpumod.benchmarks.qwen35.scoring import BenchmarkResult, LevelResult

        level_results = [
            LevelResult(level=1, name="Test", points=25, passed=True),
        ]

        metrics = {
            "vram_idle_mb": 18000,
            "vram_peak_mb": 18500,
            "mean_tps": 125.5,
            "mean_ttft_ms": 50.2,
        }

        result = BenchmarkResult(
            iteration=1,
            levels=level_results,
            iteration_metrics=metrics,
        )

        assert result.iteration_metrics == metrics


# ---------------------------------------------------------------------------
# Report with aggregated metrics tests
# ---------------------------------------------------------------------------


class TestReportWithMetrics:
    """Tests for generate_report() including metrics statistics."""

    @pytest.mark.asyncio
    async def test_report_includes_metrics_stats(self) -> None:
        """Report includes aggregated statistics for metrics."""
        from gpumod.benchmarks.qwen35.runner import BenchmarkConfig, BenchmarkRunner

        config = BenchmarkConfig(model_id="test-model", iterations=2)

        # Collector that returns realistic metrics
        mock_collector = MagicMock()
        mock_collector.measure_vram = MagicMock(return_value=18000)
        mock_collector.record_generation = MagicMock()
        mock_collector.get_iteration_metrics = MagicMock(
            return_value={
                "vram_idle_mb": 18000,
                "vram_peak_mb": 18500,
                "mean_tps": 125.5,
                "mean_ttft_ms": 50.2,
                "mean_total_ms": 1500.0,
            }
        )

        runner = BenchmarkRunner(config, metrics_collector=mock_collector)

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="response")
        runner.set_client(mock_client)

        await runner.run()
        report = runner.generate_report()

        # Report should have metrics section
        assert "metrics" in report
        metrics = report["metrics"]

        # Should have aggregated stats
        assert "mean_tps" in metrics
        assert "vram_peak_mb" in metrics

    @pytest.mark.asyncio
    async def test_report_metrics_empty_without_collector(self) -> None:
        """Report has empty metrics when no collector provided."""
        from gpumod.benchmarks.qwen35.runner import BenchmarkConfig, BenchmarkRunner

        config = BenchmarkConfig(model_id="test-model", iterations=1)
        runner = BenchmarkRunner(config)

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="response")
        runner.set_client(mock_client)

        await runner.run()
        report = runner.generate_report()

        # Report should have empty metrics
        assert report.get("metrics") == {} or report.get("metrics") is None


# ---------------------------------------------------------------------------
# DefaultMetricsCollector tests
# ---------------------------------------------------------------------------


class TestDefaultMetricsCollector:
    """Tests for the default MetricsCollector implementation."""

    def test_default_collector_exists(self) -> None:
        """DefaultMetricsCollector implementation exists."""
        from gpumod.benchmarks.qwen35.metrics_collector import DefaultMetricsCollector

        collector = DefaultMetricsCollector()
        assert collector is not None

    def test_measure_vram_returns_int_or_none(self) -> None:
        """measure_vram returns int or None."""
        from gpumod.benchmarks.qwen35.metrics_collector import DefaultMetricsCollector

        collector = DefaultMetricsCollector()
        result = collector.measure_vram()

        assert result is None or isinstance(result, int)

    def test_record_generation_stores_data(self) -> None:
        """record_generation stores generation data."""
        from gpumod.benchmarks.qwen35.metrics_collector import DefaultMetricsCollector

        collector = DefaultMetricsCollector()

        collector.record_generation(
            tokens=100,
            duration_seconds=0.8,
            ttft_seconds=0.1,
        )

        metrics = collector.get_iteration_metrics()
        assert metrics is not None
        assert "mean_tps" in metrics

    def test_get_iteration_metrics_and_reset(self) -> None:
        """get_iteration_metrics returns data and resets for next iteration."""
        from gpumod.benchmarks.qwen35.metrics_collector import DefaultMetricsCollector

        collector = DefaultMetricsCollector()

        collector.record_generation(tokens=100, duration_seconds=0.8, ttft_seconds=0.1)
        collector.record_generation(tokens=150, duration_seconds=1.0, ttft_seconds=0.15)

        metrics = collector.get_iteration_metrics()
        assert metrics["mean_tps"] > 0

        # After get_iteration_metrics, should be reset
        empty_metrics = collector.get_iteration_metrics()
        assert empty_metrics.get("mean_tps") is None or empty_metrics == {}

    def test_calculates_mean_tps_across_generations(self) -> None:
        """Calculates mean TPS across multiple generations in iteration."""
        from gpumod.benchmarks.qwen35.metrics_collector import DefaultMetricsCollector

        collector = DefaultMetricsCollector()

        # TPS1 = 100/0.5 = 200, TPS2 = 100/1.0 = 100
        collector.record_generation(tokens=100, duration_seconds=0.5, ttft_seconds=0.1)
        collector.record_generation(tokens=100, duration_seconds=1.0, ttft_seconds=0.1)

        metrics = collector.get_iteration_metrics()

        # Mean should be (200 + 100) / 2 = 150
        assert abs(metrics["mean_tps"] - 150.0) < 0.1

    def test_get_iteration_metrics_omits_mtp_aggregates_when_absent(self) -> None:
        # gpumod-76l.3: non-MTP runs do not pass draft_n / draft_n_accepted.
        # Output must NOT carry MTP keys so README tables for non-MTP models
        # stay clean.
        from gpumod.benchmarks.qwen35.metrics_collector import DefaultMetricsCollector

        collector = DefaultMetricsCollector()
        collector.record_generation(tokens=100, duration_seconds=0.5, ttft_seconds=0.1)

        metrics = collector.get_iteration_metrics()

        assert "mean_draft_acceptance" not in metrics
        assert "total_draft_n" not in metrics
        assert "total_draft_accepted" not in metrics

    def test_get_iteration_metrics_includes_mtp_aggregates_when_present(self) -> None:
        # MTP runs pass draft_n + draft_n_accepted. Output exposes:
        # - total_draft_n (sum across generations)
        # - total_draft_accepted (sum across generations)
        # - mean_draft_acceptance (mean of per-call acceptance ratio)
        from gpumod.benchmarks.qwen35.metrics_collector import DefaultMetricsCollector

        collector = DefaultMetricsCollector()
        # Gen 1: 44 drafted, 41 accepted => 93.2%
        collector.record_generation(
            tokens=64, duration_seconds=8.0, ttft_seconds=0.1,
            draft_n=44, draft_n_accepted=41,
        )
        # Gen 2: 100 drafted, 80 accepted => 80.0%
        collector.record_generation(
            tokens=80, duration_seconds=10.0, ttft_seconds=0.1,
            draft_n=100, draft_n_accepted=80,
        )

        metrics = collector.get_iteration_metrics()

        assert metrics["total_draft_n"] == 144
        assert metrics["total_draft_accepted"] == 121
        # Mean of per-call ratios: (41/44 + 80/100) / 2 = (0.9318 + 0.8) / 2 = 0.8659
        assert abs(metrics["mean_draft_acceptance"] - 0.8659) < 0.001

    def test_record_generation_accepts_zero_draft_n_without_division_error(self) -> None:
        # Edge case: MTP can emit a request where no drafts were generated
        # (e.g. very short response). Acceptance ratio for that call is
        # treated as 0.0 so it cannot raise ZeroDivisionError.
        from gpumod.benchmarks.qwen35.metrics_collector import DefaultMetricsCollector

        collector = DefaultMetricsCollector()
        collector.record_generation(
            tokens=2, duration_seconds=0.2, ttft_seconds=0.05,
            draft_n=0, draft_n_accepted=0,
        )

        metrics = collector.get_iteration_metrics()

        assert metrics["total_draft_n"] == 0
        assert metrics["total_draft_accepted"] == 0
        assert metrics["mean_draft_acceptance"] == 0.0
