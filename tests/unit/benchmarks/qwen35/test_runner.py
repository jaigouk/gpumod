"""Tests for Qwen3.5 benchmark runner."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gpumod.benchmarks.qwen35.runner import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_ITERATIONS,
    BenchmarkConfig,
    BenchmarkRunner,
)
from gpumod.benchmarks.qwen35.sampler_config import THINKING_CODING


class TestBenchmarkConfig:
    """Test benchmark configuration."""

    def test_default_iterations(self) -> None:
        assert DEFAULT_ITERATIONS == 15

    def test_default_context_size(self) -> None:
        assert DEFAULT_CONTEXT_SIZE == 32768

    def test_config_defaults(self) -> None:
        config = BenchmarkConfig(model_id="test-model")
        assert config.iterations == DEFAULT_ITERATIONS
        assert config.context_size == DEFAULT_CONTEXT_SIZE
        assert config.sampler == THINKING_CODING

    def test_config_custom_values(self) -> None:
        config = BenchmarkConfig(
            model_id="test-model",
            iterations=10,
            context_size=65536,
        )
        assert config.iterations == 10
        assert config.context_size == 65536


class TestBenchmarkRunner:
    """Test benchmark runner."""

    def test_runner_creation(self) -> None:
        config = BenchmarkConfig(model_id="test-model")
        runner = BenchmarkRunner(config)
        assert runner.config == config

    def test_runner_has_levels(self) -> None:
        config = BenchmarkConfig(model_id="test-model")
        runner = BenchmarkRunner(config)
        assert len(runner.levels) >= 5  # At least 5 levels

    @pytest.mark.asyncio
    async def test_run_single_iteration(self) -> None:
        """Test running a single iteration with mock LLM."""
        config = BenchmarkConfig(model_id="test-model", iterations=1)
        runner = BenchmarkRunner(config)

        # Mock the LLM client
        mock_client = AsyncMock()
        mock_client.generate.return_value = "def add_job(self, job): pass"
        runner.set_client(mock_client)

        results = await runner.run()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_run_multiple_iterations(self) -> None:
        """Test running multiple iterations."""
        config = BenchmarkConfig(model_id="test-model", iterations=3)
        runner = BenchmarkRunner(config)

        mock_client = AsyncMock()
        mock_client.generate.return_value = "def add_job(self, job): pass"
        runner.set_client(mock_client)

        results = await runner.run()
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_run_collects_scores(self) -> None:
        """Test that run collects scores from all iterations."""
        config = BenchmarkConfig(model_id="test-model", iterations=5)
        runner = BenchmarkRunner(config)

        mock_client = AsyncMock()
        mock_client.generate.return_value = "def add_job(self, job): pass"
        runner.set_client(mock_client)

        results = await runner.run()
        scores = [r.total_score for r in results]
        assert len(scores) == 5


class TestBenchmarkRunnerReport:
    """Test benchmark report generation."""

    @pytest.mark.asyncio
    async def test_generate_report(self) -> None:
        config = BenchmarkConfig(model_id="test-model", iterations=3)
        runner = BenchmarkRunner(config)

        mock_client = AsyncMock()
        mock_client.generate.return_value = "def add_job(self, job): pass"
        runner.set_client(mock_client)

        await runner.run()
        report = runner.generate_report()

        assert "model_id" in report
        assert report["model_id"] == "test-model"
        assert "stats" in report
        assert "mean" in report["stats"]
        assert "std" in report["stats"]
        assert "confidence_interval" in report
