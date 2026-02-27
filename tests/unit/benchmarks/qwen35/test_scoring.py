"""Tests for Qwen3.5 benchmark scoring."""

from __future__ import annotations

import pytest

from gpumod.benchmarks.qwen35.scoring import (
    BenchmarkResult,
    LevelResult,
    calculate_confidence_interval,
    calculate_stats,
)


class TestLevelResult:
    """Test individual level results."""

    def test_level_result_creation(self) -> None:
        result = LevelResult(level=1, name="Basic Queue", points=25, passed=True)
        assert result.level == 1
        assert result.name == "Basic Queue"
        assert result.points == 25
        assert result.passed is True

    def test_level_result_failed(self) -> None:
        result = LevelResult(level=2, name="Retry", points=0, passed=False)
        assert result.passed is False
        assert result.points == 0


class TestBenchmarkResult:
    """Test benchmark result aggregation."""

    def test_total_score(self) -> None:
        levels = [
            LevelResult(1, "L1", 25, True),
            LevelResult(2, "L2", 25, True),
            LevelResult(3, "L3", 0, False),
        ]
        result = BenchmarkResult(iteration=1, levels=levels)
        assert result.total_score == 50

    def test_levels_passed(self) -> None:
        levels = [
            LevelResult(1, "L1", 25, True),
            LevelResult(2, "L2", 25, True),
            LevelResult(3, "L3", 0, False),
        ]
        result = BenchmarkResult(iteration=1, levels=levels)
        assert result.levels_passed == 2


class TestCalculateStats:
    """Test statistical calculations."""

    def test_mean_calculation(self) -> None:
        scores = [50, 60, 70, 80, 90]
        stats = calculate_stats(scores)
        assert stats["mean"] == 70.0

    def test_std_calculation(self) -> None:
        scores = [50, 50, 50, 50, 50]
        stats = calculate_stats(scores)
        assert stats["std"] == 0.0

    def test_min_max(self) -> None:
        scores = [25, 50, 75, 90, 40]
        stats = calculate_stats(scores)
        assert stats["min"] == 25
        assert stats["max"] == 90

    def test_empty_scores_raises(self) -> None:
        with pytest.raises(ValueError, match="No scores"):
            calculate_stats([])

    def test_single_score(self) -> None:
        stats = calculate_stats([50])
        assert stats["mean"] == 50.0
        assert stats["std"] == 0.0


class TestConfidenceInterval:
    """Test 95% confidence interval calculation."""

    def test_ci_with_normal_data(self) -> None:
        # 15 iterations as per spike recommendation
        scores = [50, 55, 60, 45, 50, 55, 60, 45, 50, 55, 60, 45, 50, 55, 60]
        ci = calculate_confidence_interval(scores, confidence=0.95)
        assert "lower" in ci
        assert "upper" in ci
        assert ci["lower"] < ci["upper"]

    def test_ci_bounds_contain_mean(self) -> None:
        scores = [50, 55, 60, 45, 50, 55, 60, 45, 50, 55, 60, 45, 50, 55, 60]
        stats = calculate_stats(scores)
        ci = calculate_confidence_interval(scores, confidence=0.95)
        assert ci["lower"] <= stats["mean"] <= ci["upper"]

    def test_ci_zero_variance(self) -> None:
        """CI with no variance should have lower == upper == mean."""
        scores = [50, 50, 50, 50, 50]
        ci = calculate_confidence_interval(scores, confidence=0.95)
        assert ci["lower"] == 50.0
        assert ci["upper"] == 50.0

    def test_ci_single_value(self) -> None:
        """Single value should return that value as both bounds."""
        ci = calculate_confidence_interval([50], confidence=0.95)
        assert ci["lower"] == 50.0
        assert ci["upper"] == 50.0
