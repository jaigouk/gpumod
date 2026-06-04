"""Tests for performance metrics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gpumod.benchmarks.coding.performance_metrics import (
    PerformanceMetrics,
    measure_latency,
    measure_tps,
    measure_vram,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_metrics_creation(self) -> None:
        metrics = PerformanceMetrics(
            tps=100.5,
            vram_mb=18000,
            ttft_ms=150.0,
            total_ms=5000.0,
        )
        assert metrics.tps == 100.5
        assert metrics.vram_mb == 18000
        assert metrics.ttft_ms == 150.0
        assert metrics.total_ms == 5000.0

    def test_metrics_optional_vram(self) -> None:
        """VRAM should be optional (for systems without nvidia-smi)."""
        metrics = PerformanceMetrics(
            tps=100.5,
            vram_mb=None,
            ttft_ms=150.0,
            total_ms=5000.0,
        )
        assert metrics.vram_mb is None


class TestMeasureTPS:
    """Test TPS calculation."""

    def test_basic_tps(self) -> None:
        """100 tokens in 1 second = 100 TPS."""
        tps = measure_tps(tokens=100, duration_seconds=1.0)
        assert tps == 100.0

    def test_fractional_tps(self) -> None:
        """500 tokens in 2 seconds = 250 TPS."""
        tps = measure_tps(tokens=500, duration_seconds=2.0)
        assert tps == 250.0

    def test_zero_duration_raises(self) -> None:
        """Zero duration should raise ValueError."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            measure_tps(tokens=100, duration_seconds=0.0)

    def test_negative_duration_raises(self) -> None:
        """Negative duration should raise ValueError."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            measure_tps(tokens=100, duration_seconds=-1.0)


class TestMeasureLatency:
    """Test latency calculation."""

    def test_basic_latency(self) -> None:
        """Test TTFT and total latency calculation."""
        result = measure_latency(
            start_time=0.0,
            first_token_time=0.15,
            end_time=5.0,
        )
        assert result["ttft_ms"] == 150.0
        assert result["total_ms"] == 5000.0

    def test_latency_with_real_times(self) -> None:
        """Test with realistic timestamp values."""
        result = measure_latency(
            start_time=1000.0,
            first_token_time=1000.2,
            end_time=1005.0,
        )
        assert result["ttft_ms"] == pytest.approx(200.0, rel=0.01)
        assert result["total_ms"] == pytest.approx(5000.0, rel=0.01)


class TestMeasureVRAM:
    """Test VRAM measurement via nvidia-smi."""

    def test_measure_vram_success(self) -> None:
        """Test successful VRAM measurement."""
        mock_result = MagicMock()
        mock_result.stdout = "18000\n"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            vram = measure_vram()
            assert vram == 18000

    def test_measure_vram_nvidia_smi_not_found(self) -> None:
        """Return None when nvidia-smi is not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            vram = measure_vram()
            assert vram is None

    def test_measure_vram_nvidia_smi_error(self) -> None:
        """Return None when nvidia-smi returns error."""
        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            vram = measure_vram()
            assert vram is None

    def test_measure_vram_invalid_output(self) -> None:
        """Return None when nvidia-smi output is not a number."""
        mock_result = MagicMock()
        mock_result.stdout = "N/A\n"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            vram = measure_vram()
            assert vram is None
