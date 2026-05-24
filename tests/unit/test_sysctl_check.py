"""Tests for sysctl_check helper (gpumod-ej0).

The 30 GiB host has accumulated 9+ documented freezes caused by
CUDA-pinned-memory allocation failures on fragmented contiguous pages.
Bumping vm.min_free_kbytes reduces the failure probability by keeping
more high-order pages reserved at all times. This module checks that
the running kernel has the bump applied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpumod.services.sysctl_check import (
    RECOMMENDED_MIN_FREE_KBYTES,
    SysctlCheckResult,
    check_min_free_kbytes,
    read_min_free_kbytes,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestReadMinFreeKbytes:
    def test_reads_value(self, tmp_path: Path) -> None:
        f = tmp_path / "min_free_kbytes"
        f.write_text("1048576\n")
        assert read_min_free_kbytes(f) == 1048576

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        f = tmp_path / "does_not_exist"
        assert read_min_free_kbytes(f) is None

    def test_returns_none_on_malformed(self, tmp_path: Path) -> None:
        f = tmp_path / "min_free_kbytes"
        f.write_text("not-an-int\n")
        assert read_min_free_kbytes(f) is None


class TestCheckMinFreeKbytes:
    def test_passes_when_at_or_above_recommendation(self) -> None:
        result = check_min_free_kbytes(current=RECOMMENDED_MIN_FREE_KBYTES)
        assert result.ok is True
        assert result.current == RECOMMENDED_MIN_FREE_KBYTES
        assert result.threshold == RECOMMENDED_MIN_FREE_KBYTES

    def test_passes_when_well_above_recommendation(self) -> None:
        result = check_min_free_kbytes(current=2097152)  # 2 GiB
        assert result.ok is True

    def test_fails_when_below_recommendation(self) -> None:
        # 67584 ≈ default on the 30 GiB the benchmark host — the value that
        # was insufficient for the documented freezes.
        result = check_min_free_kbytes(current=67584)
        assert result.ok is False
        assert result.current == 67584
        assert result.remediation is not None
        assert "vm.min_free_kbytes" in result.remediation
        # Must point at the durable surface for the fix
        assert "/etc/sysctl.d/" in result.remediation

    def test_fails_when_unreadable(self) -> None:
        result = check_min_free_kbytes(current=None)
        assert result.ok is False
        # Different remediation when we couldn't even read the value
        assert result.remediation is not None
        assert "unable" in result.remediation.lower() or "could not" in result.remediation.lower()

    def test_custom_threshold(self) -> None:
        # Operators on more memory-stable hosts may want a lower threshold
        result = check_min_free_kbytes(current=524288, threshold=524288)
        assert result.ok is True
        # Exactly at threshold passes (>= comparison)

    def test_result_type(self) -> None:
        result = check_min_free_kbytes(current=1048576)
        assert isinstance(result, SysctlCheckResult)
        # The result object exposes the comparison numbers for downstream UIs
        assert hasattr(result, "ok")
        assert hasattr(result, "current")
        assert hasattr(result, "threshold")
        assert hasattr(result, "remediation")
