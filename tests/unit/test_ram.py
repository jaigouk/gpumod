"""Tests for RAMTracker and InsufficientRAMError."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpumod.models import RAMUsage
from gpumod.services.ram import InsufficientRAMError, RAMTracker

if TYPE_CHECKING:
    from pathlib import Path

MEMINFO_SAMPLE = """\
MemTotal:       30000000 kB
MemFree:         5120000 kB
MemAvailable:   20480000 kB
Buffers:          200000 kB
Cached:         12000000 kB
SwapCached:           0 kB
Active:          4000000 kB
"""


class TestRAMTrackerGetUsage:
    async def test_parses_meminfo_into_mb(self, tmp_path: Path) -> None:
        f = tmp_path / "meminfo"
        f.write_text(MEMINFO_SAMPLE)
        tracker = RAMTracker(meminfo_path=f)
        usage = await tracker.get_usage()
        assert isinstance(usage, RAMUsage)
        assert usage.total_mb == 29296  # 30000000 // 1024
        assert usage.available_mb == 20000  # 20480000 // 1024
        assert usage.free_mb == 5000  # 5120000 // 1024

    async def test_handles_missing_fields(self, tmp_path: Path) -> None:
        f = tmp_path / "meminfo"
        f.write_text("MemTotal:       1024 kB\n")  # only one field
        tracker = RAMTracker(meminfo_path=f)
        usage = await tracker.get_usage()
        assert usage.total_mb == 1
        assert usage.available_mb == 0
        assert usage.free_mb == 0

    async def test_ignores_malformed_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "meminfo"
        f.write_text(
            "MemTotal:       1024 kB\n"
            "GarbageLine without colon\n"
            "MemAvailable:   not_a_number kB\n"
            "MemFree:        512 kB\n"
        )
        tracker = RAMTracker(meminfo_path=f)
        usage = await tracker.get_usage()
        assert usage.total_mb == 1
        assert usage.free_mb == 0  # 512 // 1024 = 0
        assert usage.available_mb == 0  # malformed → default


class TestWaitForRamRelease:
    async def test_returns_true_immediately_when_threshold_met(self, tmp_path: Path) -> None:
        f = tmp_path / "meminfo"
        # 20 GB available → satisfies 10 GB + 1 GB margin
        f.write_text(
            "MemTotal:       30000000 kB\n"
            "MemFree:         5120000 kB\n"
            "MemAvailable:   20480000 kB\n"
        )
        tracker = RAMTracker(meminfo_path=f)
        ok = await tracker.wait_for_ram_release(
            required_mb=10_000, timeout_s=2.0, poll_interval_s=0.1
        )
        assert ok is True

    async def test_returns_false_on_timeout(self, tmp_path: Path) -> None:
        f = tmp_path / "meminfo"
        # 500 MB available → never meets 10 GB requirement
        f.write_text(
            "MemTotal:       30000000 kB\n"
            "MemFree:           20000 kB\n"
            "MemAvailable:     512000 kB\n"
        )
        tracker = RAMTracker(meminfo_path=f)
        ok = await tracker.wait_for_ram_release(
            required_mb=10_000, timeout_s=0.3, poll_interval_s=0.05
        )
        assert ok is False


class TestInsufficientRAMError:
    def test_default_message_includes_numbers(self) -> None:
        err = InsufficientRAMError(required_mb=12000, available_mb=4000)
        msg = str(err)
        assert "12000" in msg
        assert "4000" in msg
        assert err.required_mb == 12000
        assert err.available_mb == 4000

    def test_custom_message_overrides_default(self) -> None:
        err = InsufficientRAMError(required_mb=1, available_mb=2, message="custom warning")
        assert str(err) == "custom warning"
