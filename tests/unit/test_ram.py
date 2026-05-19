"""Tests for RAMTracker, InsufficientRAMError, and check_ram_safeguard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpumod.models import RAMUsage
from gpumod.services.ram import (
    RAM_ABSOLUTE_HEADROOM_MB,
    RAM_HARD_FLOOR_MB,
    RAM_HEADROOM_RATIO,
    InsufficientRAMError,
    RAMTracker,
    check_ram_safeguard,
)

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


def _tracker_with_avail_mb(tmp_path: Path, avail_mb: int) -> RAMTracker:
    """Build a RAMTracker that reports a controlled MemAvailable in MB."""
    f = tmp_path / "meminfo"
    avail_kb = avail_mb * 1024
    f.write_text(
        f"MemTotal:       30000000 kB\nMemFree:           20000 kB\nMemAvailable:  {avail_kb} kB\n"
    )
    return RAMTracker(meminfo_path=f)


class TestCheckRamSafeguard:
    """Freeze-aware RAM safeguard — extracted from ServiceManager (gpumod-ecr).

    Same contract: returns None if safe, else an error message string.
    """

    async def test_returns_none_when_ram_is_comfortable(self, tmp_path: Path) -> None:
        # 28 GB available, asking for a 22 GB-VRAM service: required ~= 11.6 GB
        tracker = _tracker_with_avail_mb(tmp_path, 28_000)
        result = await check_ram_safeguard(incoming_vram_mb=22_000, ram=tracker)
        assert result is None

    async def test_returns_error_when_under_hard_floor(self, tmp_path: Path) -> None:
        # 4 GB available, below the 6 GB hard floor → always refuse
        tracker = _tracker_with_avail_mb(tmp_path, 4_000)
        result = await check_ram_safeguard(incoming_vram_mb=22_000, ram=tracker)
        assert result is not None
        assert "hard floor" in result.lower()
        assert "4000" in result
        assert str(RAM_HARD_FLOOR_MB) in result

    async def test_returns_error_when_under_headroom_estimate(self, tmp_path: Path) -> None:
        # Above hard floor (6 GB) but below 22 GB * 0.3 + 5000 = 11.6 GB headroom
        tracker = _tracker_with_avail_mb(tmp_path, 8_000)
        result = await check_ram_safeguard(incoming_vram_mb=22_000, ram=tracker)
        assert result is not None
        assert "headroom" in result.lower() or "available" in result.lower()
        assert "8000" in result

    async def test_zero_vram_only_checks_hard_floor(self, tmp_path: Path) -> None:
        # vram_mb=0 means no incoming service; should only check hard floor
        # Above floor: pass.
        tracker_ok = _tracker_with_avail_mb(tmp_path, 8_000)
        assert await check_ram_safeguard(incoming_vram_mb=0, ram=tracker_ok) is None
        # Below floor: still fail.
        tmp2 = tmp_path / "low"
        tmp2.mkdir()
        tracker_low = _tracker_with_avail_mb(tmp2, 3_000)
        result = await check_ram_safeguard(incoming_vram_mb=0, ram=tracker_low)
        assert result is not None
        assert "hard floor" in result.lower()

    async def test_constants_are_exported(self) -> None:
        """Constants must be re-importable from gpumod.services.ram."""
        # If these have moved or were renamed, tests fail to import.
        assert RAM_HEADROOM_RATIO == 0.3
        assert RAM_ABSOLUTE_HEADROOM_MB == 5_000
        assert RAM_HARD_FLOOR_MB == 6_000

    async def test_hard_floor_mb_override_raises_threshold(self, tmp_path: Path) -> None:
        """Caller can pass a higher hard_floor_mb (e.g. via CLI --hard-floor)."""
        # 8 GB available, default hard floor is 6 GB → passes
        tracker = _tracker_with_avail_mb(tmp_path, 8_000)
        assert await check_ram_safeguard(incoming_vram_mb=0, ram=tracker) is None
        # Override hard floor to 10 GB → 8 GB now fails
        result = await check_ram_safeguard(incoming_vram_mb=0, ram=tracker, hard_floor_mb=10_000)
        assert result is not None
        assert "10000" in result
        assert "hard floor" in result.lower()

    async def test_hard_floor_mb_override_lower_value_ignored(self, tmp_path: Path) -> None:
        """An override BELOW the default still applies — the override IS the floor."""
        # 4 GB available, default hard floor 6 GB would refuse
        tracker = _tracker_with_avail_mb(tmp_path, 4_000)
        # Override to 2 GB allows 4 GB through (no headroom check since vram=0)
        result = await check_ram_safeguard(incoming_vram_mb=0, ram=tracker, hard_floor_mb=2_000)
        assert result is None
