"""Tests for RAM preflight check (gpumod-bfx).

Tests cover:
- RAMCheck preflight validation
- Passing when RAM is sufficient
- Warning when RAM is low
- Failing when RAM is critical
- Configurable thresholds
- Graceful handling of unreadable/malformed /proc/meminfo
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gpumod.preflight.ram_check import RAMCheck

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_meminfo(path: Path, mem_available_mb: int, mem_total_mb: int = 32768) -> None:
    """Write a synthetic /proc/meminfo file for testing."""
    path.write_text(
        f"MemTotal:       {mem_total_mb * 1024} kB\n"
        f"MemFree:        {mem_available_mb * 512} kB\n"
        f"MemAvailable:   {mem_available_mb * 1024} kB\n"
        f"SwapFree:       8388608 kB\n"
    )


def _make_mock_service():
    """Create a minimal mock service for RAM check."""
    from unittest.mock import MagicMock

    service = MagicMock()
    service.id = "test-svc"
    service.vram_mb = 8000
    return service


# ---------------------------------------------------------------------------
# RAMCheck Tests
# ---------------------------------------------------------------------------


class TestRAMCheck:
    """Tests for RAMCheck preflight validation."""

    def test_name_property(self, tmp_path: Path) -> None:
        """Check has correct name."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=8192)
        check = RAMCheck(meminfo_path=meminfo)
        assert check.name == "ram"

    @pytest.mark.asyncio
    async def test_passes_when_ram_sufficient(self, tmp_path: Path) -> None:
        """Check passes when MemAvailable is well above thresholds."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=8192)

        check = RAMCheck(meminfo_path=meminfo)
        result = await check.check(_make_mock_service())

        assert result.passed is True
        assert result.severity == "info"

    @pytest.mark.asyncio
    async def test_warns_when_ram_low(self, tmp_path: Path) -> None:
        """Check warns when MemAvailable is below warn threshold but above min."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=3000)  # Below 4096, above 2048

        check = RAMCheck(meminfo_path=meminfo)
        result = await check.check(_make_mock_service())

        assert result.passed is True
        assert result.severity == "warning"
        assert "3000" in result.message or "ram" in result.message.lower()

    @pytest.mark.asyncio
    async def test_fails_when_ram_critical(self, tmp_path: Path) -> None:
        """Check fails when MemAvailable is below min threshold."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=512)  # Below 1024

        check = RAMCheck(meminfo_path=meminfo)
        result = await check.check(_make_mock_service())

        assert result.passed is False
        assert result.severity == "error"

    @pytest.mark.asyncio
    async def test_error_includes_remediation(self, tmp_path: Path) -> None:
        """Error result includes actionable remediation text."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=512)

        check = RAMCheck(meminfo_path=meminfo)
        result = await check.check(_make_mock_service())

        assert result.passed is False
        assert result.remediation is not None
        # Should mention available memory and suggest action
        assert "512" in result.remediation or "available" in result.remediation.lower()

    @pytest.mark.asyncio
    async def test_custom_thresholds(self, tmp_path: Path) -> None:
        """Custom min and warn thresholds are respected."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=1500)

        # With default thresholds (min=2048), 1500 MB would be error
        # With custom thresholds (min=1024), 1500 MB should be warning (< warn=2048)
        check = RAMCheck(
            min_free_mb=1024,
            warn_free_mb=2048,
            meminfo_path=meminfo,
        )
        result = await check.check(_make_mock_service())

        assert result.passed is True
        assert result.severity == "warning"

    @pytest.mark.asyncio
    async def test_handles_unreadable_procmeminfo(self, tmp_path: Path) -> None:
        """Gracefully handles nonexistent /proc/meminfo path."""
        meminfo = tmp_path / "nonexistent_meminfo"
        # Don't create the file

        check = RAMCheck(meminfo_path=meminfo)
        result = await check.check(_make_mock_service())

        # Should warn, not error — don't block service on unreadable meminfo
        assert result.passed is True
        assert result.severity == "warning"

    @pytest.mark.asyncio
    async def test_handles_malformed_procmeminfo(self, tmp_path: Path) -> None:
        """Gracefully handles garbage content in meminfo."""
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("this is not valid meminfo content\nfoo bar baz\n")

        check = RAMCheck(meminfo_path=meminfo)
        result = await check.check(_make_mock_service())

        # Should warn — MemAvailable not found
        assert result.passed is True
        assert result.severity == "warning"

    @pytest.mark.asyncio
    async def test_passes_at_exact_warn_threshold(self, tmp_path: Path) -> None:
        """Exactly at warn threshold should be info (pass), not warning."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=4096)  # Exactly at warn threshold

        check = RAMCheck(meminfo_path=meminfo)
        result = await check.check(_make_mock_service())

        assert result.passed is True
        assert result.severity == "info"

    @pytest.mark.asyncio
    async def test_fails_at_exact_min_threshold(self, tmp_path: Path) -> None:
        """Exactly at min threshold should be warning, not error."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=1024)  # Exactly at min threshold

        check = RAMCheck(meminfo_path=meminfo)
        result = await check.check(_make_mock_service())

        # 1024 is NOT less than 1024, so should pass as warning (< 4096)
        assert result.passed is True
        assert result.severity == "warning"

    @pytest.mark.asyncio
    async def test_error_message_includes_threshold(self, tmp_path: Path) -> None:
        """Error message includes the minimum threshold value."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=500)

        check = RAMCheck(meminfo_path=meminfo)
        result = await check.check(_make_mock_service())

        assert result.passed is False
        assert "500" in result.message
        assert "1024" in result.message  # Default min threshold

    @pytest.mark.asyncio
    async def test_warning_message_includes_available_ram(self, tmp_path: Path) -> None:
        """Warning message includes the MemAvailable value."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=2000)

        check = RAMCheck(meminfo_path=meminfo)
        result = await check.check(_make_mock_service())

        assert result.passed is True
        assert result.severity == "warning"
        assert "2000" in result.message

    @pytest.mark.asyncio
    async def test_remediation_includes_total_ram(self, tmp_path: Path) -> None:
        """Error remediation includes MemTotal for context."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=500, mem_total_mb=32768)

        check = RAMCheck(meminfo_path=meminfo)
        result = await check.check(_make_mock_service())

        assert result.remediation is not None
        assert "32768" in result.remediation  # MemTotal
        assert "500" in result.remediation  # MemAvailable
