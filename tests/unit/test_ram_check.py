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

from gpumod.preflight.ram_check import (
    DEFAULT_MIN_FREE_MB,
    DEFAULT_MMAP_OVERHEAD_FACTOR,
    RAMCheck,
)

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


def _make_mock_service(model_path: str | None = None):
    """Create a minimal mock service for RAM check.

    When ``model_path`` is provided, populate ``extra_config['unit_vars']``
    so the model-file-size aware check (gpumod-lgt) can see it.
    """
    from unittest.mock import MagicMock

    service = MagicMock()
    service.id = "test-svc"
    service.vram_mb = 8000
    if model_path is not None:
        service.extra_config = {"unit_vars": {"model_path": model_path}}
    else:
        service.extra_config = {}
    return service


def _make_gguf(path: Path, size_mb: int) -> Path:
    """Create a fake GGUF file of the given size (in MB) for size-aware tests."""
    path.write_bytes(b"")
    # Use truncate to create a sparse file of the requested size without
    # actually allocating the bytes — perfect for st_size testing.
    with path.open("r+b") as f:
        f.truncate(size_mb * 1024 * 1024)
    return path


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


# ---------------------------------------------------------------------------
# gpumod-lgt: model file size aware checks
# ---------------------------------------------------------------------------


class TestRAMCheckModelFileSize:
    """RAMCheck should refuse to start a service when MemAvailable cannot
    accommodate the mmap of the service's model file plus a baseline cushion.

    Triggered the 2026-05-24 OOM hard reboot: an 18 GB GGUF loaded via
    llama.cpp mmap consumed page cache faster than MemAvailable predicted,
    and the existing threshold-only check passed (MemAvailable > 4 GB warn).
    """

    @pytest.mark.asyncio
    async def test_fails_when_model_file_exceeds_available_ram(self, tmp_path: Path) -> None:
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=10000)  # plenty by threshold
        gguf = _make_gguf(tmp_path / "model.gguf", size_mb=18000)  # 18 GB

        check = RAMCheck(meminfo_path=meminfo, mmap_overhead_factor=1.1)
        result = await check.check(_make_mock_service(model_path=str(gguf)))

        # 18000 * 1.1 = 19800 MB required, only 10000 MB available
        assert result.passed is False
        assert result.severity == "error"
        # Message should reference the file size and the gap
        assert "18000" in result.message or "19" in result.message
        assert result.remediation is not None
        assert "model" in result.remediation.lower()

    @pytest.mark.asyncio
    async def test_passes_when_model_fits_with_overhead(self, tmp_path: Path) -> None:
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=25000)  # 25 GB
        gguf = _make_gguf(tmp_path / "model.gguf", size_mb=18000)  # 18 GB

        check = RAMCheck(meminfo_path=meminfo, mmap_overhead_factor=1.1)
        result = await check.check(_make_mock_service(model_path=str(gguf)))

        # 18000 * 1.1 = 19800 + min_free 1024 = 20824 MB. 25000 > 20824 ✓
        assert result.passed is True
        assert result.severity == "info"

    @pytest.mark.asyncio
    async def test_falls_back_when_no_model_path(self, tmp_path: Path) -> None:
        """Services without a model_path use the threshold-only check."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=8000)

        check = RAMCheck(meminfo_path=meminfo, mmap_overhead_factor=1.1)
        # No model_path → check falls back to threshold semantics
        result = await check.check(_make_mock_service())

        assert result.passed is True
        assert result.severity == "info"

    @pytest.mark.asyncio
    async def test_handles_nonexistent_model_file_gracefully(self, tmp_path: Path) -> None:
        """If model_path is configured but the file doesn't exist, the check
        falls back to threshold-only (don't pre-fail when the downloader
        hasn't run yet)."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=8000)

        check = RAMCheck(meminfo_path=meminfo, mmap_overhead_factor=1.1)
        result = await check.check(
            _make_mock_service(model_path=str(tmp_path / "nonexistent.gguf"))
        )

        # File doesn't exist → can't size-check → fall back to threshold,
        # which passes at 8000 MB.
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_mmap_overhead_factor_configurable(self, tmp_path: Path) -> None:
        """Higher factor refuses tighter scenarios that the default would
        allow — operators on fragmentation-prone hosts can dial it up."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=20000)  # 20 GB
        gguf = _make_gguf(tmp_path / "model.gguf", size_mb=18000)

        # Default factor 1.1 → 18000 * 1.1 = 19800 + 1024 = 20824 → fails by 824
        default = RAMCheck(meminfo_path=meminfo, mmap_overhead_factor=1.1)
        result_default = await default.check(_make_mock_service(model_path=str(gguf)))
        # Aggressive factor 1.5 → 18000 * 1.5 = 27000 + 1024 → fails harder
        aggressive = RAMCheck(meminfo_path=meminfo, mmap_overhead_factor=1.5)
        result_aggressive = await aggressive.check(_make_mock_service(model_path=str(gguf)))

        # Both fail at 20 GB available
        assert result_default.passed is False
        assert result_aggressive.passed is False
        # Aggressive should report the higher overhead factor
        assert "1.5" in result_aggressive.message
        assert "1.1" in result_default.message

    @pytest.mark.asyncio
    async def test_failure_message_includes_file_size_and_deficit(self, tmp_path: Path) -> None:
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=5000)
        gguf = _make_gguf(tmp_path / "qwen.gguf", size_mb=18000)

        check = RAMCheck(meminfo_path=meminfo, mmap_overhead_factor=1.1)
        result = await check.check(_make_mock_service(model_path=str(gguf)))

        assert result.passed is False
        # Remediation must surface enough numbers for the operator to act
        assert result.remediation is not None
        text = result.remediation
        assert "5000" in text  # available
        assert "18000" in text or "18 GB" in text  # file size
        assert "qwen.gguf" in text or "model" in text.lower()


# ---------------------------------------------------------------------------
# gpumod-ki89: default constants guard + factor=1.0 behavior
# ---------------------------------------------------------------------------


class TestRAMCheckDefaultConstants:
    """Guard tests for the default constants after gpumod-ki89 recalibration.

    GGML_CUDA_NO_PINNED=1 (gpumod-56md) eliminated the cudaHostAlloc freeze
    class, making factor=1.0 safe. These tests catch accidental drift.
    """

    def test_default_overhead_factor_is_0_9(self) -> None:
        assert DEFAULT_MMAP_OVERHEAD_FACTOR == 0.9

    def test_default_min_free_mb_is_1024(self) -> None:
        assert DEFAULT_MIN_FREE_MB == 1024

    @pytest.mark.asyncio
    async def test_default_factor_passes_at_threshold(self, tmp_path: Path) -> None:
        """With factor=0.9, MemAvailable = int(model_size*0.9) + 1024 should pass."""
        meminfo = tmp_path / "meminfo"
        # 17365 * 0.9 = 15628.5 → int() = 15628, + 1024 = 16652
        _write_meminfo(meminfo, mem_available_mb=16653)
        gguf = _make_gguf(tmp_path / "model.gguf", size_mb=17365)

        check = RAMCheck(meminfo_path=meminfo)  # uses defaults
        result = await check.check(_make_mock_service(model_path=str(gguf)))

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_default_factor_fails_below_threshold(self, tmp_path: Path) -> None:
        """With factor=0.9, MemAvailable below threshold should fail."""
        meminfo = tmp_path / "meminfo"
        _write_meminfo(meminfo, mem_available_mb=16651)  # 1 MB below threshold
        gguf = _make_gguf(tmp_path / "model.gguf", size_mb=17365)

        check = RAMCheck(meminfo_path=meminfo)  # uses defaults
        result = await check.check(_make_mock_service(model_path=str(gguf)))

        assert result.passed is False
        assert result.severity == "error"
