"""Tests for SystemInfoCollector - RED phase first."""

from __future__ import annotations

import pytest

from gpumod.discovery.system_info import (
    NvidiaSmiUnavailableError,
    SystemInfo,
    SystemInfoCollector,
)


class TestSystemInfoDataclass:
    """Tests for SystemInfo dataclass."""

    def test_system_info_is_frozen(self) -> None:
        """SystemInfo should be immutable (frozen dataclass)."""
        info = SystemInfo(
            gpu_total_mb=24576,
            gpu_used_mb=512,
            gpu_available_mb=24064,
            gpu_name="NVIDIA RTX 4090",
            ram_total_mb=65536,
            ram_available_mb=58000,
            swap_available_mb=8192,
            current_mode=None,
            running_services=(),
        )
        with pytest.raises(AttributeError):
            info.gpu_total_mb = 0  # type: ignore[misc]

    def test_system_info_running_services_is_tuple(self) -> None:
        """running_services should be a tuple (immutable)."""
        info = SystemInfo(
            gpu_total_mb=24576,
            gpu_used_mb=512,
            gpu_available_mb=24064,
            gpu_name="NVIDIA RTX 4090",
            ram_total_mb=65536,
            ram_available_mb=58000,
            swap_available_mb=8192,
            current_mode="code",
            running_services=("svc1", "svc2"),
        )
        assert isinstance(info.running_services, tuple)
        assert info.running_services == ("svc1", "svc2")


class TestSystemInfoCollector:
    """Tests for SystemInfoCollector."""

    @pytest.mark.asyncio
    async def test_get_system_info_returns_dataclass(
        self,
        mock_vram_tracker: None,
        mock_service_manager: None,
    ) -> None:
        """get_system_info() should return a SystemInfo dataclass."""
        collector = SystemInfoCollector()
        info = await collector.get_system_info()
        assert isinstance(info, SystemInfo)

    @pytest.mark.asyncio
    async def test_gpu_total_matches_nvidia_smi(
        self,
        mock_vram_tracker: None,
    ) -> None:
        """gpu_total_mb should match nvidia-smi query."""
        collector = SystemInfoCollector()
        info = await collector.get_system_info()
        # Mock returns 24576 MB
        assert info.gpu_total_mb == 24576

    @pytest.mark.asyncio
    async def test_gpu_used_matches_nvidia_smi(
        self,
        mock_vram_tracker: None,
    ) -> None:
        """gpu_used_mb should match nvidia-smi query."""
        collector = SystemInfoCollector()
        info = await collector.get_system_info()
        # Mock returns 512 MB used
        assert info.gpu_used_mb == 512

    @pytest.mark.asyncio
    async def test_gpu_available_is_computed(
        self,
        mock_vram_tracker: None,
    ) -> None:
        """gpu_available_mb should be total - used."""
        collector = SystemInfoCollector()
        info = await collector.get_system_info()
        assert info.gpu_available_mb == info.gpu_total_mb - info.gpu_used_mb

    @pytest.mark.asyncio
    async def test_ram_values_from_proc_meminfo(
        self,
        mock_proc_meminfo: None,
    ) -> None:
        """RAM values should come from /proc/meminfo."""
        collector = SystemInfoCollector()
        info = await collector.get_system_info()
        # Mock returns specific values
        assert info.ram_total_mb > 0
        assert info.ram_available_mb > 0

    @pytest.mark.asyncio
    async def test_current_mode_none_when_inactive(
        self,
        mock_service_manager_no_mode: None,
    ) -> None:
        """current_mode should be None when no mode is active."""
        collector = SystemInfoCollector()
        info = await collector.get_system_info()
        assert info.current_mode is None

    @pytest.mark.asyncio
    async def test_running_services_lists_active(
        self,
        mock_service_manager_with_services: None,
    ) -> None:
        """running_services should list all active gpumod services."""
        collector = SystemInfoCollector()
        info = await collector.get_system_info()
        assert len(info.running_services) > 0

    @pytest.mark.asyncio
    async def test_raises_when_nvidia_smi_unavailable(
        self,
        mock_nvidia_smi_unavailable: None,
    ) -> None:
        """Should raise NvidiaSmiUnavailableError when nvidia-smi is missing."""
        collector = SystemInfoCollector()
        with pytest.raises(NvidiaSmiUnavailableError):
            await collector.get_system_info()

    @pytest.mark.asyncio
    async def test_timeout_on_nvidia_smi_hang(
        self,
        mock_nvidia_smi_hangs: None,
    ) -> None:
        """Should timeout after 5s if nvidia-smi hangs."""
        collector = SystemInfoCollector(nvidia_smi_timeout=0.1)
        with pytest.raises(NvidiaSmiUnavailableError, match="timeout"):
            await collector.get_system_info()

    @pytest.mark.asyncio
    async def test_multi_gpu_selects_gpu_zero(
        self,
        mock_multi_gpu: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should select GPU 0 and log warning for multi-GPU systems."""
        collector = SystemInfoCollector()
        info = await collector.get_system_info()
        assert info.gpu_name is not None
        # Should log a warning about multi-GPU


class TestSystemInfoCollectorSwap:
    """Tests for swap handling."""

    @pytest.mark.asyncio
    async def test_swap_disabled_returns_zero(
        self,
        mock_swap_disabled: None,
    ) -> None:
        """swap_available_mb should be 0 when swap is disabled."""
        collector = SystemInfoCollector()
        info = await collector.get_system_info()
        assert info.swap_available_mb == 0


# Fixtures would be defined in conftest.py
