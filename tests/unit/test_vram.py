"""Tests for the VRAMTracker and NvidiaSmiError."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpumod.models import DriverType, GPUInfo, Service, SleepMode, VRAMUsage
from gpumod.services.vram import NvidiaSmiError, VRAMTracker

# ---------------------------------------------------------------------------
# Sample nvidia-smi outputs
# ---------------------------------------------------------------------------
GPU_INFO_CSV = "NVIDIA GeForce RTX 4090, 24564, 560.35.03\n"
USAGE_CSV = "21700, 2864\n"
PROCESS_XML = """\
<?xml version="1.0" ?>
<nvidia_smi_log>
  <gpu>
    <processes>
      <process_info>
        <pid>12345</pid>
        <used_memory>2500 MiB</used_memory>
      </process_info>
      <process_info>
        <pid>67890</pid>
        <used_memory>19100 MiB</used_memory>
      </process_info>
    </processes>
  </gpu>
</nvidia_smi_log>
"""
NO_PROCESSES_XML = """\
<?xml version="1.0" ?>
<nvidia_smi_log>
  <gpu>
    <processes>
      No running processes found
    </processes>
  </gpu>
</nvidia_smi_log>
"""


def _make_fake_process(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Build a fake asyncio.subprocess.Process."""
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout.encode(), stderr.encode()))
    proc.returncode = returncode
    return proc


def _make_service() -> Service:
    """Create a minimal service for estimate_service_vram tests."""
    return Service(
        id="qwen3-asr",
        name="Qwen3 ASR",
        driver=DriverType.FASTAPI,
        port=8300,
        vram_mb=3000,
        sleep_mode=SleepMode.NONE,
    )


class TestGetGpuInfo:
    """get_gpu_info() parses nvidia-smi CSV output into GPUInfo."""

    async def test_parses_csv_into_gpu_info(self) -> None:
        tracker = VRAMTracker()
        proc = _make_fake_process(stdout=GPU_INFO_CSV)
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc):
            info = await tracker.get_gpu_info()
        assert isinstance(info, GPUInfo)
        assert info.name == "NVIDIA GeForce RTX 4090"
        assert info.vram_total_mb == 24564
        assert info.driver == "560.35.03"


class TestGetUsage:
    """get_usage() parses nvidia-smi CSV output into VRAMUsage."""

    async def test_parses_csv_into_vram_usage(self) -> None:
        tracker = VRAMTracker()
        proc = _make_fake_process(stdout=USAGE_CSV)
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc):
            usage = await tracker.get_usage()
        assert isinstance(usage, VRAMUsage)
        assert usage.used_mb == 21700
        assert usage.free_mb == 2864


class TestGetProcessVram:
    """get_process_vram() parses nvidia-smi XML for per-process memory."""

    async def test_parses_xml_into_process_dict(self) -> None:
        tracker = VRAMTracker()
        proc = _make_fake_process(stdout=PROCESS_XML)
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc):
            result = await tracker.get_process_vram()
        assert result == {12345: 2500, 67890: 19100}

    async def test_no_processes_returns_empty_dict(self) -> None:
        no_proc_xml = """\
<?xml version="1.0" ?>
<nvidia_smi_log>
  <gpu>
    <processes>
    </processes>
  </gpu>
</nvidia_smi_log>
"""
        tracker = VRAMTracker()
        proc = _make_fake_process(stdout=no_proc_xml)
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc):
            result = await tracker.get_process_vram()
        assert result == {}

    async def test_no_running_processes_text_returns_empty_dict(self) -> None:
        tracker = VRAMTracker()
        proc = _make_fake_process(stdout=NO_PROCESSES_XML)
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc):
            result = await tracker.get_process_vram()
        assert result == {}


class TestRunNvidiaSmi:
    """_run_nvidia_smi raises NvidiaSmiError on failures."""

    async def test_raises_on_nonzero_exit_code(self) -> None:
        tracker = VRAMTracker()
        proc = _make_fake_process(stdout="", stderr="NVIDIA-SMI has failed", returncode=1)
        with (
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc),
            pytest.raises(NvidiaSmiError),
        ):
            await tracker._run_nvidia_smi(["--query-gpu=name"])

    async def test_raises_when_nvidia_smi_not_found(self) -> None:
        tracker = VRAMTracker()
        with (
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                side_effect=FileNotFoundError("nvidia-smi not found"),
            ),
            pytest.raises(NvidiaSmiError),
        ):
            await tracker._run_nvidia_smi(["--query-gpu=name"])

    async def test_error_includes_stderr(self) -> None:
        tracker = VRAMTracker()
        proc = _make_fake_process(stdout="", stderr="GPU access blocked", returncode=1)
        with (
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc),
            pytest.raises(NvidiaSmiError, match="GPU access blocked"),
        ):
            await tracker._run_nvidia_smi(["--query-gpu=name"])


class TestEstimateServiceVram:
    """estimate_service_vram simply returns service.vram_mb."""

    async def test_returns_service_vram_mb(self) -> None:
        tracker = VRAMTracker()
        svc = _make_service()
        result = await tracker.estimate_service_vram(svc)
        assert result == 3000


class TestWaitForVramRelease:
    """wait_for_vram_release polls GPU memory until enough is free."""

    async def test_returns_immediately_when_vram_available(self) -> None:
        """When free VRAM >= required, returns True immediately."""
        tracker = VRAMTracker()
        # Mock pynvml to return 20000 MB free
        with patch.object(tracker, "_get_free_vram_mb", return_value=20000):
            result = await tracker.wait_for_vram_release(
                required_mb=10000,
                timeout_s=5.0,
                poll_interval_s=0.1,
            )
        assert result is True

    async def test_polls_until_vram_released(self) -> None:
        """Polls multiple times until VRAM is released."""
        tracker = VRAMTracker()
        # Mock: first call returns low VRAM, second returns enough
        free_values = [5000, 5000, 20000]  # 3rd call has enough
        call_count = 0

        async def mock_get_free(device_id: int = 0) -> int:
            nonlocal call_count
            val = free_values[min(call_count, len(free_values) - 1)]
            call_count += 1
            return val

        with patch.object(tracker, "_get_free_vram_mb", side_effect=mock_get_free):
            result = await tracker.wait_for_vram_release(
                required_mb=10000,
                timeout_s=5.0,
                poll_interval_s=0.01,
            )
        assert result is True
        assert call_count == 3

    async def test_timeout_returns_false(self) -> None:
        """When timeout expires without enough VRAM, returns False."""
        tracker = VRAMTracker()
        # Always return insufficient VRAM
        with patch.object(tracker, "_get_free_vram_mb", return_value=1000):
            result = await tracker.wait_for_vram_release(
                required_mb=10000,
                timeout_s=0.1,  # Short timeout
                poll_interval_s=0.02,
            )
        assert result is False

    async def test_includes_safety_margin(self) -> None:
        """Free VRAM must exceed required + safety margin."""
        tracker = VRAMTracker()
        # Mock: exactly required amount (10000), but not enough with margin
        with patch.object(tracker, "_get_free_vram_mb", return_value=10000):
            result = await tracker.wait_for_vram_release(
                required_mb=10000,
                timeout_s=0.1,
                poll_interval_s=0.02,
                safety_margin_mb=512,
            )
        # Should fail because 10000 < 10000 + 512
        assert result is False

    async def test_pynvml_used_when_available(self) -> None:
        """Uses pynvml when available for faster polling."""
        tracker = VRAMTracker()
        mock_nvml = MagicMock()
        mock_memory = MagicMock()
        mock_memory.free = 20 * 1024 * 1024 * 1024  # 20 GB in bytes

        with (
            patch.object(tracker, "_pynvml_available", True),
            patch("gpumod.services.vram.pynvml", mock_nvml),
        ):
            mock_nvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
            mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory
            free_mb = await tracker._get_free_vram_mb()

        assert free_mb == 20 * 1024  # 20 GB in MB

    async def test_fallback_to_nvidia_smi(self) -> None:
        """Falls back to nvidia-smi when pynvml unavailable."""
        tracker = VRAMTracker()
        tracker._pynvml_available = False

        proc = _make_fake_process(stdout=USAGE_CSV)  # "21700, 2864\n"
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc):
            free_mb = await tracker._get_free_vram_mb()

        assert free_mb == 2864
