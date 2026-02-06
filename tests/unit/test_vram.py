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
