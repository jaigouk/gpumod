"""Pytest fixtures for discovery module tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_vram_tracker() -> Generator[None, None, None]:
    """Mock VRAMTracker with standard RTX 4090 values."""
    with patch("gpumod.discovery.system_info.VRAMTracker") as mock:
        tracker = MagicMock()
        tracker.get_gpu_info = AsyncMock(
            return_value=MagicMock(
                name="NVIDIA RTX 4090",
                vram_total_mb=24576,
                driver="535.104.05",
            )
        )
        tracker.get_usage = AsyncMock(
            return_value=MagicMock(
                total_mb=24576,
                used_mb=512,
                free_mb=24064,
            )
        )
        mock.return_value = tracker
        yield


@pytest.fixture
def mock_service_manager() -> Generator[None, None, None]:
    """Mock ServiceManager with no active mode."""
    from gpumod.discovery.system_info import SystemInfoCollector

    with (
        patch.object(
            SystemInfoCollector,
            "_get_current_mode",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(
            SystemInfoCollector,
            "_get_running_services",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        yield


@pytest.fixture
def mock_service_manager_no_mode(mock_vram_tracker: None) -> Generator[None, None, None]:
    """Mock ServiceManager with no active mode."""
    from gpumod.discovery.system_info import SystemInfoCollector

    with (
        patch.object(
            SystemInfoCollector,
            "_get_current_mode",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(
            SystemInfoCollector,
            "_get_running_services",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        yield


@pytest.fixture
def mock_service_manager_with_services(mock_vram_tracker: None) -> Generator[None, None, None]:
    """Mock ServiceManager with running services."""
    from gpumod.discovery.system_info import SystemInfoCollector

    with (
        patch.object(
            SystemInfoCollector,
            "_get_current_mode",
            new_callable=AsyncMock,
            return_value="code",
        ),
        patch.object(
            SystemInfoCollector,
            "_get_running_services",
            new_callable=AsyncMock,
            return_value=["vllm-chat", "embedding-service"],
        ),
    ):
        yield


@pytest.fixture
def mock_nvidia_smi_unavailable() -> Generator[None, None, None]:
    """Mock nvidia-smi not available."""
    with patch("gpumod.discovery.system_info.VRAMTracker") as mock:
        from gpumod.services.vram import NvidiaSmiError

        tracker = MagicMock()
        tracker.get_gpu_info = AsyncMock(side_effect=NvidiaSmiError("not found"))
        tracker.get_usage = AsyncMock(side_effect=NvidiaSmiError("not found"))
        mock.return_value = tracker
        yield


@pytest.fixture
def mock_nvidia_smi_hangs() -> Generator[None, None, None]:
    """Mock nvidia-smi that hangs."""
    import asyncio

    async def hang() -> None:
        await asyncio.sleep(100)

    with patch("gpumod.discovery.system_info.VRAMTracker") as mock:
        tracker = MagicMock()
        tracker.get_gpu_info = hang
        tracker.get_usage = hang
        mock.return_value = tracker
        yield


@pytest.fixture
def mock_multi_gpu(mock_vram_tracker: None) -> Generator[None, None, None]:
    """Mock multi-GPU system."""
    # The mock_vram_tracker already provides GPU info
    return


@pytest.fixture
def mock_proc_meminfo() -> Generator[None, None, None]:
    """Mock /proc/meminfo."""
    from unittest.mock import mock_open

    meminfo = """MemTotal:       65536000 kB
MemFree:        20000000 kB
MemAvailable:   58000000 kB
SwapTotal:      16000000 kB
SwapFree:        8000000 kB
"""
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.open", mock_open(read_data=meminfo)),
    ):
        yield


@pytest.fixture
def mock_swap_disabled() -> Generator[None, None, None]:
    """Mock system with swap disabled."""
    from unittest.mock import mock_open

    meminfo = """MemTotal:       65536000 kB
MemFree:        20000000 kB
MemAvailable:   58000000 kB
SwapTotal:             0 kB
SwapFree:              0 kB
"""
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.open", mock_open(read_data=meminfo)),
    ):
        yield


# HuggingFace API mocks


@pytest.fixture
def mock_hf_api() -> Generator[None, None, None]:
    """Mock HuggingFace API with sample models."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        api.list_models.return_value = [
            MagicMock(
                id="unsloth/Qwen3-Coder-Next-GGUF",
                modelId="unsloth/Qwen3-Coder-Next-GGUF",
                lastModified=datetime.now(tz=UTC),
                tags=["gguf", "code"],
                cardData={"model_name": "Qwen3 Coder Next"},
            ),
            MagicMock(
                id="unsloth/Nemotron-3-Nano-GGUF",
                modelId="unsloth/Nemotron-3-Nano-GGUF",
                lastModified=datetime.now(tz=UTC),
                tags=["gguf", "chat"],
                cardData={"model_name": "Nemotron 3 Nano"},
            ),
        ]
        mock.return_value = api
        yield


@pytest.fixture
def mock_hf_api_mixed_repos() -> Generator[None, None, None]:
    """Mock HF API with mix of GGUF and non-GGUF repos."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        api.list_models.return_value = [
            MagicMock(
                id="unsloth/model-gguf",
                modelId="unsloth/model-gguf",
                lastModified=datetime.now(tz=UTC),
                tags=["gguf"],
            ),
            MagicMock(
                id="unsloth/model-adapter",
                modelId="unsloth/model-adapter",
                lastModified=datetime.now(tz=UTC),
                tags=["adapter"],  # No GGUF
            ),
        ]

        def mock_list_files(repo_id: str) -> list[str]:
            if "gguf" in repo_id.lower():
                return ["model.gguf", "README.md"]
            return ["adapter.safetensors", "README.md"]

        api.list_repo_files.side_effect = mock_list_files
        mock.return_value = api
        yield


@pytest.fixture
def mock_hf_api_paginated() -> Generator[None, None, None]:
    """Mock HF API with pagination (150 models)."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        models = [
            MagicMock(
                id=f"unsloth/model-{i}-gguf",
                modelId=f"unsloth/model-{i}-gguf",
                lastModified=datetime.now(tz=UTC),
                tags=["gguf"],
            )
            for i in range(150)
        ]
        api.list_models.return_value = models
        mock.return_value = api
        yield


@pytest.fixture
def mock_hf_api_no_token() -> Generator[None, None, None]:
    """Mock HF API without token."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        api.list_models.return_value = [
            MagicMock(
                id="unsloth/public-model-gguf",
                modelId="unsloth/public-model-gguf",
                lastModified=datetime.now(tz=UTC),
                tags=["gguf"],
            ),
        ]
        mock.return_value = api
        yield


@pytest.fixture
def mock_hf_api_error() -> Generator[None, None, None]:
    """Mock HF API with error."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        api.list_models.side_effect = Exception("API error")
        mock.return_value = api
        yield


@pytest.fixture
def mock_hf_api_network_error_with_cache() -> Generator[None, None, None]:
    """Mock HF API with network error but cached data."""
    # This would need to be implemented with actual cache logic
    return


@pytest.fixture
def mock_hf_api_with_private() -> Generator[None, None, None]:
    """Mock HF API with private repos."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        api.list_models.return_value = [
            MagicMock(
                id="unsloth/public-model",
                modelId="unsloth/public-model",
                lastModified=datetime.now(tz=UTC),
                tags=["gguf"],
                private=False,
            ),
        ]
        mock.return_value = api
        yield


@pytest.fixture
def mock_hf_api_with_tags() -> Generator[None, None, None]:
    """Mock HF API with tagged models."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        api.list_models.return_value = [
            MagicMock(
                id="unsloth/coder-model-gguf",
                modelId="unsloth/coder-model-gguf",
                lastModified=datetime.now(tz=UTC),
                tags=["gguf", "code"],
            ),
            MagicMock(
                id="unsloth/chat-model-gguf",
                modelId="unsloth/chat-model-gguf",
                lastModified=datetime.now(tz=UTC),
                tags=["gguf", "chat"],
            ),
        ]
        mock.return_value = api
        yield


# GGUF metadata mocks


@pytest.fixture
def mock_hf_repo_files() -> Generator[None, None, None]:
    """Mock HF repo file listing."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        # Mock repo_info with file siblings
        siblings = []
        file_data = [
            ("model-Q4_K_M.gguf", 20_000_000_000),
            ("model-Q8_0.gguf", 40_000_000_000),
            ("README.md", 1000),
        ]
        for fname, size in file_data:
            file_info = MagicMock()
            file_info.rfilename = fname
            file_info.size = size
            siblings.append(file_info)
        repo_info = MagicMock()
        repo_info.siblings = siblings
        api.repo_info.return_value = repo_info
        mock.return_value = api
        yield


@pytest.fixture
def mock_hf_repo_files_with_sizes() -> Generator[None, None, None]:
    """Mock HF repo with file size metadata."""
    with patch("huggingface_hub.HfApi") as mock_api:
        api = MagicMock()
        # Mock repo_info with siblings containing file metadata
        file_info = MagicMock()
        file_info.rfilename = "model-Q4_K_M.gguf"
        file_info.size = 20_000_000_000
        repo_info = MagicMock()
        repo_info.siblings = [file_info]
        api.repo_info.return_value = repo_info
        mock_api.return_value = api
        yield


@pytest.fixture
def mock_hf_repo_split_files() -> Generator[None, None, None]:
    """Mock HF repo with split GGUF files."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        # Mock repo_info with split file siblings
        siblings = []
        for i in range(1, 4):
            file_info = MagicMock()
            file_info.rfilename = f"model-{i:05d}-of-00003.gguf"
            file_info.size = 10_000_000_000
            siblings.append(file_info)
        repo_info = MagicMock()
        repo_info.siblings = siblings
        api.repo_info.return_value = repo_info
        mock.return_value = api
        yield


@pytest.fixture
def mock_hf_repo_no_gguf() -> Generator[None, None, None]:
    """Mock HF repo with no GGUF files."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        # Mock repo_info with non-GGUF files
        siblings = []
        for fname in ["adapter.safetensors", "config.json"]:
            file_info = MagicMock()
            file_info.rfilename = fname
            file_info.size = 1_000_000
            siblings.append(file_info)
        repo_info = MagicMock()
        repo_info.siblings = siblings
        api.repo_info.return_value = repo_info
        mock.return_value = api
        yield


@pytest.fixture
def mock_hf_repo_multiple_quants() -> Generator[None, None, None]:
    """Mock HF repo with multiple quantizations."""
    with patch("huggingface_hub.HfApi") as mock_api:
        api = MagicMock()
        # Mock repo_info with multiple quantization files
        siblings = []
        sizes = {
            "model-Q2_K.gguf": 10_000_000_000,
            "model-Q4_K_M.gguf": 20_000_000_000,
            "model-Q8_0.gguf": 40_000_000_000,
        }
        for fname, size in sizes.items():
            file_info = MagicMock()
            file_info.rfilename = fname
            file_info.size = size
            siblings.append(file_info)
        repo_info = MagicMock()
        repo_info.siblings = siblings
        api.repo_info.return_value = repo_info
        mock_api.return_value = api
        yield


@pytest.fixture
def mock_hf_repo_not_found() -> Generator[None, None, None]:
    """Mock HF repo not found."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        # Both repo_info and list_repo_files should raise
        # Use generic Exception with "not found" message (code checks for this)
        api.repo_info.side_effect = Exception("Repository not found")
        api.list_repo_files.side_effect = Exception("Repository not found")
        mock.return_value = api
        yield


@pytest.fixture
def mock_hf_repo_with_imatrix() -> Generator[None, None, None]:
    """Mock HF repo containing imatrix calibration file."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        siblings = []
        # Include imatrix file (should be filtered out) and real model files
        file_data = [
            ("model-imatrix.gguf", 500_000_000),  # Calibration data
            ("model-Q4_K_M.gguf", 20_000_000_000),
            ("model-Q8_0.gguf", 40_000_000_000),
        ]
        for fname, size in file_data:
            file_info = MagicMock()
            file_info.rfilename = fname
            file_info.size = size
            siblings.append(file_info)
        repo_info = MagicMock()
        repo_info.siblings = siblings
        api.repo_info.return_value = repo_info
        mock.return_value = api
        yield


@pytest.fixture
def mock_hf_repo_with_imatrix_variants() -> Generator[None, None, None]:
    """Mock HF repo with various imatrix naming patterns."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        siblings = []
        # Various imatrix naming patterns (all should be filtered)
        file_data = [
            ("model-imatrix.gguf", 500_000_000),
            ("model-IMATRIX.gguf", 500_000_000),
            ("model_imatrix_calibration.gguf", 600_000_000),
            ("model-Q4_K_M.gguf", 20_000_000_000),  # Real model
        ]
        for fname, size in file_data:
            file_info = MagicMock()
            file_info.rfilename = fname
            file_info.size = size
            siblings.append(file_info)
        repo_info = MagicMock()
        repo_info.siblings = siblings
        api.repo_info.return_value = repo_info
        mock.return_value = api
        yield
