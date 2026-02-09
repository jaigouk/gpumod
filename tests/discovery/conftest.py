"""Pytest fixtures for discovery module tests."""

from __future__ import annotations

from datetime import datetime, timezone
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
    with patch.object(
        __import__("gpumod.discovery.system_info", fromlist=["SystemInfoCollector"]).SystemInfoCollector,
        "_get_current_mode",
        new_callable=AsyncMock,
        return_value=None,
    ):
        with patch.object(
            __import__("gpumod.discovery.system_info", fromlist=["SystemInfoCollector"]).SystemInfoCollector,
            "_get_running_services",
            new_callable=AsyncMock,
            return_value=[],
        ):
            yield


@pytest.fixture
def mock_service_manager_no_mode(mock_vram_tracker: None) -> Generator[None, None, None]:
    """Mock ServiceManager with no active mode."""
    from gpumod.discovery.system_info import SystemInfoCollector

    with patch.object(
        SystemInfoCollector,
        "_get_current_mode",
        new_callable=AsyncMock,
        return_value=None,
    ):
        with patch.object(
            SystemInfoCollector,
            "_get_running_services",
            new_callable=AsyncMock,
            return_value=[],
        ):
            yield


@pytest.fixture
def mock_service_manager_with_services(mock_vram_tracker: None) -> Generator[None, None, None]:
    """Mock ServiceManager with running services."""
    from gpumod.discovery.system_info import SystemInfoCollector

    with patch.object(
        SystemInfoCollector,
        "_get_current_mode",
        new_callable=AsyncMock,
        return_value="code",
    ):
        with patch.object(
            SystemInfoCollector,
            "_get_running_services",
            new_callable=AsyncMock,
            return_value=["vllm-chat", "embedding-service"],
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
    yield


@pytest.fixture
def mock_proc_meminfo() -> Generator[None, None, None]:
    """Mock /proc/meminfo."""
    from pathlib import Path as RealPath
    from unittest.mock import mock_open

    meminfo = """MemTotal:       65536000 kB
MemFree:        20000000 kB
MemAvailable:   58000000 kB
SwapTotal:      16000000 kB
SwapFree:        8000000 kB
"""
    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.open", mock_open(read_data=meminfo)):
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
    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.open", mock_open(read_data=meminfo)):
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
                lastModified=datetime.now(tz=timezone.utc),
                tags=["gguf", "code"],
                cardData={"model_name": "Qwen3 Coder Next"},
            ),
            MagicMock(
                id="unsloth/Nemotron-3-Nano-GGUF",
                modelId="unsloth/Nemotron-3-Nano-GGUF",
                lastModified=datetime.now(tz=timezone.utc),
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
                lastModified=datetime.now(tz=timezone.utc),
                tags=["gguf"],
            ),
            MagicMock(
                id="unsloth/model-adapter",
                modelId="unsloth/model-adapter",
                lastModified=datetime.now(tz=timezone.utc),
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
                lastModified=datetime.now(tz=timezone.utc),
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
                lastModified=datetime.now(tz=timezone.utc),
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
    yield


@pytest.fixture
def mock_hf_api_with_private() -> Generator[None, None, None]:
    """Mock HF API with private repos."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        api.list_models.return_value = [
            MagicMock(
                id="unsloth/public-model",
                modelId="unsloth/public-model",
                lastModified=datetime.now(tz=timezone.utc),
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
                lastModified=datetime.now(tz=timezone.utc),
                tags=["gguf", "code"],
            ),
            MagicMock(
                id="unsloth/chat-model-gguf",
                modelId="unsloth/chat-model-gguf",
                lastModified=datetime.now(tz=timezone.utc),
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
        api.list_repo_files.return_value = [
            "model-Q4_K_M.gguf",
            "model-Q8_0.gguf",
            "README.md",
        ]
        mock.return_value = api
        with patch("huggingface_hub.get_hf_file_metadata") as mock_meta:
            mock_meta.return_value = MagicMock(size=20_000_000_000)
            yield


@pytest.fixture
def mock_hf_repo_files_with_sizes() -> Generator[None, None, None]:
    """Mock HF repo with file size metadata."""
    with patch("huggingface_hub.HfApi") as mock_api:
        with patch("huggingface_hub.get_hf_file_metadata") as mock_meta:
            api = MagicMock()
            api.list_repo_files.return_value = ["model-Q4_K_M.gguf"]
            mock_api.return_value = api
            mock_meta.return_value = MagicMock(size=20_000_000_000)
            yield


@pytest.fixture
def mock_hf_repo_split_files() -> Generator[None, None, None]:
    """Mock HF repo with split GGUF files."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        api.list_repo_files.return_value = [
            "model-00001-of-00003.gguf",
            "model-00002-of-00003.gguf",
            "model-00003-of-00003.gguf",
        ]
        mock.return_value = api
        with patch("huggingface_hub.get_hf_file_metadata") as mock_meta:
            mock_meta.return_value = MagicMock(size=10_000_000_000)
            yield


@pytest.fixture
def mock_hf_repo_no_gguf() -> Generator[None, None, None]:
    """Mock HF repo with no GGUF files."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        api.list_repo_files.return_value = [
            "adapter.safetensors",
            "config.json",
        ]
        mock.return_value = api
        yield


@pytest.fixture
def mock_hf_repo_multiple_quants() -> Generator[None, None, None]:
    """Mock HF repo with multiple quantizations."""
    with patch("huggingface_hub.HfApi") as mock_api:
        with patch("huggingface_hub.get_hf_file_metadata") as mock_meta:
            api = MagicMock()
            api.list_repo_files.return_value = [
                "model-Q2_K.gguf",
                "model-Q4_K_M.gguf",
                "model-Q8_0.gguf",
            ]
            mock_api.return_value = api

            def get_size(url: str) -> MagicMock:
                if "Q2_K" in url:
                    return MagicMock(size=10_000_000_000)
                if "Q4_K_M" in url:
                    return MagicMock(size=20_000_000_000)
                return MagicMock(size=40_000_000_000)

            mock_meta.side_effect = get_size
            yield


@pytest.fixture
def mock_hf_repo_not_found() -> Generator[None, None, None]:
    """Mock HF repo not found."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        # Use a generic exception since RepositoryNotFoundError requires complex args
        api.list_repo_files.side_effect = Exception("Repository not found")
        mock.return_value = api
        yield
