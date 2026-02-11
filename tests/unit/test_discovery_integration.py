"""Integration-style tests for discovery layer components.

These tests mock at the HTTP/API level to exercise more code paths
in the discovery module, improving overall coverage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# TestGGUFMetadataFetcher - GGUF file listing
# ---------------------------------------------------------------------------


class TestGGUFMetadataFetcherIntegration:
    """Integration tests for GGUFMetadataFetcher."""

    async def test_list_gguf_files_parses_filenames(self) -> None:
        """list_gguf_files correctly parses quantization from filenames."""
        from gpumod.discovery.gguf_metadata import GGUFMetadataFetcher

        fetcher = GGUFMetadataFetcher()

        # Mock HfApi via huggingface_hub module
        with patch("huggingface_hub.HfApi") as mock_api_cls:
            mock_api = MagicMock()

            # Create mock repo_info response
            mock_sibling1 = MagicMock()
            mock_sibling1.rfilename = "model-Q4_K_M.gguf"
            mock_sibling1.size = 8_000_000_000

            mock_sibling2 = MagicMock()
            mock_sibling2.rfilename = "model-Q8_0.gguf"
            mock_sibling2.size = 16_000_000_000

            mock_sibling3 = MagicMock()
            mock_sibling3.rfilename = "README.md"  # Should be filtered out
            mock_sibling3.size = 1000

            mock_repo_info = MagicMock()
            mock_repo_info.siblings = [mock_sibling1, mock_sibling2, mock_sibling3]

            mock_api.repo_info.return_value = mock_repo_info
            mock_api_cls.return_value = mock_api

            files = await fetcher.list_gguf_files("test/repo")

            assert len(files) == 2
            assert files[0].quant_type == "Q4_K_M"
            assert files[1].quant_type == "Q8_0"
            # Sorted by size (smallest first)
            assert files[0].size_bytes < files[1].size_bytes

    async def test_list_gguf_files_handles_split_models(self) -> None:
        """list_gguf_files correctly groups split model files."""
        from gpumod.discovery.gguf_metadata import GGUFMetadataFetcher

        fetcher = GGUFMetadataFetcher()

        with patch("huggingface_hub.HfApi") as mock_api_cls:
            mock_api = MagicMock()

            # Split model files
            siblings = []
            for i in range(1, 4):
                sibling = MagicMock()
                sibling.rfilename = f"model-Q8_0-{i:05d}-of-00003.gguf"
                sibling.size = 10_000_000_000
                siblings.append(sibling)

            mock_repo_info = MagicMock()
            mock_repo_info.siblings = siblings

            mock_api.repo_info.return_value = mock_repo_info
            mock_api_cls.return_value = mock_api

            files = await fetcher.list_gguf_files("test/split-model")

            # Should be grouped into one entry
            assert len(files) == 1
            assert files[0].is_split is True
            assert files[0].split_parts == 3
            assert files[0].size_bytes == 30_000_000_000  # Total size

    async def test_list_gguf_files_repo_not_found(self) -> None:
        """list_gguf_files raises RepoNotFoundError for missing repo."""
        import asyncio

        from huggingface_hub.utils import RepositoryNotFoundError as HfRepoNotFoundError

        from gpumod.discovery.gguf_metadata import GGUFMetadataFetcher, RepoNotFoundError

        fetcher = GGUFMetadataFetcher()

        # Patch asyncio.to_thread to raise the HuggingFace exception
        async def mock_to_thread(func, *args, **kwargs):
            # Create a proper mock response for HfRepoNotFoundError
            mock_response = MagicMock()
            mock_response.status_code = 404
            raise HfRepoNotFoundError("Not found", response=mock_response)

        with patch.object(asyncio, "to_thread", mock_to_thread), pytest.raises(RepoNotFoundError):
            await fetcher.list_gguf_files("fake/nonexistent")

    async def test_list_gguf_files_filters_imatrix(self) -> None:
        """list_gguf_files excludes imatrix calibration files."""
        from gpumod.discovery.gguf_metadata import GGUFMetadataFetcher

        fetcher = GGUFMetadataFetcher()

        with patch("huggingface_hub.HfApi") as mock_api_cls:
            mock_api = MagicMock()

            mock_sibling1 = MagicMock()
            mock_sibling1.rfilename = "model-Q4_K_M.gguf"
            mock_sibling1.size = 8_000_000_000

            mock_sibling2 = MagicMock()
            mock_sibling2.rfilename = "model-imatrix.gguf"  # Should be filtered
            mock_sibling2.size = 1_000_000

            mock_repo_info = MagicMock()
            mock_repo_info.siblings = [mock_sibling1, mock_sibling2]

            mock_api.repo_info.return_value = mock_repo_info
            mock_api_cls.return_value = mock_api

            files = await fetcher.list_gguf_files("test/repo")

            assert len(files) == 1
            assert "imatrix" not in files[0].filename.lower()

    async def test_list_gguf_files_empty_repo(self) -> None:
        """list_gguf_files returns empty list for repo with no GGUF files."""
        from gpumod.discovery.gguf_metadata import GGUFMetadataFetcher

        fetcher = GGUFMetadataFetcher()

        with patch("huggingface_hub.HfApi") as mock_api_cls:
            mock_api = MagicMock()

            mock_sibling = MagicMock()
            mock_sibling.rfilename = "model.safetensors"
            mock_sibling.size = 8_000_000_000

            mock_repo_info = MagicMock()
            mock_repo_info.siblings = [mock_sibling]

            mock_api.repo_info.return_value = mock_repo_info
            mock_api_cls.return_value = mock_api

            files = await fetcher.list_gguf_files("test/safetensors-only")

            assert files == []


class TestGGUFMetadataQuantParsing:
    """Tests for quantization type parsing."""

    def test_parse_quant_type_standard(self) -> None:
        """Parse standard llama.cpp quant types."""
        from gpumod.discovery.gguf_metadata import GGUFMetadataFetcher

        fetcher = GGUFMetadataFetcher()

        test_cases = [
            ("model-Q4_K_M.gguf", "Q4_K_M"),
            ("model-Q5_K_S.gguf", "Q5_K_S"),
            ("model-Q8_0.gguf", "Q8_0"),
            ("model-Q2_K.gguf", "Q2_K"),
            ("model-IQ4_NL.gguf", "IQ4_NL"),
            ("model-IQ2_XXS.gguf", "IQ2_XXS"),
        ]

        for filename, expected in test_cases:
            result = fetcher._parse_quant_type(filename)
            assert result == expected, f"Failed for {filename}"

    def test_parse_quant_type_unsloth_dynamic(self) -> None:
        """Parse Unsloth dynamic quant types."""
        from gpumod.discovery.gguf_metadata import GGUFMetadataFetcher

        fetcher = GGUFMetadataFetcher()

        test_cases = [
            ("model-UD-Q4_K_M.gguf", "UD-Q4_K_M"),
            ("model-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
        ]

        for filename, expected in test_cases:
            result = fetcher._parse_quant_type(filename)
            assert result == expected, f"Failed for {filename}"

    def test_parse_quant_type_none(self) -> None:
        """Return None for unrecognized filenames."""
        from gpumod.discovery.gguf_metadata import GGUFMetadataFetcher

        fetcher = GGUFMetadataFetcher()

        result = fetcher._parse_quant_type("model.gguf")
        assert result is None

    def test_estimate_vram(self) -> None:
        """Estimate VRAM from file size."""
        from gpumod.discovery.gguf_metadata import GGUFMetadataFetcher

        fetcher = GGUFMetadataFetcher()

        # 8GB file should estimate ~8.8GB VRAM (10% overhead)
        vram = fetcher._estimate_vram(8_000_000_000)
        assert 8000 < vram < 9000

        # Zero size should return 0
        assert fetcher._estimate_vram(0) == 0


# ---------------------------------------------------------------------------
# TestHuggingFaceSearcher - Model search
# ---------------------------------------------------------------------------


class TestHuggingFaceSearcherIntegration:
    """Integration tests for HuggingFaceSearcher."""

    async def test_search_with_driver_filter(self) -> None:
        """Search with driver filter returns correct format."""
        import asyncio

        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        searcher = HuggingFaceSearcher()

        # Create mock model
        mock_model = MagicMock()
        mock_model.id = "unsloth/Qwen3-GGUF"
        mock_model.tags = ["gguf", "llama-cpp"]
        mock_model.lastModified = datetime.now(tz=UTC)
        mock_model.cardData = None

        # Patch asyncio.to_thread to return mock models
        async def mock_to_thread(func, *args, **kwargs):
            return [mock_model]

        with patch.object(asyncio, "to_thread", mock_to_thread):
            results = await searcher.search(query="qwen", driver="llamacpp")

            assert len(results) == 1
            assert results[0].repo_id == "unsloth/Qwen3-GGUF"
            assert results[0].model_format == "gguf"

    async def test_search_caching(self) -> None:
        """Second search with same params uses cache."""
        import asyncio

        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        searcher = HuggingFaceSearcher(cache_ttl_seconds=3600)

        call_count = 0

        async def mock_to_thread(func, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return []

        with patch.object(asyncio, "to_thread", mock_to_thread):
            await searcher.search(query="test", driver="llamacpp")
            await searcher.search(query="test", driver="llamacpp")

            # Should only call API once due to caching
            assert call_count == 1

    async def test_search_force_refresh(self) -> None:
        """force_refresh bypasses cache."""
        import asyncio

        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        searcher = HuggingFaceSearcher(cache_ttl_seconds=3600)

        call_count = 0

        async def mock_to_thread(func, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return []

        with patch.object(asyncio, "to_thread", mock_to_thread):
            await searcher.search(query="test", driver="llamacpp")
            await searcher.search(query="test", driver="llamacpp", force_refresh=True)

            assert call_count == 2

    def test_clear_cache(self) -> None:
        """clear_cache removes all cached entries."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        searcher = HuggingFaceSearcher()
        searcher._cache["test_key"] = ([], 0.0)

        searcher.clear_cache()

        assert len(searcher._cache) == 0


# ---------------------------------------------------------------------------
# TestUnslothModelLister - Unsloth org model listing
# ---------------------------------------------------------------------------


class TestUnslothModelListerIntegration:
    """Integration tests for UnslothModelLister."""

    async def test_list_models_returns_models(self) -> None:
        """list_models returns UnslothModel instances."""
        from gpumod.discovery.unsloth_lister import UnslothModelLister

        lister = UnslothModelLister()

        with patch("huggingface_hub.HfApi") as mock_api_cls:
            mock_api = MagicMock()

            mock_model = MagicMock()
            mock_model.id = "unsloth/Qwen3-GGUF"
            mock_model.tags = ["gguf"]
            mock_model.last_modified = datetime.now(tz=UTC)
            mock_model.downloads = 1000
            mock_model.likes = 50

            mock_api.list_models.return_value = [mock_model]
            mock_api_cls.return_value = mock_api

            models = await lister.list_models()

            assert len(models) == 1
            assert models[0].repo_id == "unsloth/Qwen3-GGUF"

    async def test_list_models_with_search(self) -> None:
        """list_models filters by search term."""
        from gpumod.discovery.unsloth_lister import UnslothModelLister

        lister = UnslothModelLister()

        with patch("huggingface_hub.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = []
            mock_api_cls.return_value = mock_api

            await lister.list_models(search="qwen")

            mock_api.list_models.assert_called_once()

    async def test_list_models_caching(self) -> None:
        """list_models uses TTL caching."""
        from gpumod.discovery.unsloth_lister import UnslothModelLister

        lister = UnslothModelLister(cache_ttl_seconds=3600)

        with patch("huggingface_hub.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = []
            mock_api_cls.return_value = mock_api

            await lister.list_models()
            await lister.list_models()

            # Should cache second call
            assert mock_api.list_models.call_count == 1


# ---------------------------------------------------------------------------
# TestDetectModelFormat - Format detection helpers
# ---------------------------------------------------------------------------


class TestModelFormatDetection:
    """Tests for model format detection utilities."""

    def test_detect_model_format_gguf(self) -> None:
        """Detect GGUF format from tags."""
        from gpumod.discovery.hf_searcher import detect_model_format

        assert detect_model_format(["gguf", "llama-cpp"]) == "gguf"
        assert detect_model_format(["GGUF"]) == "gguf"

    def test_detect_model_format_safetensors(self) -> None:
        """Detect Safetensors format from tags."""
        from gpumod.discovery.hf_searcher import detect_model_format

        assert detect_model_format(["safetensors", "transformers"]) == "safetensors"

    def test_detect_model_format_unknown(self) -> None:
        """Return unknown for unrecognized tags."""
        from gpumod.discovery.hf_searcher import detect_model_format

        assert detect_model_format(["pytorch", "text-generation"]) == "unknown"
        assert detect_model_format([]) == "unknown"

    def test_get_driver_hint_gguf(self) -> None:
        """Get driver hint for GGUF format."""
        from gpumod.discovery.hf_searcher import get_driver_hint

        assert get_driver_hint("gguf") == "llamacpp"

    def test_get_driver_hint_safetensors(self) -> None:
        """Get driver hint for Safetensors format."""
        from gpumod.discovery.hf_searcher import get_driver_hint

        assert get_driver_hint("safetensors") == "vllm"

    def test_get_driver_hint_unknown(self) -> None:
        """Get None for unknown format."""
        from gpumod.discovery.hf_searcher import get_driver_hint

        assert get_driver_hint("unknown") is None


# ---------------------------------------------------------------------------
# TestLlamaCppOptions - llama.cpp flag recommendations
# ---------------------------------------------------------------------------


class TestLlamaCppOptionsIntegration:
    """Integration tests for LlamaCppOptions."""

    def test_get_recommended_config_basic(self) -> None:
        """get_recommended_config returns RecommendedConfig for basic model."""
        from gpumod.discovery.llamacpp_options import LlamaCppOptions

        options = LlamaCppOptions()

        config = options.get_recommended_config(
            model_size_b=7,  # 7B params
            available_vram_mb=24000,
        )

        assert config is not None
        assert config.n_gpu_layers != 0
        assert config.ctx_size > 0

    def test_get_recommended_config_moe(self) -> None:
        """get_recommended_config handles MoE models."""
        from gpumod.discovery.llamacpp_options import LlamaCppOptions

        options = LlamaCppOptions()

        config = options.get_recommended_config(
            model_size_b=47,  # 47B params (Mixtral-sized)
            available_vram_mb=24000,
            is_moe=True,
        )

        assert config is not None
        assert config.n_gpu_layers is not None

    def test_get_recommended_config_small_vram(self) -> None:
        """get_recommended_config recommends partial offload for small VRAM."""
        from gpumod.discovery.llamacpp_options import LlamaCppOptions

        options = LlamaCppOptions()

        config = options.get_recommended_config(
            model_size_b=70,  # Large model
            available_vram_mb=8000,  # Small VRAM
        )

        assert config is not None
        # Should recommend partial offload (not all layers)
        assert config.n_gpu_layers is not None

    def test_get_option(self) -> None:
        """get_option returns OptionSpec for known options."""
        from gpumod.discovery.llamacpp_options import LlamaCppOptions

        options = LlamaCppOptions()

        spec = options.get_option("n_gpu_layers")
        assert spec is not None
        assert spec.name == "n_gpu_layers"

    def test_get_all_options(self) -> None:
        """get_all_options returns dict of all OptionSpec."""
        from gpumod.discovery.llamacpp_options import LlamaCppOptions

        options = LlamaCppOptions()

        all_specs = options.get_all_options()
        assert len(all_specs) > 0
        # Returns dict[str, OptionSpec], check values have name attribute
        assert all(hasattr(s, "name") for s in all_specs.values())


# ---------------------------------------------------------------------------
# TestSystemInfoCollector - System detection
# ---------------------------------------------------------------------------


class TestSystemInfoCollectorIntegration:
    """Integration tests for SystemInfoCollector."""

    async def test_get_system_info_returns_system_info(self) -> None:
        """get_system_info() returns SystemInfo dataclass."""
        from gpumod.discovery.system_info import SystemInfoCollector
        from gpumod.services.vram import GPUInfo, VRAMTracker, VRAMUsage

        # Create mock VRAMTracker
        mock_tracker = MagicMock(spec=VRAMTracker)

        async def mock_get_gpu_info():
            return GPUInfo(name="NVIDIA RTX 4090", vram_total_mb=24576)

        async def mock_get_usage():
            return VRAMUsage(total_mb=24576, used_mb=1000, free_mb=23576)

        mock_tracker.get_gpu_info = mock_get_gpu_info
        mock_tracker.get_usage = mock_get_usage

        collector = SystemInfoCollector(vram_tracker=mock_tracker)

        # Also mock the mode/services DB calls
        with (
            patch.object(collector, "_get_current_mode", return_value=None),
            patch.object(collector, "_get_running_services", return_value=[]),
        ):
            info = await collector.get_system_info()

            assert info is not None
            assert info.gpu_name == "NVIDIA RTX 4090"
            assert info.gpu_total_mb == 24576

    async def test_get_system_info_handles_timeout(self) -> None:
        """get_system_info() raises NvidiaSmiUnavailableError on timeout."""
        import asyncio

        from gpumod.discovery.system_info import NvidiaSmiUnavailableError, SystemInfoCollector
        from gpumod.services.vram import VRAMTracker

        # Create mock VRAMTracker that times out
        mock_tracker = MagicMock(spec=VRAMTracker)

        async def slow_get_gpu_info():
            await asyncio.sleep(10)  # Longer than timeout

        mock_tracker.get_gpu_info = slow_get_gpu_info
        mock_tracker.get_usage = slow_get_gpu_info

        collector = SystemInfoCollector(vram_tracker=mock_tracker, nvidia_smi_timeout=0.01)

        with pytest.raises(NvidiaSmiUnavailableError):
            await collector.get_system_info()
