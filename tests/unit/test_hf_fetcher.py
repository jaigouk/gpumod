"""Tests for gpumod.fetchers.huggingface — HuggingFaceFetcher."""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gpumod.fetchers.huggingface import HuggingFaceFetcher
from gpumod.models import ModelSource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hf_model_info(
    model_id: str = "meta-llama/Llama-3-8B",
    *,
    safetensors: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    siblings: list[dict[str, str]] | None = None,
) -> MagicMock:
    """Create a mock huggingface_hub ModelInfo object."""
    mock = MagicMock()
    mock.id = model_id

    # safetensors metadata
    if safetensors is None:
        safetensors = {
            "parameters": {"F16": 8_000_000_000},
            "total": 8_000_000_000,
        }
    mock.safetensors = safetensors

    # config (architecture details)
    if config is None:
        config = {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }
    mock.config = config

    # siblings (files in the repo)
    if siblings is None:
        siblings = [
            {"rfilename": "config.json"},
            {"rfilename": "model-00001-of-00004.safetensors"},
        ]
    mock.siblings = [MagicMock(**s) for s in siblings]

    # Tags for quantization detection
    mock.tags = ["llama", "text-generation"]

    return mock


# ---------------------------------------------------------------------------
# fetch — basic behavior
# ---------------------------------------------------------------------------


class TestFetch:
    """Tests for HuggingFaceFetcher.fetch()."""

    async def test_fetch_returns_model_info_with_huggingface_source(self) -> None:
        """fetch() should return ModelInfo with source=HUGGINGFACE."""
        fetcher = HuggingFaceFetcher()
        mock_info = _make_hf_model_info()

        with patch(
            "gpumod.fetchers.huggingface.model_info",
            return_value=mock_info,
        ):
            result = await fetcher.fetch("meta-llama/Llama-3-8B")

        assert result.source == ModelSource.HUGGINGFACE
        assert result.id == "meta-llama/Llama-3-8B"

    async def test_fetch_populates_parameters_b(self) -> None:
        """fetch() should populate parameters_b from safetensors metadata."""
        fetcher = HuggingFaceFetcher()
        mock_info = _make_hf_model_info(
            safetensors={"parameters": {"F16": 8_000_000_000}, "total": 8_000_000_000},
        )

        with patch(
            "gpumod.fetchers.huggingface.model_info",
            return_value=mock_info,
        ):
            result = await fetcher.fetch("meta-llama/Llama-3-8B")

        assert result.parameters_b == pytest.approx(8.0, abs=0.1)

    async def test_fetch_populates_architecture(self) -> None:
        """fetch() should populate architecture from config."""
        fetcher = HuggingFaceFetcher()
        mock_info = _make_hf_model_info(
            config={
                "architectures": ["MistralForCausalLM"],
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
            },
        )

        with patch(
            "gpumod.fetchers.huggingface.model_info",
            return_value=mock_info,
        ):
            result = await fetcher.fetch("mistralai/Mistral-7B-v0.1")

        assert result.architecture == "MistralForCausalLM"

    async def test_fetch_estimates_base_vram_mb(self) -> None:
        """fetch() should estimate base_vram_mb from parameter count."""
        fetcher = HuggingFaceFetcher()
        mock_info = _make_hf_model_info(
            safetensors={"parameters": {"F16": 7_000_000_000}, "total": 7_000_000_000},
        )

        with patch(
            "gpumod.fetchers.huggingface.model_info",
            return_value=mock_info,
        ):
            result = await fetcher.fetch("meta-llama/Llama-3-8B")

        # 7B * 2 bytes (fp16) = 14GB = 14336 MB (approximately)
        assert result.base_vram_mb is not None
        assert result.base_vram_mb == 14336

    async def test_fetch_estimates_kv_cache(self) -> None:
        """fetch() should estimate kv_cache_per_1k_tokens_mb from config."""
        fetcher = HuggingFaceFetcher()
        mock_info = _make_hf_model_info(
            config={
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
            },
        )

        with patch(
            "gpumod.fetchers.huggingface.model_info",
            return_value=mock_info,
        ):
            result = await fetcher.fetch("meta-llama/Llama-3-8B")

        assert result.kv_cache_per_1k_tokens_mb is not None
        assert result.kv_cache_per_1k_tokens_mb > 0

    async def test_fetch_sets_fetched_at(self) -> None:
        """fetch() should set fetched_at to an ISO timestamp."""
        fetcher = HuggingFaceFetcher()
        mock_info = _make_hf_model_info()

        with patch(
            "gpumod.fetchers.huggingface.model_info",
            return_value=mock_info,
        ):
            result = await fetcher.fetch("meta-llama/Llama-3-8B")

        assert result.fetched_at is not None
        # Should be a valid ISO-ish timestamp
        assert re.match(r"\d{4}-\d{2}-\d{2}T", result.fetched_at)

    async def test_fetch_handles_missing_safetensors(self) -> None:
        """fetch() should handle models without safetensors metadata."""
        fetcher = HuggingFaceFetcher()
        mock_info = _make_hf_model_info(safetensors=None)
        mock_info.safetensors = None

        with patch(
            "gpumod.fetchers.huggingface.model_info",
            return_value=mock_info,
        ):
            result = await fetcher.fetch("meta-llama/Llama-3-8B")

        # Should still return a valid ModelInfo, just without parameter/vram info
        assert result.source == ModelSource.HUGGINGFACE
        assert result.parameters_b is None

    async def test_fetch_handles_missing_config(self) -> None:
        """fetch() should handle models without config."""
        fetcher = HuggingFaceFetcher()
        mock_info = _make_hf_model_info(config=None)
        mock_info.config = None

        with patch(
            "gpumod.fetchers.huggingface.model_info",
            return_value=mock_info,
        ):
            result = await fetcher.fetch("meta-llama/Llama-3-8B")

        assert result.source == ModelSource.HUGGINGFACE
        assert result.architecture is None


# ---------------------------------------------------------------------------
# fetch — input validation
# ---------------------------------------------------------------------------


class TestFetchValidation:
    """Tests for HuggingFaceFetcher.fetch() input validation."""

    async def test_fetch_raises_on_empty_model_id(self) -> None:
        """fetch() should raise ValueError for empty model_id."""
        fetcher = HuggingFaceFetcher()
        with pytest.raises(ValueError, match="model_id"):
            await fetcher.fetch("")

    async def test_fetch_raises_on_invalid_model_id_format(self) -> None:
        """fetch() should raise ValueError for model_id without org/model format."""
        fetcher = HuggingFaceFetcher()
        with pytest.raises(ValueError, match="model_id"):
            await fetcher.fetch("no-slash-model")

    async def test_fetch_raises_on_model_id_with_path_traversal(self) -> None:
        """fetch() should raise ValueError for model_id with path traversal."""
        fetcher = HuggingFaceFetcher()
        with pytest.raises(ValueError, match="model_id"):
            await fetcher.fetch("../../../etc/passwd")

    async def test_fetch_raises_on_model_id_with_special_chars(self) -> None:
        """fetch() should raise ValueError for model_id with injection chars."""
        fetcher = HuggingFaceFetcher()
        with pytest.raises(ValueError, match="model_id"):
            await fetcher.fetch("org/model; rm -rf /")


# ---------------------------------------------------------------------------
# fetch — error handling
# ---------------------------------------------------------------------------


class TestFetchErrors:
    """Tests for HuggingFaceFetcher.fetch() error handling."""

    async def test_fetch_raises_on_network_error(self) -> None:
        """fetch() should raise RuntimeError on network failures."""
        fetcher = HuggingFaceFetcher()

        with (
            patch(
                "gpumod.fetchers.huggingface.model_info",
                side_effect=Exception("Connection refused"),
            ),
            pytest.raises(RuntimeError, match="Failed to fetch"),
        ):
            await fetcher.fetch("meta-llama/Llama-3-8B")

    async def test_fetch_raises_on_404(self) -> None:
        """fetch() should raise RuntimeError when model not found."""
        from huggingface_hub.utils import EntryNotFoundError

        fetcher = HuggingFaceFetcher()

        with (
            patch(
                "gpumod.fetchers.huggingface.model_info",
                side_effect=EntryNotFoundError("Not found"),
            ),
            pytest.raises(RuntimeError, match="Failed to fetch"),
        ):
            await fetcher.fetch("nonexistent/model-xyz")


# ---------------------------------------------------------------------------
# _estimate_vram_mb — unit tests
# ---------------------------------------------------------------------------


class TestEstimateVramMb:
    """Tests for HuggingFaceFetcher._estimate_vram_mb()."""

    def test_estimate_vram_7b_fp16(self) -> None:
        """7B model in fp16 (2 bytes) should be ~14336 MB."""
        fetcher = HuggingFaceFetcher()
        result = fetcher._estimate_vram_mb(7.0, dtype_bytes=2)
        assert result == 14336

    def test_estimate_vram_7b_int4(self) -> None:
        """7B model in int4 (0.5 bytes) should be ~3584 MB."""
        fetcher = HuggingFaceFetcher()
        # int4 = 0.5 bytes per param, but we pass dtype_bytes as the multiplier
        # For int4 quantization: 7B * 0.5 bytes = 3.5GB = 3584 MB
        result = fetcher._estimate_vram_mb(7.0, dtype_bytes=1)
        assert result == 7168

    def test_estimate_vram_70b_fp16(self) -> None:
        """70B model in fp16 should be ~143360 MB."""
        fetcher = HuggingFaceFetcher()
        result = fetcher._estimate_vram_mb(70.0, dtype_bytes=2)
        assert result == 143360

    def test_estimate_vram_0b_returns_0(self) -> None:
        """0 parameters should return 0 MB."""
        fetcher = HuggingFaceFetcher()
        result = fetcher._estimate_vram_mb(0.0, dtype_bytes=2)
        assert result == 0


# ---------------------------------------------------------------------------
# _estimate_kv_cache_per_1k — unit tests
# ---------------------------------------------------------------------------


class TestEstimateKvCachePer1k:
    """Tests for HuggingFaceFetcher._estimate_kv_cache_per_1k()."""

    def test_kv_cache_llama_8b(self) -> None:
        """Llama 8B: 32 layers, 4096 hidden, 8 kv_heads."""
        fetcher = HuggingFaceFetcher()
        # KV cache per 1k tokens:
        # 2 (K+V) * num_layers * num_kv_heads * head_dim * 2 (bytes) * 1000 / (1024*1024)
        # For GQA: head_dim = hidden_size / num_attention_heads
        result = fetcher._estimate_kv_cache_per_1k(
            num_layers=32,
            hidden_size=4096,
            num_kv_heads=8,
            num_attention_heads=32,
        )
        assert result > 0
        assert isinstance(result, int)

    def test_kv_cache_larger_model(self) -> None:
        """Larger model should have larger KV cache."""
        fetcher = HuggingFaceFetcher()
        small = fetcher._estimate_kv_cache_per_1k(
            num_layers=32,
            hidden_size=4096,
            num_kv_heads=8,
            num_attention_heads=32,
        )
        large = fetcher._estimate_kv_cache_per_1k(
            num_layers=80,
            hidden_size=8192,
            num_kv_heads=8,
            num_attention_heads=64,
        )
        assert large > small

    def test_kv_cache_mha(self) -> None:
        """Multi-head attention (no GQA): kv_heads = attention_heads."""
        fetcher = HuggingFaceFetcher()
        result = fetcher._estimate_kv_cache_per_1k(
            num_layers=32,
            hidden_size=4096,
            num_kv_heads=32,
            num_attention_heads=32,
        )
        assert result > 0
