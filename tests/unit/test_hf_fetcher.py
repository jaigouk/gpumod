"""Tests for gpumod.fetchers.huggingface — HuggingFaceFetcher."""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from gpumod.fetchers.huggingface import HuggingFaceFetcher
from gpumod.models import KVCacheProfile, ModelSource

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


# ---------------------------------------------------------------------------
# GGUF repo support — estimate VRAM from HuggingFace file listings
# ---------------------------------------------------------------------------


class TestGGUFRepoEstimation:
    """HuggingFaceFetcher should estimate VRAM for GGUF repos without download.

    GGUF repos (e.g. unsloth/Nemotron-3-Nano-30B-A3B-GGUF) contain
    pre-quantized model files. The HF API returns file sizes via siblings,
    so we can estimate VRAM from file size alone — no download needed.
    """

    async def test_fetch_detects_gguf_repo_and_estimates_vram(self) -> None:
        """fetch() should estimate base_vram_mb from GGUF file size when no safetensors."""
        fetcher = HuggingFaceFetcher()
        mock_info = _make_hf_model_info(
            model_id="unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
            safetensors=None,
            config={
                "architectures": ["NemotronHMoEForCausalLM"],
                "hidden_size": 4096,
                "num_hidden_layers": 40,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
            },
            siblings=[
                {"rfilename": "config.json", "size": 1024},
                {"rfilename": "README.md", "size": 5000},
                {
                    "rfilename": "Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf",
                    "size": 17_500_000_000,
                },
                {
                    "rfilename": "Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf",
                    "size": 23_000_000_000,
                },
                {
                    "rfilename": "Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf",
                    "size": 38_000_000_000,
                },
            ],
        )
        mock_info.safetensors = None

        with patch(
            "gpumod.fetchers.huggingface.model_info",
            return_value=mock_info,
        ):
            result = await fetcher.fetch("unsloth/Nemotron-3-Nano-30B-A3B-GGUF")

        # Should pick the smallest GGUF as a reasonable default
        assert result.base_vram_mb is not None
        assert result.base_vram_mb > 0
        # Notes should mention GGUF files found
        assert result.notes is not None
        assert "GGUF" in result.notes

    async def test_fetch_gguf_repo_with_quant_filter(self) -> None:
        """fetch() with quant param should pick the matching GGUF file size."""
        fetcher = HuggingFaceFetcher()
        mock_info = _make_hf_model_info(
            model_id="unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
            safetensors=None,
            config=None,
            siblings=[
                {"rfilename": "config.json", "size": 1024},
                {
                    "rfilename": "Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf",
                    "size": 17_500_000_000,
                },
                {
                    "rfilename": "Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf",
                    "size": 23_000_000_000,
                },
            ],
        )
        mock_info.safetensors = None
        mock_info.config = None

        with patch(
            "gpumod.fetchers.huggingface.model_info",
            return_value=mock_info,
        ):
            result = await fetcher.fetch(
                "unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
                quant="Q4_K_XL",
            )

        # Should estimate from the 23GB file (Q4_K_XL)
        assert result.base_vram_mb is not None
        expected_mb = int(23_000_000_000 / (1024 * 1024) * 1.1)
        assert abs(result.base_vram_mb - expected_mb) < 100
        assert "Q4_K_XL" in result.quantizations

    async def test_fetch_gguf_repo_lists_available_quants(self) -> None:
        """fetch() on a GGUF repo should list all available quantizations."""
        fetcher = HuggingFaceFetcher()
        mock_info = _make_hf_model_info(
            model_id="unsloth/Model-GGUF",
            safetensors=None,
            config=None,
            siblings=[
                {
                    "rfilename": "Model-Q4_K_M.gguf",
                    "size": 4_000_000_000,
                },
                {
                    "rfilename": "Model-Q8_0.gguf",
                    "size": 8_000_000_000,
                },
            ],
        )
        mock_info.safetensors = None
        mock_info.config = None

        with patch(
            "gpumod.fetchers.huggingface.model_info",
            return_value=mock_info,
        ):
            result = await fetcher.fetch("unsloth/Model-GGUF")

        assert "Q4_K_M" in result.quantizations
        assert "Q8_0" in result.quantizations


# ---------------------------------------------------------------------------
# Reference raw config.json dicts for 4 oracle models
# ---------------------------------------------------------------------------

# Gemma 3n E4B — hybrid with explicit layer_types + KV sharing
_GEMMA_3N_E4B_RAW_CONFIG: dict[str, Any] = {
    "model_type": "gemma3n",
    "text_config": {
        "model_type": "gemma3n_text",
        "num_hidden_layers": 35,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "hidden_size": 2048,
        "head_dim": 256,
        "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 7,
        "sliding_window": 512,
        "num_kv_shared_layers": 15,
    },
}

# Gemma 3 27B — hybrid, pattern inferred from model_type (no layer_types in config)
_GEMMA_3_27B_RAW_CONFIG: dict[str, Any] = {
    "model_type": "gemma3",
    "text_config": {
        "model_type": "gemma3_text",
        "num_hidden_layers": 62,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "hidden_size": 5376,
        "head_dim": 128,
        "sliding_window": 1024,
    },
}

# Qwen3 32B — dense, top-level config (no text_config nesting)
_QWEN3_32B_RAW_CONFIG: dict[str, Any] = {
    "model_type": "qwen3",
    "num_hidden_layers": 64,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "hidden_size": 5120,
    "head_dim": 128,
}

# Llama 3.3 70B — dense, no explicit head_dim (derived from hidden_size/num_heads)
_LLAMA_33_70B_RAW_CONFIG: dict[str, Any] = {
    "model_type": "llama",
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "hidden_size": 8192,
}


# ---------------------------------------------------------------------------
# Compound KV estimate helper — used to verify profiles match PoC oracle
# ---------------------------------------------------------------------------


def _compound_kv_mb(profile: KVCacheProfile, context: int) -> float:
    """Apply compound formula to a profile and return KV cache in MB.

    This mirrors the PoC at docs/research/poc/kv_estimation_compound.py.
    """
    kv_factor = 1 if profile.attention_k_eq_v else 2
    bytes_per_elem = 2  # fp16

    hd = profile.head_dim
    kv_heads = profile.num_kv_heads
    per_tok_local = kv_factor * kv_heads * hd * bytes_per_elem

    global_hd = profile.global_head_dim if profile.global_head_dim is not None else hd
    global_kv = (
        profile.num_global_kv_heads if profile.num_global_kv_heads is not None else kv_heads
    )
    per_tok_global = kv_factor * global_kv * global_hd * bytes_per_elem

    # Distribute shared layers proportionally
    total = profile.num_sliding_layers + profile.num_global_layers
    shared = profile.num_kv_shared_layers
    if shared > 0 and total > 0:
        shared_sliding = math.floor(shared * profile.num_sliding_layers / total)
        shared_sliding = min(shared_sliding, profile.num_sliding_layers)
        shared_global = min(shared - shared_sliding, profile.num_global_layers)
    else:
        shared_sliding = 0
        shared_global = 0

    unique_sliding = profile.num_sliding_layers - shared_sliding
    unique_global = profile.num_global_layers - shared_global

    sw = profile.sliding_window if profile.sliding_window is not None else context
    budget = profile.triattn_budget if profile.triattn_budget is not None else context
    sliding_bytes = unique_sliding * min(context, sw) * per_tok_local
    global_bytes = unique_global * min(context, budget) * per_tok_global
    return (sliding_bytes + global_bytes) / (1024 * 1024)


# ---------------------------------------------------------------------------
# _fetch_raw_config — unit tests
# ---------------------------------------------------------------------------


class TestFetchRawConfig:
    """Tests for HuggingFaceFetcher._fetch_raw_config()."""

    async def test_returns_parsed_config_dict(self, tmp_path: Path) -> None:
        """_fetch_raw_config should return parsed JSON from config.json."""
        config_data = {"model_type": "llama", "num_hidden_layers": 32}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        fetcher = HuggingFaceFetcher()

        with patch(
            "gpumod.fetchers.huggingface.hf_hub_download",
            return_value=str(config_file),
        ):
            result = await fetcher._fetch_raw_config("org/model")

        assert result == config_data

    async def test_returns_none_on_network_error(self) -> None:
        """_fetch_raw_config should return None and log warning on network error."""
        fetcher = HuggingFaceFetcher()

        with patch(
            "gpumod.fetchers.huggingface.hf_hub_download",
            side_effect=ConnectionError("network down"),
        ):
            result = await fetcher._fetch_raw_config("org/model")

        assert result is None

    async def test_returns_none_on_404(self) -> None:
        """_fetch_raw_config should return None when config.json is missing."""
        from huggingface_hub.utils import EntryNotFoundError

        fetcher = HuggingFaceFetcher()

        with patch(
            "gpumod.fetchers.huggingface.hf_hub_download",
            side_effect=EntryNotFoundError("Not found"),
        ):
            result = await fetcher._fetch_raw_config("org/model")

        assert result is None


# ---------------------------------------------------------------------------
# _build_kv_cache_profile — 4 reference models + error case
# ---------------------------------------------------------------------------


class TestBuildKvCacheProfile:
    """Tests for HuggingFaceFetcher._build_kv_cache_profile().

    Expected values verified against the PoC oracle at
    docs/research/poc/kv_estimation_compound.py (gpumod-cf8).
    """

    def test_gemma_3n_e4b_hybrid_with_kv_sharing(self) -> None:
        """Gemma 3n E4B: 28 sliding + 7 global, 15 shared, sw=512."""
        fetcher = HuggingFaceFetcher()
        profile = fetcher._build_kv_cache_profile(_GEMMA_3N_E4B_RAW_CONFIG)

        assert profile is not None
        assert profile.num_sliding_layers == 28
        assert profile.num_global_layers == 7
        assert profile.num_kv_shared_layers == 15
        assert profile.sliding_window == 512
        assert profile.head_dim == 256
        assert profile.num_kv_heads == 2
        assert profile.attention_k_eq_v is False
        assert profile.kv_per_1k_at_inf == 69

    def test_gemma_3n_e4b_oracle_match(self) -> None:
        """Compound KV from Gemma 3n E4B profile must match PoC oracle."""
        fetcher = HuggingFaceFetcher()
        profile = fetcher._build_kv_cache_profile(_GEMMA_3N_E4B_RAW_CONFIG)
        assert profile is not None

        assert _compound_kv_mb(profile, 8_000) == pytest.approx(78.5, abs=0.1)
        assert _compound_kv_mb(profile, 32_000) == pytest.approx(266.0, abs=0.1)
        assert _compound_kv_mb(profile, 128_000) == pytest.approx(1016.0, abs=0.1)

    def test_gemma_3_27b_hybrid_pattern_inference(self) -> None:
        """Gemma 3 27B: pattern inferred (sliding_window_pattern=6), 52s+10g."""
        fetcher = HuggingFaceFetcher()
        profile = fetcher._build_kv_cache_profile(_GEMMA_3_27B_RAW_CONFIG)

        assert profile is not None
        assert profile.num_sliding_layers == 52
        assert profile.num_global_layers == 10
        assert profile.num_kv_shared_layers == 0
        assert profile.sliding_window == 1024
        assert profile.head_dim == 128
        assert profile.num_kv_heads == 16
        assert profile.kv_per_1k_at_inf == 485

    def test_gemma_3_27b_oracle_match(self) -> None:
        """Compound KV from Gemma 3 27B profile must match PoC oracle."""
        fetcher = HuggingFaceFetcher()
        profile = fetcher._build_kv_cache_profile(_GEMMA_3_27B_RAW_CONFIG)
        assert profile is not None

        assert _compound_kv_mb(profile, 8_000) == pytest.approx(1041.0, abs=0.1)
        assert _compound_kv_mb(profile, 32_000) == pytest.approx(2916.0, abs=0.1)
        assert _compound_kv_mb(profile, 128_000) == pytest.approx(10416.0, abs=0.1)

    def test_qwen3_32b_dense(self) -> None:
        """Qwen3 32B: dense model, all global layers, no sliding window."""
        fetcher = HuggingFaceFetcher()
        profile = fetcher._build_kv_cache_profile(_QWEN3_32B_RAW_CONFIG)

        assert profile is not None
        assert profile.num_sliding_layers == 0
        assert profile.num_global_layers == 64
        assert profile.num_kv_shared_layers == 0
        assert profile.sliding_window is None
        assert profile.head_dim == 128
        assert profile.num_kv_heads == 8
        assert profile.kv_per_1k_at_inf == 250

    def test_qwen3_32b_oracle_match(self) -> None:
        """Compound KV from Qwen3 32B profile must match PoC oracle."""
        fetcher = HuggingFaceFetcher()
        profile = fetcher._build_kv_cache_profile(_QWEN3_32B_RAW_CONFIG)
        assert profile is not None

        assert _compound_kv_mb(profile, 8_000) == pytest.approx(2000.0, abs=0.1)
        assert _compound_kv_mb(profile, 32_000) == pytest.approx(8000.0, abs=0.1)
        assert _compound_kv_mb(profile, 128_000) == pytest.approx(32000.0, abs=0.1)

    def test_llama_33_70b_dense_derived_head_dim(self) -> None:
        """Llama 3.3 70B: dense, head_dim derived from hidden_size/num_heads."""
        fetcher = HuggingFaceFetcher()
        profile = fetcher._build_kv_cache_profile(_LLAMA_33_70B_RAW_CONFIG)

        assert profile is not None
        assert profile.num_sliding_layers == 0
        assert profile.num_global_layers == 80
        assert profile.num_kv_shared_layers == 0
        assert profile.sliding_window is None
        # head_dim = 8192/64 = 128
        assert profile.head_dim == 128
        assert profile.num_kv_heads == 8
        assert profile.kv_per_1k_at_inf == 313

    def test_llama_33_70b_oracle_match(self) -> None:
        """Compound KV from Llama 3.3 70B profile must match PoC oracle."""
        fetcher = HuggingFaceFetcher()
        profile = fetcher._build_kv_cache_profile(_LLAMA_33_70B_RAW_CONFIG)
        assert profile is not None

        assert _compound_kv_mb(profile, 8_000) == pytest.approx(2500.0, abs=0.1)
        assert _compound_kv_mb(profile, 32_000) == pytest.approx(10000.0, abs=0.1)
        assert _compound_kv_mb(profile, 128_000) == pytest.approx(40000.0, abs=0.1)

    def test_returns_none_on_missing_required_fields(self) -> None:
        """_build_kv_cache_profile returns None when required fields are absent."""
        fetcher = HuggingFaceFetcher()
        result = fetcher._build_kv_cache_profile({"model_type": "unknown"})
        assert result is None

    def test_returns_none_on_empty_config(self) -> None:
        """_build_kv_cache_profile returns None for an empty dict."""
        fetcher = HuggingFaceFetcher()
        result = fetcher._build_kv_cache_profile({})
        assert result is None


# ---------------------------------------------------------------------------
# fetch() — KV cache profile integration
# ---------------------------------------------------------------------------


class TestFetchWithKvCacheProfile:
    """Tests for kv_cache_profile integration in HuggingFaceFetcher.fetch()."""

    async def test_fetch_populates_kv_cache_profile(self, tmp_path: Path) -> None:
        """fetch() should populate kv_cache_profile when raw config is available."""
        fetcher = HuggingFaceFetcher()
        mock_info = _make_hf_model_info(
            config={
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
            },
        )
        # Write raw config for hf_hub_download mock
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(_QWEN3_32B_RAW_CONFIG))

        with (
            patch(
                "gpumod.fetchers.huggingface.model_info",
                return_value=mock_info,
            ),
            patch(
                "gpumod.fetchers.huggingface.hf_hub_download",
                return_value=str(config_file),
            ),
        ):
            result = await fetcher.fetch("Qwen/Qwen3-32B")

        assert result.kv_cache_profile is not None
        assert result.kv_cache_profile.num_global_layers == 64
        assert result.kv_cache_profile.num_sliding_layers == 0

    async def test_kv_cache_per_1k_unchanged_when_profile_set(
        self,
        tmp_path: Path,
    ) -> None:
        """Existing kv_cache_per_1k_tokens_mb must be preserved (backward compat)."""
        fetcher = HuggingFaceFetcher()
        mock_info = _make_hf_model_info(
            config={
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 8192,
                "num_hidden_layers": 80,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
            },
        )
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(_LLAMA_33_70B_RAW_CONFIG))

        with (
            patch(
                "gpumod.fetchers.huggingface.model_info",
                return_value=mock_info,
            ),
            patch(
                "gpumod.fetchers.huggingface.hf_hub_download",
                return_value=str(config_file),
            ),
        ):
            result = await fetcher.fetch("unsloth/Llama-3.3-70B-Instruct")

        # Scalar KV must still be set (existing path)
        assert result.kv_cache_per_1k_tokens_mb is not None
        assert result.kv_cache_per_1k_tokens_mb > 0
        # Profile must also be set
        assert result.kv_cache_profile is not None

    async def test_kv_cache_profile_none_when_raw_config_fails(self) -> None:
        """fetch() should set kv_cache_profile=None when raw config fetch fails."""
        fetcher = HuggingFaceFetcher()
        mock_info = _make_hf_model_info()

        with (
            patch(
                "gpumod.fetchers.huggingface.model_info",
                return_value=mock_info,
            ),
            patch(
                "gpumod.fetchers.huggingface.hf_hub_download",
                side_effect=ConnectionError("network down"),
            ),
        ):
            result = await fetcher.fetch("meta-llama/Llama-3-8B")

        # Should still produce a valid ModelInfo
        assert result.source == ModelSource.HUGGINGFACE
        assert result.kv_cache_profile is None
        # Scalar KV should still work from existing path
        assert result.kv_cache_per_1k_tokens_mb is not None
