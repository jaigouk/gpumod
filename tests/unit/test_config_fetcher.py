"""Tests for ConfigFetcher — fetches model config.json from HuggingFace.

TDD: RED phase - these tests define the ConfigFetcher interface.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config_json() -> dict[str, Any]:
    """Sample config.json for a standard model (Llama-style)."""
    return {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "max_position_embeddings": 131072,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "torch_dtype": "bfloat16",
    }


@pytest.fixture
def moe_config_json() -> dict[str, Any]:
    """Sample config.json for a MoE model (Mixtral-style)."""
    return {
        "architectures": ["MixtralForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_experts_per_tok": 2,
        "num_local_experts": 8,
        "vocab_size": 32000,
    }


@pytest.fixture
def deepseek_moe_config_json() -> dict[str, Any]:
    """Sample config.json for DeepSeek MoE model."""
    return {
        "architectures": ["DeepseekV2ForCausalLM"],
        "hidden_size": 5120,
        "intermediate_size": 12288,
        "max_position_embeddings": 163840,
        "n_routed_experts": 160,
        "num_experts_per_tok": 6,
        "num_hidden_layers": 60,
        "vocab_size": 102400,
    }


# ---------------------------------------------------------------------------
# TestModelConfig - dataclass structure
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_fields(self) -> None:
        """ModelConfig has all required fields."""
        from gpumod.discovery.config_fetcher import ModelConfig

        config = ModelConfig(
            repo_id="meta-llama/Llama-3.1-8B",
            architectures=["LlamaForCausalLM"],
            total_params=8_000_000_000,
            is_moe=False,
            num_experts=None,
            context_length=131072,
            vocab_size=128256,
            raw_config={"key": "value"},
        )

        assert config.repo_id == "meta-llama/Llama-3.1-8B"
        assert config.architectures == ["LlamaForCausalLM"]
        assert config.total_params == 8_000_000_000
        assert config.is_moe is False
        assert config.num_experts is None
        assert config.context_length == 131072
        assert config.vocab_size == 128256
        assert config.raw_config == {"key": "value"}

    def test_model_config_moe_fields(self) -> None:
        """ModelConfig correctly represents MoE model."""
        from gpumod.discovery.config_fetcher import ModelConfig

        config = ModelConfig(
            repo_id="mistralai/Mixtral-8x7B-v0.1",
            architectures=["MixtralForCausalLM"],
            total_params=46_700_000_000,
            is_moe=True,
            num_experts=8,
            context_length=32768,
            vocab_size=32000,
            raw_config={},
        )

        assert config.is_moe is True
        assert config.num_experts == 8


# ---------------------------------------------------------------------------
# TestConfigFetcher - fetching config.json
# ---------------------------------------------------------------------------


class TestConfigFetcher:
    """Tests for ConfigFetcher class."""

    async def test_fetch_returns_model_config(
        self, sample_config_json: dict[str, Any]
    ) -> None:
        """fetch() returns ModelConfig with parsed data."""
        from gpumod.discovery.config_fetcher import ConfigFetcher, ModelConfig

        fetcher = ConfigFetcher()

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_config_json
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            result = await fetcher.fetch("meta-llama/Llama-3.1-8B")

            assert isinstance(result, ModelConfig)
            assert result.repo_id == "meta-llama/Llama-3.1-8B"
            assert result.architectures == ["LlamaForCausalLM"]
            assert result.context_length == 131072
            assert result.vocab_size == 128256

    async def test_fetch_detects_moe_model(
        self, moe_config_json: dict[str, Any]
    ) -> None:
        """fetch() correctly detects MoE models."""
        from gpumod.discovery.config_fetcher import ConfigFetcher

        fetcher = ConfigFetcher()

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = moe_config_json
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            result = await fetcher.fetch("mistralai/Mixtral-8x7B-v0.1")

            assert result.is_moe is True
            assert result.num_experts == 8

    async def test_fetch_detects_deepseek_moe(
        self, deepseek_moe_config_json: dict[str, Any]
    ) -> None:
        """fetch() correctly detects DeepSeek MoE with n_routed_experts."""
        from gpumod.discovery.config_fetcher import ConfigFetcher

        fetcher = ConfigFetcher()

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = deepseek_moe_config_json
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            result = await fetcher.fetch("deepseek-ai/DeepSeek-V2")

            assert result.is_moe is True
            assert result.num_experts == 160

    async def test_fetch_uses_correct_url(self) -> None:
        """fetch() fetches from correct HuggingFace URL."""
        from gpumod.discovery.config_fetcher import ConfigFetcher

        fetcher = ConfigFetcher()

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"architectures": ["Test"]}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            await fetcher.fetch("unsloth/Qwen3-Coder-Next-GGUF")

            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            url = call_args[0][0]
            assert "huggingface.co" in url
            assert "unsloth/Qwen3-Coder-Next-GGUF" in url
            assert "config.json" in url

    async def test_fetch_raises_on_404(self) -> None:
        """fetch() raises ConfigNotFoundError on 404."""
        import httpx

        from gpumod.discovery.config_fetcher import ConfigFetcher, ConfigNotFoundError

        fetcher = ConfigFetcher()

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            # Preserve real exception types for catching
            mock_httpx.HTTPStatusError = httpx.HTTPStatusError
            mock_httpx.RequestError = httpx.RequestError

            mock_response = MagicMock()
            mock_response.status_code = 404

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            with pytest.raises(ConfigNotFoundError):
                await fetcher.fetch("fake/nonexistent-repo")

    async def test_fetch_raises_on_repo_not_found(self) -> None:
        """fetch() raises ConfigNotFoundError when repo doesn't exist (via 404)."""
        import httpx

        from gpumod.discovery.config_fetcher import ConfigFetcher, ConfigNotFoundError

        fetcher = ConfigFetcher()

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            # Preserve real exception types for catching
            mock_httpx.HTTPStatusError = httpx.HTTPStatusError
            mock_httpx.RequestError = httpx.RequestError

            mock_response = MagicMock()
            mock_response.status_code = 404

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            # 404 raises ConfigNotFoundError (can't distinguish repo vs file not found)
            with pytest.raises(ConfigNotFoundError):
                await fetcher.fetch("fake/nonexistent-repo")

    async def test_fetch_preserves_raw_config(
        self, sample_config_json: dict[str, Any]
    ) -> None:
        """fetch() preserves raw config.json in result."""
        from gpumod.discovery.config_fetcher import ConfigFetcher

        fetcher = ConfigFetcher()

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_config_json
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            result = await fetcher.fetch("test/model")

            assert result.raw_config == sample_config_json


# ---------------------------------------------------------------------------
# TestConfigFetcherCache - TTL caching
# ---------------------------------------------------------------------------


class TestConfigFetcherCache:
    """Tests for ConfigFetcher caching behavior."""

    async def test_cache_hit_returns_cached_result(
        self, sample_config_json: dict[str, Any]
    ) -> None:
        """Second fetch uses cache, doesn't make HTTP request."""
        from gpumod.discovery.config_fetcher import ConfigFetcher

        fetcher = ConfigFetcher(cache_ttl_seconds=3600)

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_config_json
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            # First fetch - cache miss
            result1 = await fetcher.fetch("test/model")
            assert mock_client.get.call_count == 1

            # Second fetch - cache hit
            result2 = await fetcher.fetch("test/model")
            assert mock_client.get.call_count == 1  # No additional call

            assert result1.repo_id == result2.repo_id

    async def test_cache_miss_after_ttl_expires(
        self, sample_config_json: dict[str, Any]
    ) -> None:
        """Cache expires after TTL, makes new request."""
        from gpumod.discovery.config_fetcher import ConfigFetcher

        # Very short TTL for testing
        fetcher = ConfigFetcher(cache_ttl_seconds=0)

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_config_json
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            # First fetch
            await fetcher.fetch("test/model")

            # Wait for TTL to expire (immediate with TTL=0)
            await asyncio.sleep(0.01)

            # Second fetch - cache expired
            await fetcher.fetch("test/model")
            assert mock_client.get.call_count == 2

    async def test_different_repos_have_separate_cache(
        self, sample_config_json: dict[str, Any]
    ) -> None:
        """Different repos have separate cache entries."""
        from gpumod.discovery.config_fetcher import ConfigFetcher

        fetcher = ConfigFetcher(cache_ttl_seconds=3600)

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_config_json
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            await fetcher.fetch("repo/one")
            await fetcher.fetch("repo/two")
            await fetcher.fetch("repo/one")  # Cache hit
            await fetcher.fetch("repo/two")  # Cache hit

            assert mock_client.get.call_count == 2

    def test_default_ttl_is_one_hour(self) -> None:
        """Default cache TTL is 1 hour (3600 seconds)."""
        from gpumod.discovery.config_fetcher import ConfigFetcher

        fetcher = ConfigFetcher()
        assert fetcher._ttl == 3600


# ---------------------------------------------------------------------------
# TestConfigFetcherParseLogic - parsing config.json fields
# ---------------------------------------------------------------------------


class TestConfigFetcherParseLogic:
    """Tests for config.json parsing logic."""

    async def test_parse_max_position_embeddings(self) -> None:
        """Parses max_position_embeddings as context_length."""
        from gpumod.discovery.config_fetcher import ConfigFetcher

        fetcher = ConfigFetcher()

        config_json = {
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 8192,
        }

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = config_json
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            result = await fetcher.fetch("test/model")
            assert result.context_length == 8192

    async def test_parse_max_sequence_length_fallback(self) -> None:
        """Falls back to max_sequence_length for context."""
        from gpumod.discovery.config_fetcher import ConfigFetcher

        fetcher = ConfigFetcher()

        config_json = {
            "architectures": ["GPTNeoXForCausalLM"],
            "max_sequence_length": 4096,
        }

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = config_json
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            result = await fetcher.fetch("test/model")
            assert result.context_length == 4096

    async def test_parse_missing_architectures(self) -> None:
        """Handles missing architectures field."""
        from gpumod.discovery.config_fetcher import ConfigFetcher

        fetcher = ConfigFetcher()

        config_json = {
            "model_type": "llama",
            "hidden_size": 4096,
        }

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = config_json
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            result = await fetcher.fetch("test/model")
            assert result.architectures == []

    async def test_parse_num_local_experts_for_mixtral(self) -> None:
        """Parses num_local_experts for Mixtral-style MoE."""
        from gpumod.discovery.config_fetcher import ConfigFetcher

        fetcher = ConfigFetcher()

        config_json = {
            "architectures": ["MixtralForCausalLM"],
            "num_local_experts": 8,
        }

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = config_json
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            result = await fetcher.fetch("test/model")
            assert result.is_moe is True
            assert result.num_experts == 8

    async def test_non_moe_model(self) -> None:
        """Non-MoE model has is_moe=False and num_experts=None."""
        from gpumod.discovery.config_fetcher import ConfigFetcher

        fetcher = ConfigFetcher()

        config_json = {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
        }

        with patch("gpumod.discovery.config_fetcher.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = config_json
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.AsyncClient.return_value = mock_client

            result = await fetcher.fetch("test/model")
            assert result.is_moe is False
            assert result.num_experts is None
