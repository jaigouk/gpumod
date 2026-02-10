"""Tests for HuggingFaceSearcher - TDD RED phase.

Following TDD workflow:
1. RED: Write failing tests first (this file)
2. GREEN: Implement minimal code to pass
3. REFACTOR: Improve code quality
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

# These imports will fail initially (RED phase)
# from gpumod.discovery.hf_searcher import HuggingFaceSearcher
# from gpumod.discovery.protocols import ModelSearcher, SearchResult


# ---------------------------------------------------------------------------
# Protocol Tests (ISP/DIP)
# ---------------------------------------------------------------------------


class TestModelSearcherProtocol:
    """Tests for ModelSearcher protocol definition."""

    def test_protocol_exists(self) -> None:
        """ModelSearcher protocol is importable."""
        from gpumod.discovery.protocols import ModelSearcher

        assert ModelSearcher is not None

    def test_protocol_is_runtime_checkable(self) -> None:
        """ModelSearcher is runtime_checkable for isinstance checks."""
        from gpumod.discovery.protocols import ModelSearcher

        assert hasattr(ModelSearcher, "__protocol_attrs__") or hasattr(
            ModelSearcher, "_is_protocol"
        )

    def test_protocol_has_search_method(self) -> None:
        """ModelSearcher requires async search method."""
        from gpumod.discovery.protocols import ModelSearcher

        # Protocol should define search method signature
        assert hasattr(ModelSearcher, "search")


class TestSearchResultDataclass:
    """Tests for SearchResult dataclass."""

    def test_search_result_exists(self) -> None:
        """SearchResult dataclass is importable."""
        from gpumod.discovery.protocols import SearchResult

        assert SearchResult is not None

    def test_search_result_has_required_fields(self) -> None:
        """SearchResult has all required fields."""
        from gpumod.discovery.protocols import SearchResult

        result = SearchResult(
            repo_id="unsloth/Qwen-GGUF",
            name="Qwen",
            description="A model",
            last_modified=datetime.now(tz=UTC),
            tags=("gguf", "llm"),
            model_format="gguf",
            driver_hint="llamacpp",
        )

        assert result.repo_id == "unsloth/Qwen-GGUF"
        assert result.name == "Qwen"
        assert result.model_format == "gguf"
        assert result.driver_hint == "llamacpp"

    def test_search_result_is_frozen(self) -> None:
        """SearchResult is immutable."""
        from gpumod.discovery.protocols import SearchResult

        result = SearchResult(
            repo_id="test/model",
            name="Test",
            description=None,
            last_modified=datetime.now(tz=UTC),
            tags=(),
            model_format="gguf",
            driver_hint="llamacpp",
        )

        with pytest.raises((AttributeError, TypeError)):
            result.repo_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HuggingFaceSearcher Tests
# ---------------------------------------------------------------------------


class TestHuggingFaceSearcherInit:
    """Tests for HuggingFaceSearcher initialization."""

    def test_searcher_exists(self) -> None:
        """HuggingFaceSearcher class is importable."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        assert HuggingFaceSearcher is not None

    def test_searcher_implements_protocol(self) -> None:
        """HuggingFaceSearcher implements ModelSearcher protocol."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher
        from gpumod.discovery.protocols import ModelSearcher

        searcher = HuggingFaceSearcher()
        assert isinstance(searcher, ModelSearcher)

    def test_searcher_default_cache_ttl(self) -> None:
        """Default cache TTL is 1 hour."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        searcher = HuggingFaceSearcher()
        assert searcher._cache_ttl == 3600

    def test_searcher_custom_cache_ttl(self) -> None:
        """Cache TTL can be customized."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        searcher = HuggingFaceSearcher(cache_ttl_seconds=1800)
        assert searcher._cache_ttl == 1800


class TestHuggingFaceSearcherSearch:
    """Tests for HuggingFaceSearcher.search() method."""

    @pytest.fixture
    def mock_hf_api(self) -> MagicMock:
        """Mock HuggingFace Hub API."""
        return MagicMock()

    @pytest.fixture
    def sample_hf_models(self) -> list[MagicMock]:
        """Sample models from HF API response."""
        gguf_model = MagicMock()
        gguf_model.id = "NexaAI/DeepSeek-OCR-GGUF"
        gguf_model.tags = ["gguf", "text-generation"]
        gguf_model.lastModified = datetime.now(tz=UTC)
        gguf_model.cardData = {"description": "DeepSeek OCR GGUF"}

        safetensors_model = MagicMock()
        safetensors_model.id = "deepseek-ai/DeepSeek-OCR-2"
        safetensors_model.tags = ["transformers", "safetensors", "text-generation"]
        safetensors_model.lastModified = datetime.now(tz=UTC)
        safetensors_model.cardData = {"description": "DeepSeek OCR v2"}

        return [gguf_model, safetensors_model]

    @pytest.mark.asyncio
    async def test_search_returns_list(self, sample_hf_models: list[MagicMock]) -> None:
        """search() returns list of SearchResult."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher
        from gpumod.discovery.protocols import SearchResult

        with patch("gpumod.discovery.hf_searcher.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = sample_hf_models
            mock_api_cls.return_value = mock_api

            searcher = HuggingFaceSearcher()
            results = await searcher.search(query="deepseek ocr")

        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_with_driver_filter_gguf(self, sample_hf_models: list[MagicMock]) -> None:
        """driver='llamacpp' returns only GGUF models."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        with patch("gpumod.discovery.hf_searcher.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = sample_hf_models
            mock_api_cls.return_value = mock_api

            searcher = HuggingFaceSearcher()
            results = await searcher.search(query="deepseek", driver="llamacpp")

        assert len(results) >= 1
        assert all(r.driver_hint == "llamacpp" for r in results)
        assert all(r.model_format == "gguf" for r in results)

    @pytest.mark.asyncio
    async def test_search_with_driver_filter_vllm(self, sample_hf_models: list[MagicMock]) -> None:
        """driver='vllm' returns only Safetensors models."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        with patch("gpumod.discovery.hf_searcher.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = sample_hf_models
            mock_api_cls.return_value = mock_api

            searcher = HuggingFaceSearcher()
            results = await searcher.search(query="deepseek", driver="vllm")

        assert len(results) >= 1
        assert all(r.driver_hint == "vllm" for r in results)
        assert all(r.model_format == "safetensors" for r in results)

    @pytest.mark.asyncio
    async def test_search_with_author_filter(self, sample_hf_models: list[MagicMock]) -> None:
        """author parameter filters by organization."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        with patch("gpumod.discovery.hf_searcher.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = sample_hf_models
            mock_api_cls.return_value = mock_api

            searcher = HuggingFaceSearcher()
            await searcher.search(query="model", author="NexaAI")

            # Verify API was called with author filter
            call_kwargs = mock_api.list_models.call_args
            assert call_kwargs.kwargs.get("author") == "NexaAI"

    @pytest.mark.asyncio
    async def test_search_detects_gguf_format(self) -> None:
        """GGUF models are detected correctly."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        gguf_model = MagicMock()
        gguf_model.id = "user/model-GGUF"
        gguf_model.tags = ["gguf"]
        gguf_model.lastModified = datetime.now(tz=UTC)
        gguf_model.cardData = None

        with patch("gpumod.discovery.hf_searcher.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [gguf_model]
            mock_api_cls.return_value = mock_api

            searcher = HuggingFaceSearcher()
            results = await searcher.search(query="model")

        assert len(results) == 1
        assert results[0].model_format == "gguf"
        assert results[0].driver_hint == "llamacpp"

    @pytest.mark.asyncio
    async def test_search_detects_safetensors_format(self) -> None:
        """Safetensors models are detected correctly."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        st_model = MagicMock()
        st_model.id = "user/model"
        st_model.tags = ["transformers", "safetensors", "pytorch"]
        st_model.lastModified = datetime.now(tz=UTC)
        st_model.cardData = None

        with patch("gpumod.discovery.hf_searcher.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [st_model]
            mock_api_cls.return_value = mock_api

            searcher = HuggingFaceSearcher()
            results = await searcher.search(query="model")

        assert len(results) == 1
        assert results[0].model_format == "safetensors"
        assert results[0].driver_hint == "vllm"

    @pytest.mark.asyncio
    async def test_search_caches_results(self) -> None:
        """Results are cached to avoid repeated API calls."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        model = MagicMock()
        model.id = "user/model-GGUF"
        model.tags = ["gguf"]
        model.lastModified = datetime.now(tz=UTC)
        model.cardData = None

        with patch("gpumod.discovery.hf_searcher.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [model]
            mock_api_cls.return_value = mock_api

            searcher = HuggingFaceSearcher()

            # First call
            await searcher.search(query="model")
            # Second call (should use cache)
            await searcher.search(query="model")

        # API should only be called once
        assert mock_api.list_models.call_count == 1

    @pytest.mark.asyncio
    async def test_search_force_refresh_bypasses_cache(self) -> None:
        """force_refresh=True bypasses cache."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        model = MagicMock()
        model.id = "user/model-GGUF"
        model.tags = ["gguf"]
        model.lastModified = datetime.now(tz=UTC)
        model.cardData = None

        with patch("gpumod.discovery.hf_searcher.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [model]
            mock_api_cls.return_value = mock_api

            searcher = HuggingFaceSearcher()

            await searcher.search(query="model")
            await searcher.search(query="model", force_refresh=True)

        # API should be called twice (cache bypassed)
        assert mock_api.list_models.call_count == 2


class TestHuggingFaceSearcherBackwardCompat:
    """Tests for backward compatibility with UnslothModelLister."""

    @pytest.mark.asyncio
    async def test_default_search_includes_gguf_filter(self) -> None:
        """Default search includes GGUF models."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        with patch("gpumod.discovery.hf_searcher.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = []
            mock_api_cls.return_value = mock_api

            searcher = HuggingFaceSearcher()
            await searcher.search(query="model")

            # Should search for GGUF by default (backward compat)
            call_kwargs = mock_api.list_models.call_args
            # Either filter contains gguf or search includes gguf
            assert "search" in call_kwargs.kwargs or "filter" in call_kwargs.kwargs


# ---------------------------------------------------------------------------
# Format Detection Tests
# ---------------------------------------------------------------------------


class TestModelFormatDetection:
    """Tests for model format detection from tags."""

    def test_detect_gguf_from_tags(self) -> None:
        """Detect GGUF format from tags."""
        from gpumod.discovery.hf_searcher import detect_model_format

        assert detect_model_format(["gguf", "llm"]) == "gguf"
        assert detect_model_format(["GGUF", "text-generation"]) == "gguf"

    def test_detect_safetensors_from_tags(self) -> None:
        """Detect Safetensors format from tags."""
        from gpumod.discovery.hf_searcher import detect_model_format

        assert detect_model_format(["safetensors", "transformers"]) == "safetensors"
        assert detect_model_format(["pytorch", "safetensors"]) == "safetensors"

    def test_detect_format_prefers_gguf(self) -> None:
        """When both formats present, prefer GGUF (specific quantization)."""
        from gpumod.discovery.hf_searcher import detect_model_format

        # Some repos have both - GGUF is more specific
        result = detect_model_format(["gguf", "safetensors", "transformers"])
        assert result == "gguf"

    def test_detect_unknown_format(self) -> None:
        """Unknown format returns 'unknown'."""
        from gpumod.discovery.hf_searcher import detect_model_format

        assert detect_model_format(["pytorch", "text-generation"]) == "unknown"
        assert detect_model_format([]) == "unknown"


class TestDriverHintMapping:
    """Tests for format-to-driver mapping."""

    def test_gguf_maps_to_llamacpp(self) -> None:
        """GGUF format maps to llamacpp driver."""
        from gpumod.discovery.hf_searcher import get_driver_hint

        assert get_driver_hint("gguf") == "llamacpp"

    def test_safetensors_maps_to_vllm(self) -> None:
        """Safetensors format maps to vllm driver."""
        from gpumod.discovery.hf_searcher import get_driver_hint

        assert get_driver_hint("safetensors") == "vllm"

    def test_unknown_maps_to_none(self) -> None:
        """Unknown format maps to None."""
        from gpumod.discovery.hf_searcher import get_driver_hint

        assert get_driver_hint("unknown") is None
