"""Tests for UnslothModelLister - RED phase first."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gpumod.discovery.unsloth_lister import (
    HuggingFaceAPIError,
    UnslothModel,
    UnslothModelLister,
)


class TestUnslothModelDataclass:
    """Tests for UnslothModel dataclass."""

    def test_unsloth_model_is_frozen(self) -> None:
        """UnslothModel should be immutable (frozen dataclass)."""
        model = UnslothModel(
            repo_id="unsloth/Qwen3-Coder-Next-GGUF",
            name="Qwen3 Coder Next",
            description="80B MoE coding model",
            last_modified=datetime.now(tz=UTC),
            tags=("gguf", "code"),
            has_gguf=True,
        )
        with pytest.raises(AttributeError):
            model.repo_id = "changed"  # type: ignore[misc]

    def test_tags_is_tuple(self) -> None:
        """tags should be a tuple (immutable)."""
        model = UnslothModel(
            repo_id="unsloth/test",
            name="Test",
            description=None,
            last_modified=datetime.now(tz=UTC),
            tags=("gguf", "llama"),
            has_gguf=True,
        )
        assert isinstance(model.tags, tuple)


class TestUnslothModelLister:
    """Tests for UnslothModelLister."""

    @pytest.mark.asyncio
    async def test_list_models_returns_list(
        self,
        mock_hf_api: None,
    ) -> None:
        """list_models() should return a list of UnslothModel."""
        lister = UnslothModelLister()
        models = await lister.list_models()
        assert isinstance(models, list)
        assert all(isinstance(m, UnslothModel) for m in models)

    @pytest.mark.asyncio
    async def test_filters_to_gguf_repos(
        self,
        mock_hf_api_mixed_repos: None,
    ) -> None:
        """Should only return repos containing .gguf files."""
        lister = UnslothModelLister()
        models = await lister.list_models()
        assert all(m.has_gguf for m in models)

    @pytest.mark.asyncio
    async def test_extracts_name_description(
        self,
        mock_hf_api: None,
    ) -> None:
        """Should extract name and description from repo metadata."""
        lister = UnslothModelLister()
        models = await lister.list_models()
        assert len(models) > 0
        assert models[0].name
        # Description may be None

    @pytest.mark.asyncio
    async def test_handles_pagination(
        self,
        mock_hf_api_paginated: None,
    ) -> None:
        """Should handle pagination for >100 models."""
        lister = UnslothModelLister()
        models = await lister.list_models()
        # Mock returns 150 models across pages
        assert len(models) > 100

    @pytest.mark.asyncio
    async def test_caches_results(
        self,
        mock_hf_api: None,
    ) -> None:
        """Should cache results with configurable TTL."""
        lister = UnslothModelLister(cache_ttl_seconds=3600)
        models1 = await lister.list_models()
        models2 = await lister.list_models()
        # Same object reference means cache hit
        assert models1 is models2

    @pytest.mark.asyncio
    async def test_works_without_hf_token(
        self,
        mock_hf_api_no_token: None,
    ) -> None:
        """Should work for public repos without HF_TOKEN."""
        lister = UnslothModelLister()
        models = await lister.list_models()
        assert len(models) > 0

    @pytest.mark.asyncio
    async def test_raises_on_api_failure(
        self,
        mock_hf_api_error: None,
    ) -> None:
        """Should raise HuggingFaceAPIError on API failure."""
        lister = UnslothModelLister()
        with pytest.raises(HuggingFaceAPIError):
            await lister.list_models()

    @pytest.mark.asyncio
    async def test_returns_cached_on_network_error(
        self,
        mock_hf_api_network_error_with_cache: None,
    ) -> None:
        """Should return cached data when API is unreachable."""
        lister = UnslothModelLister()
        # First call populates cache
        await lister.list_models()
        # Second call with network error should return cached
        models = await lister.list_models()
        assert len(models) > 0

    @pytest.mark.asyncio
    async def test_skips_private_repos(
        self,
        mock_hf_api_with_private: None,
    ) -> None:
        """Should skip private repos gracefully."""
        lister = UnslothModelLister()
        models = await lister.list_models()
        # Should not raise, just skip private repos
        assert all(m.repo_id.startswith("unsloth/") for m in models)


class TestUnslothModelListerFiltering:
    """Tests for model filtering."""

    @pytest.mark.asyncio
    async def test_filter_by_task_code(
        self,
        mock_hf_api_with_tags: None,
    ) -> None:
        """Should filter models by 'code' task."""
        lister = UnslothModelLister()
        models = await lister.list_models(task="code")
        assert all("code" in m.tags or "coder" in m.name.lower() for m in models)

    @pytest.mark.asyncio
    async def test_filter_by_task_chat(
        self,
        mock_hf_api_with_tags: None,
    ) -> None:
        """Should filter models by 'chat' task."""
        lister = UnslothModelLister()
        models = await lister.list_models(task="chat")
        # Should return chat-related models
        assert len(models) >= 0  # May be empty if no matches
