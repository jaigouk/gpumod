"""Tests for gpumod.registry — ModelRegistry."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from gpumod.db import Database
from gpumod.models import ModelInfo, ModelSource
from gpumod.registry import ModelRegistry

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_info(
    model_id: str = "meta-llama/Llama-3-8B",
    source: ModelSource = ModelSource.HUGGINGFACE,
    parameters_b: float | None = 8.0,
    architecture: str | None = "LlamaForCausalLM",
    base_vram_mb: int | None = 16384,
    kv_cache_per_1k_tokens_mb: int | None = 64,
) -> ModelInfo:
    """Create a ModelInfo for testing."""
    return ModelInfo(
        id=model_id,
        source=source,
        parameters_b=parameters_b,
        architecture=architecture,
        base_vram_mb=base_vram_mb,
        kv_cache_per_1k_tokens_mb=kv_cache_per_1k_tokens_mb,
        fetched_at="2025-01-15T10:00:00",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """Create a connected Database for testing."""
    database = Database(tmp_path / "model_registry_test.db")
    await database.connect()
    yield database  # type: ignore[misc]
    await database.close()


@pytest.fixture
def registry(db: Database) -> ModelRegistry:
    """Create a ModelRegistry backed by the test DB."""
    return ModelRegistry(db)


# ---------------------------------------------------------------------------
# register — HuggingFace source
# ---------------------------------------------------------------------------


class TestRegisterHuggingFace:
    """Tests for ModelRegistry.register() with HuggingFace source."""

    async def test_register_hf_calls_fetcher_and_stores(self, registry: ModelRegistry) -> None:
        """register() should call HF fetcher and store result in DB."""
        mock_info = _make_model_info()

        with patch.object(
            registry._hf_fetcher, "fetch", new_callable=AsyncMock, return_value=mock_info
        ):
            result = await registry.register("meta-llama/Llama-3-8B", ModelSource.HUGGINGFACE)

        assert result.id == "meta-llama/Llama-3-8B"
        assert result.source == ModelSource.HUGGINGFACE

        # Verify it was stored in DB
        stored = await registry.get("meta-llama/Llama-3-8B")
        assert stored is not None
        assert stored.id == "meta-llama/Llama-3-8B"

    async def test_register_hf_passes_model_id_to_fetcher(self, registry: ModelRegistry) -> None:
        """register() should pass the correct model_id to the HF fetcher."""
        mock_info = _make_model_info(model_id="mistralai/Mistral-7B-v0.1")
        mock_fetch = AsyncMock(return_value=mock_info)

        with patch.object(registry._hf_fetcher, "fetch", mock_fetch):
            await registry.register("mistralai/Mistral-7B-v0.1", ModelSource.HUGGINGFACE)

        mock_fetch.assert_called_once_with("mistralai/Mistral-7B-v0.1", quant=None)


# ---------------------------------------------------------------------------
# register — GGUF source
# ---------------------------------------------------------------------------


class TestRegisterGGUF:
    """Tests for ModelRegistry.register() with GGUF source."""

    async def test_register_gguf_calls_fetcher_and_stores(self, registry: ModelRegistry) -> None:
        """register() should call GGUF fetcher and store result in DB."""
        mock_info = _make_model_info(
            model_id="Llama-3-8B-Q4_K_M.gguf",
            source=ModelSource.GGUF,
        )

        with patch.object(
            registry._gguf_fetcher, "fetch", new_callable=AsyncMock, return_value=mock_info
        ):
            result = await registry.register(
                "Llama-3-8B-Q4_K_M.gguf",
                ModelSource.GGUF,
                file_path="/models/Llama-3-8B-Q4_K_M.gguf",
            )

        assert result.id == "Llama-3-8B-Q4_K_M.gguf"
        assert result.source == ModelSource.GGUF

    async def test_register_gguf_passes_file_path_to_fetcher(
        self, registry: ModelRegistry
    ) -> None:
        """register() should pass file_path kwarg to the GGUF fetcher."""
        mock_info = _make_model_info(model_id="test.gguf", source=ModelSource.GGUF)
        mock_fetch = AsyncMock(return_value=mock_info)

        with patch.object(registry._gguf_fetcher, "fetch", mock_fetch):
            await registry.register(
                "test.gguf",
                ModelSource.GGUF,
                file_path="/models/test.gguf",
            )

        mock_fetch.assert_called_once_with("/models/test.gguf")

    async def test_register_gguf_requires_file_path(self, registry: ModelRegistry) -> None:
        """register() with GGUF source should raise ValueError if no file_path."""
        with pytest.raises(ValueError, match="file_path"):
            await registry.register("test.gguf", ModelSource.GGUF)


# ---------------------------------------------------------------------------
# register — LOCAL source
# ---------------------------------------------------------------------------


class TestRegisterLocal:
    """Tests for ModelRegistry.register() with LOCAL source."""

    async def test_register_local_creates_model_from_kwargs(self, registry: ModelRegistry) -> None:
        """register() with LOCAL source should create ModelInfo from kwargs."""
        result = await registry.register(
            "my-local-model",
            ModelSource.LOCAL,
            parameters_b=7.0,
            architecture="llama",
            base_vram_mb=14000,
            kv_cache_per_1k_tokens_mb=50,
        )

        assert result.id == "my-local-model"
        assert result.source == ModelSource.LOCAL
        assert result.parameters_b == 7.0
        assert result.architecture == "llama"
        assert result.base_vram_mb == 14000
        assert result.kv_cache_per_1k_tokens_mb == 50

    async def test_register_local_stores_in_db(self, registry: ModelRegistry) -> None:
        """register() with LOCAL source should store in DB."""
        await registry.register(
            "my-local-model",
            ModelSource.LOCAL,
            base_vram_mb=14000,
        )

        stored = await registry.get("my-local-model")
        assert stored is not None
        assert stored.id == "my-local-model"
        assert stored.base_vram_mb == 14000


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    """Tests for ModelRegistry.get()."""

    async def test_get_returns_none_for_unknown(self, registry: ModelRegistry) -> None:
        """get() should return None for unregistered model."""
        result = await registry.get("nonexistent/model")
        assert result is None

    async def test_get_returns_stored_model(self, registry: ModelRegistry, db: Database) -> None:
        """get() should return ModelInfo stored in DB."""
        model = _make_model_info()
        await db.insert_model(model)

        result = await registry.get("meta-llama/Llama-3-8B")
        assert result is not None
        assert result.id == "meta-llama/Llama-3-8B"
        assert result.source == ModelSource.HUGGINGFACE


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    """Tests for ModelRegistry.list_models()."""

    async def test_list_models_empty(self, registry: ModelRegistry) -> None:
        """list_models() should return empty list when no models registered."""
        result = await registry.list_models()
        assert result == []

    async def test_list_models_returns_all(self, registry: ModelRegistry, db: Database) -> None:
        """list_models() should return all registered models."""
        model1 = _make_model_info(model_id="model-a")
        model2 = _make_model_info(model_id="model-b", source=ModelSource.GGUF)
        await db.insert_model(model1)
        await db.insert_model(model2)

        result = await registry.list_models()
        assert len(result) == 2
        ids = {m.id for m in result}
        assert ids == {"model-a", "model-b"}


# ---------------------------------------------------------------------------
# estimate_vram
# ---------------------------------------------------------------------------


class TestEstimateVram:
    """Tests for ModelRegistry.estimate_vram()."""

    async def test_estimate_vram_basic(self, registry: ModelRegistry, db: Database) -> None:
        """estimate_vram() should calculate total = base + (context/1000) * kv_cache."""
        model = _make_model_info(
            base_vram_mb=16384,
            kv_cache_per_1k_tokens_mb=64,
        )
        await db.insert_model(model)

        # total = 16384 + (4096/1000) * 64 = 16384 + 262 = 16646
        result = await registry.estimate_vram("meta-llama/Llama-3-8B", context_size=4096)
        assert result == 16646

    async def test_estimate_vram_default_context(
        self, registry: ModelRegistry, db: Database
    ) -> None:
        """estimate_vram() should default to 4096 context tokens."""
        model = _make_model_info(
            base_vram_mb=16384,
            kv_cache_per_1k_tokens_mb=64,
        )
        await db.insert_model(model)

        result = await registry.estimate_vram("meta-llama/Llama-3-8B")
        # Default context_size=4096
        expected = 16384 + int((4096 / 1000) * 64)
        assert result == expected

    async def test_estimate_vram_raises_for_unknown_model(self, registry: ModelRegistry) -> None:
        """estimate_vram() should raise ValueError for unregistered model."""
        with pytest.raises(ValueError, match="not found"):
            await registry.estimate_vram("nonexistent/model")

    async def test_estimate_vram_with_no_kv_cache_info(
        self, registry: ModelRegistry, db: Database
    ) -> None:
        """estimate_vram() should return base_vram when kv_cache info is missing."""
        model = _make_model_info(
            base_vram_mb=16384,
            kv_cache_per_1k_tokens_mb=None,
        )
        await db.insert_model(model)

        result = await registry.estimate_vram("meta-llama/Llama-3-8B", context_size=8192)
        assert result == 16384

    async def test_estimate_vram_with_no_base_vram(
        self, registry: ModelRegistry, db: Database
    ) -> None:
        """estimate_vram() should raise ValueError when base_vram_mb is None."""
        model = _make_model_info(base_vram_mb=None)
        await db.insert_model(model)

        with pytest.raises(ValueError, match="VRAM"):
            await registry.estimate_vram("meta-llama/Llama-3-8B")

    async def test_estimate_vram_large_context(
        self, registry: ModelRegistry, db: Database
    ) -> None:
        """estimate_vram() with large context should increase VRAM significantly."""
        model = _make_model_info(
            base_vram_mb=16384,
            kv_cache_per_1k_tokens_mb=64,
        )
        await db.insert_model(model)

        result = await registry.estimate_vram("meta-llama/Llama-3-8B", context_size=128000)
        # 16384 + (128000/1000) * 64 = 16384 + 8192 = 24576
        assert result == 24576


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


class TestRemove:
    """Tests for ModelRegistry.remove()."""

    async def test_remove_deletes_from_db(self, registry: ModelRegistry, db: Database) -> None:
        """remove() should delete the model from DB."""
        model = _make_model_info()
        await db.insert_model(model)

        await registry.remove("meta-llama/Llama-3-8B")

        result = await registry.get("meta-llama/Llama-3-8B")
        assert result is None

    async def test_remove_nonexistent_does_not_raise(self, registry: ModelRegistry) -> None:
        """remove() should not raise for nonexistent model (idempotent)."""
        # Should not raise
        await registry.remove("nonexistent/model")
