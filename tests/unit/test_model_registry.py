"""Tests for gpumod.registry — ModelRegistry."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from gpumod.db import Database
from gpumod.models import KVCacheProfile, ModelInfo, ModelSource
from gpumod.registry import ModelRegistry, estimate_kv_mb

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


# ---------------------------------------------------------------------------
# estimate_kv_mb — compound formula (gpumod-cf8)
# ---------------------------------------------------------------------------

# Reference KVCacheProfile fixtures derived from PoC oracle values.

GEMMA_3N_E4B_PROFILE = KVCacheProfile(
    num_sliding_layers=28,
    num_global_layers=7,
    num_kv_shared_layers=15,
    sliding_window=512,
    head_dim=256,
    num_kv_heads=2,
)

GEMMA_3_27B_PROFILE = KVCacheProfile(
    num_sliding_layers=52,
    num_global_layers=10,
    num_kv_shared_layers=0,
    sliding_window=1024,
    head_dim=128,
    num_kv_heads=16,
)

QWEN3_32B_PROFILE = KVCacheProfile(
    num_sliding_layers=0,
    num_global_layers=64,
    head_dim=128,
    num_kv_heads=8,
)

LLAMA33_70B_PROFILE = KVCacheProfile(
    num_sliding_layers=0,
    num_global_layers=80,
    head_dim=128,
    num_kv_heads=8,
)


class TestEstimateKvMb:
    """Tests for the estimate_kv_mb pure function (compound formula)."""

    def test_gemma_3n_e4b_8k(self) -> None:
        """Gemma 3n E4B at 8K context: oracle = 78.5 MB."""
        result = estimate_kv_mb(GEMMA_3N_E4B_PROFILE, 8_000)
        assert abs(result - 78.5) < 1.0

    def test_gemma_3n_e4b_32k(self) -> None:
        """Gemma 3n E4B at 32K context: oracle = 266.0 MB."""
        result = estimate_kv_mb(GEMMA_3N_E4B_PROFILE, 32_000)
        assert abs(result - 266.0) < 1.0

    def test_gemma_3n_e4b_128k(self) -> None:
        """Gemma 3n E4B at 128K context: oracle = 1016.0 MB."""
        result = estimate_kv_mb(GEMMA_3N_E4B_PROFILE, 128_000)
        assert abs(result - 1016.0) < 1.0

    def test_gemma_3_27b_8k(self) -> None:
        """Gemma 3 27B at 8K context: oracle = 1041.0 MB."""
        result = estimate_kv_mb(GEMMA_3_27B_PROFILE, 8_000)
        assert abs(result - 1041.0) < 1.0

    def test_gemma_3_27b_32k(self) -> None:
        """Gemma 3 27B at 32K context: oracle = 2916.0 MB."""
        result = estimate_kv_mb(GEMMA_3_27B_PROFILE, 32_000)
        assert abs(result - 2916.0) < 1.0

    def test_gemma_3_27b_128k(self) -> None:
        """Gemma 3 27B at 128K context: oracle = 10416.0 MB."""
        result = estimate_kv_mb(GEMMA_3_27B_PROFILE, 128_000)
        assert abs(result - 10416.0) < 1.0

    def test_qwen3_32b_8k(self) -> None:
        """Qwen3 32B (dense) at 8K context: oracle = 2000.0 MB."""
        result = estimate_kv_mb(QWEN3_32B_PROFILE, 8_000)
        assert abs(result - 2000.0) < 1.0

    def test_qwen3_32b_128k(self) -> None:
        """Qwen3 32B (dense) at 128K context: oracle = 32000.0 MB."""
        result = estimate_kv_mb(QWEN3_32B_PROFILE, 128_000)
        assert abs(result - 32000.0) < 1.0

    def test_llama33_70b_8k(self) -> None:
        """Llama 3.3 70B (dense) at 8K context: oracle = 2500.0 MB."""
        result = estimate_kv_mb(LLAMA33_70B_PROFILE, 8_000)
        assert abs(result - 2500.0) < 1.0

    def test_llama33_70b_128k(self) -> None:
        """Llama 3.3 70B (dense) at 128K context: oracle = 40000.0 MB."""
        result = estimate_kv_mb(LLAMA33_70B_PROFILE, 128_000)
        assert abs(result - 40000.0) < 1.0


# ---------------------------------------------------------------------------
# estimate_vram — compound formula integration
# ---------------------------------------------------------------------------


class TestEstimateVramCompound:
    """Tests for estimate_vram using kv_cache_profile (compound path)."""

    async def test_uses_compound_when_profile_present(self, registry: ModelRegistry) -> None:
        """estimate_vram prefers compound formula when kv_cache_profile is set."""
        model = ModelInfo(
            id="google/gemma-3n-E4B-it",
            source=ModelSource.HUGGINGFACE,
            base_vram_mb=5000,
            kv_cache_per_1k_tokens_mb=69,
            kv_cache_profile=GEMMA_3N_E4B_PROFILE,
        )
        with patch.object(registry._db, "get_model", new_callable=AsyncMock, return_value=model):
            result = await registry.estimate_vram("google/gemma-3n-E4B-it", context_size=8_000)

        # compound: 78.5 MB → total = 5000 + 78 = 5078
        expected_kv = estimate_kv_mb(GEMMA_3N_E4B_PROFILE, 8_000)
        expected = 5000 + int(expected_kv)
        assert result == expected

    async def test_compound_differs_from_linear(self, registry: ModelRegistry) -> None:
        """Compound formula should give different (lower) value than linear for hybrid models."""
        model = ModelInfo(
            id="google/gemma-3n-E4B-it",
            source=ModelSource.HUGGINGFACE,
            base_vram_mb=5000,
            kv_cache_per_1k_tokens_mb=69,
            kv_cache_profile=GEMMA_3N_E4B_PROFILE,
        )
        with patch.object(registry._db, "get_model", new_callable=AsyncMock, return_value=model):
            compound_result = await registry.estimate_vram(
                "google/gemma-3n-E4B-it", context_size=8_000
            )

        # Linear path would give: 5000 + int((8000/1000) * 69) = 5000 + 552 = 5552
        linear_result = 5000 + int((8_000 / 1000) * 69)
        assert compound_result < linear_result

    async def test_falls_back_to_linear_when_no_profile(
        self, registry: ModelRegistry, db: Database
    ) -> None:
        """estimate_vram uses linear formula when kv_cache_profile is None."""
        model = _make_model_info(
            base_vram_mb=16384,
            kv_cache_per_1k_tokens_mb=64,
        )
        await db.insert_model(model)

        result = await registry.estimate_vram("meta-llama/Llama-3-8B", context_size=4096)
        # Linear: 16384 + int((4096/1000) * 64) = 16384 + 262 = 16646
        assert result == 16646

    async def test_backward_compat_dense_model_exact_match(
        self, registry: ModelRegistry, db: Database
    ) -> None:
        """Dense model with no profile produces exactly the current linear value."""
        model = _make_model_info(
            model_id="dense/test-model",
            base_vram_mb=20000,
            kv_cache_per_1k_tokens_mb=250,
        )
        await db.insert_model(model)

        result = await registry.estimate_vram("dense/test-model", context_size=8000)
        # Exact linear formula: 20000 + int((8000/1000) * 250) = 20000 + 2000 = 22000
        assert result == 22000

    async def test_compound_gemma_3_27b(self, registry: ModelRegistry) -> None:
        """Gemma 3 27B compound estimate matches PoC oracle within tolerance."""
        model = ModelInfo(
            id="google/gemma-3-27b-it",
            source=ModelSource.HUGGINGFACE,
            base_vram_mb=14000,
            kv_cache_per_1k_tokens_mb=485,
            kv_cache_profile=GEMMA_3_27B_PROFILE,
        )
        with patch.object(registry._db, "get_model", new_callable=AsyncMock, return_value=model):
            result = await registry.estimate_vram("google/gemma-3-27b-it", context_size=8_000)

        expected_kv = estimate_kv_mb(GEMMA_3_27B_PROFILE, 8_000)
        expected = 14000 + int(expected_kv)
        assert result == expected

    async def test_compound_qwen3_dense(self, registry: ModelRegistry) -> None:
        """Dense model with profile: compound matches linear for all-global layers."""
        model = ModelInfo(
            id="Qwen/Qwen3-32B",
            source=ModelSource.HUGGINGFACE,
            base_vram_mb=18000,
            kv_cache_per_1k_tokens_mb=250,
            kv_cache_profile=QWEN3_32B_PROFILE,
        )
        with patch.object(registry._db, "get_model", new_callable=AsyncMock, return_value=model):
            result = await registry.estimate_vram("Qwen/Qwen3-32B", context_size=8_000)

        # For dense model, compound = linear in raw KV
        expected_kv = estimate_kv_mb(QWEN3_32B_PROFILE, 8_000)
        expected = 18000 + int(expected_kv)
        assert result == expected

    async def test_no_kv_info_returns_base_only(self, registry: ModelRegistry) -> None:
        """Model with neither kv_cache_per_1k nor kv_cache_profile returns base_vram_mb."""
        model = ModelInfo(
            id="bare/model",
            source=ModelSource.LOCAL,
            base_vram_mb=10000,
            kv_cache_per_1k_tokens_mb=None,
            kv_cache_profile=None,
        )
        with patch.object(registry._db, "get_model", new_callable=AsyncMock, return_value=model):
            result = await registry.estimate_vram("bare/model", context_size=32_000)

        assert result == 10000
