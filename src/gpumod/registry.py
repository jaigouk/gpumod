"""Model registry with VRAM estimation.

Combines HuggingFace and GGUF fetchers with database storage for
model metadata and VRAM calculations. Provides a unified interface
for registering, querying, and estimating resource requirements
for ML models.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from gpumod.fetchers.gguf import GGUFFetcher
from gpumod.fetchers.huggingface import HuggingFaceFetcher
from gpumod.models import ModelInfo, ModelSource

if TYPE_CHECKING:
    from gpumod.db import Database


class ModelRegistry:
    """Registry for ML models with VRAM estimation capabilities.

    Combines fetchers (HuggingFace, GGUF) with DB storage for model
    metadata and VRAM calculations.

    Parameters
    ----------
    db:
        The Database instance for model persistence.
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._hf_fetcher = HuggingFaceFetcher()
        self._gguf_fetcher = GGUFFetcher()

    async def register(self, model_id: str, source: ModelSource, **kwargs: Any) -> ModelInfo:
        """Register a model by fetching its info and storing in DB.

        Parameters
        ----------
        model_id:
            Identifier for the model. For HuggingFace: "org/model".
            For GGUF: filename or path identifier. For LOCAL: arbitrary string.
        source:
            The model source type (HUGGINGFACE, GGUF, or LOCAL).
        **kwargs:
            Additional keyword arguments. For GGUF: requires ``file_path``.
            For LOCAL: accepts ``parameters_b``, ``architecture``,
            ``base_vram_mb``, ``kv_cache_per_1k_tokens_mb``, ``quantizations``,
            ``capabilities``, ``notes``.

        Returns
        -------
        ModelInfo
            The registered model metadata.

        Raises
        ------
        ValueError
            If required kwargs are missing (e.g., file_path for GGUF).
        RuntimeError
            If the fetcher fails to retrieve model info.
        """
        if source == ModelSource.HUGGINGFACE:
            model_info = await self._hf_fetcher.fetch(model_id)
        elif source == ModelSource.GGUF:
            file_path = kwargs.get("file_path")
            if file_path is None:
                msg = "file_path is required for GGUF source"
                raise ValueError(msg)
            model_info = await self._gguf_fetcher.fetch(str(file_path))
        elif source == ModelSource.LOCAL:
            model_info = ModelInfo(
                id=model_id,
                source=ModelSource.LOCAL,
                parameters_b=kwargs.get("parameters_b"),
                architecture=kwargs.get("architecture"),
                base_vram_mb=kwargs.get("base_vram_mb"),
                kv_cache_per_1k_tokens_mb=kwargs.get("kv_cache_per_1k_tokens_mb"),
                quantizations=kwargs.get("quantizations", []),
                capabilities=kwargs.get("capabilities", []),
                notes=kwargs.get("notes"),
                fetched_at=datetime.now(tz=UTC).isoformat(),
            )
        else:
            msg = f"Unsupported model source: {source!r}"
            raise ValueError(msg)

        await self._db.insert_model(model_info)
        return model_info

    async def get(self, model_id: str) -> ModelInfo | None:
        """Get model info from registry (DB lookup).

        Parameters
        ----------
        model_id:
            The model identifier to look up.

        Returns
        -------
        ModelInfo | None
            The model metadata, or None if not found.
        """
        return await self._db.get_model(model_id)

    async def list_models(self) -> list[ModelInfo]:
        """List all registered models.

        Returns
        -------
        list[ModelInfo]
            All models in the registry, ordered by ID.
        """
        return await self._db.list_models()

    async def estimate_vram(
        self,
        model_id: str,
        context_size: int = 4096,
        quantization: str | None = None,
    ) -> int:
        """Estimate total VRAM for a model with given context.

        Calculates: total = base_vram + (context_size / 1000) * kv_cache_per_1k

        Parameters
        ----------
        model_id:
            The model identifier to estimate VRAM for.
        context_size:
            Number of context tokens (default 4096).
        quantization:
            Optional quantization type (reserved for future use).

        Returns
        -------
        int
            Estimated total VRAM in megabytes.

        Raises
        ------
        ValueError
            If the model is not found or has no VRAM estimation data.
        """
        model = await self._db.get_model(model_id)
        if model is None:
            msg = f"Model not found in registry: {model_id!r}"
            raise ValueError(msg)

        if model.base_vram_mb is None:
            msg = f"No VRAM estimation data for model: {model_id!r}"
            raise ValueError(msg)

        total = model.base_vram_mb

        if model.kv_cache_per_1k_tokens_mb is not None:
            kv_addition = int((context_size / 1000) * model.kv_cache_per_1k_tokens_mb)
            total += kv_addition

        return total

    async def remove(self, model_id: str) -> None:
        """Remove a model from the registry.

        This operation is idempotent -- removing a nonexistent model
        does not raise an error.

        Parameters
        ----------
        model_id:
            The model identifier to remove.
        """
        await self._db.delete_model(model_id)
