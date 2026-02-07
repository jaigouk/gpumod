"""HuggingFace model info fetcher using huggingface_hub.

Fetches model metadata from the HuggingFace Hub API for VRAM estimation
and model registry purposes. All network calls are wrapped in
asyncio.to_thread() since huggingface_hub is a synchronous library.
"""

from __future__ import annotations

import asyncio
import math
import re
from datetime import UTC, datetime

from huggingface_hub import model_info

from gpumod.models import ModelInfo, ModelSource

# Regex for valid HuggingFace model IDs: org/model with alphanumeric, hyphens, underscores, dots
_MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


class HuggingFaceFetcher:
    """Fetches model metadata from HuggingFace Hub.

    Uses huggingface_hub.model_info() to retrieve model configuration,
    safetensors metadata, and architecture details for VRAM estimation.
    """

    async def fetch(self, model_id: str) -> ModelInfo:
        """Fetch model info from HuggingFace Hub.

        Parameters
        ----------
        model_id:
            HuggingFace model identifier in "org/model" format.

        Returns
        -------
        ModelInfo
            Model metadata with source=ModelSource.HUGGINGFACE.

        Raises
        ------
        ValueError
            If model_id is empty or has invalid format.
        RuntimeError
            If the HuggingFace API call fails.
        """
        self._validate_model_id(model_id)

        try:
            info = await asyncio.to_thread(model_info, model_id)
        except Exception as exc:
            msg = f"Failed to fetch model info for {model_id!r}: {exc}"
            raise RuntimeError(msg) from exc

        # Extract parameter count from safetensors metadata
        parameters_b: float | None = None
        if info.safetensors is not None:
            params = info.safetensors.get("total") if isinstance(info.safetensors, dict) else None
            if params is None and isinstance(info.safetensors, dict):
                param_dict = info.safetensors.get("parameters", {})
                if isinstance(param_dict, dict) and param_dict:
                    params = next(iter(param_dict.values()))
            if params is not None:
                parameters_b = params / 1_000_000_000

        # Extract architecture from config
        architecture: str | None = None
        config = info.config if hasattr(info, "config") else None
        num_layers: int | None = None
        hidden_size: int | None = None
        num_attention_heads: int | None = None
        num_kv_heads: int | None = None

        if isinstance(config, dict):
            archs = config.get("architectures")
            if isinstance(archs, list) and archs:
                architecture = archs[0]
            num_layers = config.get("num_hidden_layers")
            hidden_size = config.get("hidden_size")
            num_attention_heads = config.get("num_attention_heads")
            num_kv_heads = config.get("num_key_value_heads", num_attention_heads)

        # Estimate VRAM
        base_vram_mb: int | None = None
        if parameters_b is not None:
            base_vram_mb = self._estimate_vram_mb(parameters_b, dtype_bytes=2)

        # Estimate KV cache
        kv_cache_per_1k: int | None = None
        if (
            num_layers is not None
            and hidden_size is not None
            and num_kv_heads is not None
            and num_attention_heads is not None
        ):
            kv_cache_per_1k = self._estimate_kv_cache_per_1k(
                num_layers=num_layers,
                hidden_size=hidden_size,
                num_kv_heads=num_kv_heads,
                num_attention_heads=num_attention_heads,
            )

        return ModelInfo(
            id=model_id,
            source=ModelSource.HUGGINGFACE,
            parameters_b=parameters_b,
            architecture=architecture,
            base_vram_mb=base_vram_mb,
            kv_cache_per_1k_tokens_mb=kv_cache_per_1k,
            fetched_at=datetime.now(tz=UTC).isoformat(),
        )

    def _estimate_vram_mb(self, parameters_b: float, dtype_bytes: int = 2) -> int:
        """Estimate base VRAM in MB from parameter count.

        Parameters
        ----------
        parameters_b:
            Number of parameters in billions.
        dtype_bytes:
            Bytes per parameter (2 for fp16, 1 for int8, 4 for fp32).

        Returns
        -------
        int
            Estimated VRAM in megabytes.
        """
        return int(parameters_b * dtype_bytes * 1024)

    def _estimate_kv_cache_per_1k(
        self,
        num_layers: int,
        hidden_size: int,
        num_kv_heads: int,
        num_attention_heads: int,
    ) -> int:
        """Estimate KV cache memory per 1K tokens in MB.

        For models with Grouped Query Attention (GQA), the KV cache size is
        proportional to the number of KV heads rather than attention heads.

        Formula:
            2 (K+V) * num_layers * num_kv_heads * head_dim * 2 (bytes/fp16) * 1000 / (1024^2)

        Parameters
        ----------
        num_layers:
            Number of transformer layers.
        hidden_size:
            Model hidden dimension size.
        num_kv_heads:
            Number of key-value heads (may differ from attention heads in GQA).
        num_attention_heads:
            Number of attention heads (used to compute head_dim).

        Returns
        -------
        int
            Estimated KV cache in MB per 1000 tokens.
        """
        if num_attention_heads == 0:
            return 0
        head_dim = hidden_size // num_attention_heads
        # 2 for K+V, 2 for fp16 bytes
        bytes_per_1k = 2 * num_layers * num_kv_heads * head_dim * 2 * 1000
        return math.ceil(bytes_per_1k / (1024 * 1024))

    @staticmethod
    def _validate_model_id(model_id: str) -> None:
        """Validate HuggingFace model ID format.

        Raises
        ------
        ValueError
            If model_id is empty, missing org/model format, or contains
            potentially dangerous characters.
        """
        if not model_id:
            msg = "model_id must not be empty"
            raise ValueError(msg)

        if not _MODEL_ID_PATTERN.match(model_id):
            msg = (
                f"Invalid model_id format: {model_id!r}. "
                "Expected 'org/model' with alphanumeric, hyphens, underscores, dots."
            )
            raise ValueError(msg)
