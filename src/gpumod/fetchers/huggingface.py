"""HuggingFace model info fetcher using huggingface_hub.

Fetches model metadata from the HuggingFace Hub API for VRAM estimation
and model registry purposes. All network calls are wrapped in
asyncio.to_thread() since huggingface_hub is a synchronous library.

Supports both standard model repos (safetensors) and GGUF repos,
estimating VRAM from file sizes without requiring a download.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from datetime import UTC, datetime

from huggingface_hub import hf_hub_download, model_info

from gpumod.models import KVCacheProfile, ModelInfo, ModelSource

logger = logging.getLogger(__name__)

# Regex for valid HuggingFace model IDs: org/model with alphanumeric, hyphens, underscores, dots
_MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")

# Known GGUF quantization patterns (longest first to avoid partial matches)
_GGUF_QUANT_PATTERNS: tuple[str, ...] = (
    "Q4_K_XL",
    "Q8_K_XL",
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q2_K",
    "Q3_K_M",
    "Q3_K_S",
    "IQ4_XS",
    "IQ4_NL",
)

# Overhead factor for VRAM estimation from GGUF file size
_VRAM_OVERHEAD_FACTOR = 1.1


class HuggingFaceFetcher:
    """Fetches model metadata from HuggingFace Hub.

    Uses huggingface_hub.model_info() to retrieve model configuration,
    safetensors metadata, and architecture details for VRAM estimation.
    For GGUF repos, estimates VRAM from file sizes without downloading.
    """

    async def fetch(
        self,
        model_id: str,
        *,
        quant: str | None = None,
    ) -> ModelInfo:
        """Fetch model info from HuggingFace Hub.

        Parameters
        ----------
        model_id:
            HuggingFace model identifier in "org/model" format.
        quant:
            Optional quantization filter for GGUF repos (e.g. "Q4_K_XL").
            When set, VRAM is estimated from the matching GGUF file size.

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
            info = await asyncio.to_thread(
                model_info,
                model_id,
                files_metadata=True,
            )
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

        # Detect GGUF files from siblings
        gguf_files = self._find_gguf_files(info)
        quantizations = self._extract_quants(gguf_files)

        # Estimate VRAM
        base_vram_mb, notes = self._estimate_base_vram(
            gguf_files,
            quant,
            parameters_b,
        )

        # Estimate KV cache (scalar — backward compat, unchanged)
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

        # Build structured KV cache profile from raw config.json
        kv_cache_profile: KVCacheProfile | None = None
        raw_config = await self._fetch_raw_config(model_id)
        if raw_config is not None:
            kv_cache_profile = self._build_kv_cache_profile(raw_config)

        return ModelInfo(
            id=model_id,
            source=ModelSource.HUGGINGFACE,
            parameters_b=parameters_b,
            architecture=architecture,
            base_vram_mb=base_vram_mb,
            kv_cache_per_1k_tokens_mb=kv_cache_per_1k,
            kv_cache_profile=kv_cache_profile,
            quantizations=quantizations,
            fetched_at=datetime.now(tz=UTC).isoformat(),
            notes=notes,
        )

    # ------------------------------------------------------------------
    # KV cache profile: raw config fetch + profile build
    # ------------------------------------------------------------------

    async def _fetch_raw_config(self, repo_id: str) -> dict[str, object] | None:
        """Fetch and parse raw ``config.json`` from HuggingFace Hub.

        Uses :func:`huggingface_hub.hf_hub_download` which caches locally.
        Returns ``None`` on any error (gated repos, network issues, missing
        file) so the caller can fall back to the scalar KV estimate.
        """
        try:
            path: str = await asyncio.to_thread(
                hf_hub_download,
                repo_id=repo_id,
                filename="config.json",
            )
            raw: dict[str, object] = await asyncio.to_thread(self._read_json, path)
            return raw
        except Exception:
            logger.warning(
                "Failed to fetch raw config.json for %s; falling back to scalar KV estimate",
                repo_id,
            )
            return None

    @staticmethod
    def _read_json(path: str) -> dict[str, object]:
        """Read and parse a JSON file (sync, meant for ``asyncio.to_thread``)."""
        with open(path) as fh:
            result: dict[str, object] = json.load(fh)
        return result

    def _build_kv_cache_profile(
        self,
        raw_config: dict[str, object],
    ) -> KVCacheProfile | None:
        """Build a :class:`KVCacheProfile` from a raw ``config.json`` dict.

        Handles both top-level configs (Qwen, Llama) and nested
        ``text_config`` (Gemma 3, 3n, 4).  Returns ``None`` when required
        fields are missing.
        """
        # Resolve text_config nesting for multimodal models
        tc = raw_config.get("text_config", raw_config)
        if not isinstance(tc, dict):
            return None

        # Extract required fields
        num_layers = tc.get("num_hidden_layers")
        num_heads = tc.get("num_attention_heads")
        hidden_size = tc.get("hidden_size")
        if not isinstance(num_layers, int) or not isinstance(num_heads, int):
            return None
        if not isinstance(hidden_size, int) or num_heads == 0:
            return None

        num_kv_heads_raw = tc.get("num_key_value_heads", num_heads)
        num_kv_heads = num_kv_heads_raw if isinstance(num_kv_heads_raw, int) else num_heads

        # Head dim: explicit field if present, else derived
        head_dim_raw = tc.get("head_dim")
        head_dim: int = head_dim_raw if isinstance(head_dim_raw, int) else hidden_size // num_heads

        # --- Determine layer types ---
        layer_types: list[str] | None = None
        raw_layer_types = tc.get("layer_types")
        if isinstance(raw_layer_types, list) and raw_layer_types:
            layer_types = [str(lt) for lt in raw_layer_types]

        sliding_window_raw = tc.get("sliding_window")
        sliding_window: int | None = (
            sliding_window_raw if isinstance(sliding_window_raw, int) else None
        )

        sliding_window_pattern_raw = tc.get("sliding_window_pattern")
        sliding_window_pattern: int | None = (
            sliding_window_pattern_raw if isinstance(sliding_window_pattern_raw, int) else None
        )

        # Derive layer_types from sliding_window_pattern if explicit list absent
        if layer_types is None and sliding_window_pattern is not None:
            layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(num_layers)
            ]

        # Gemma 3 family: infer pattern=6 when only sliding_window is present
        # Source: transformers v5.1.0 Gemma3TextConfig sliding_window_pattern=6
        model_type = tc.get("model_type", "")
        if (
            layer_types is None
            and sliding_window is not None
            and sliding_window_pattern is None
            and isinstance(model_type, str)
            and "gemma3" in model_type
        ):
            layer_types = [
                "sliding_attention" if bool((i + 1) % 6) else "full_attention"
                for i in range(num_layers)
            ]

        # Count layer types
        if layer_types is not None:
            n_sliding = sum(1 for lt in layer_types if lt == "sliding_attention")
            n_global = sum(1 for lt in layer_types if lt == "full_attention")
        else:
            # Dense model: all layers are global
            n_sliding = 0
            n_global = num_layers

        # --- KV sharing ---
        shared_raw = tc.get("num_kv_shared_layers", 0)
        num_kv_shared: int = shared_raw if isinstance(shared_raw, int) else 0

        # --- Global-layer overrides (Gemma 4 style) ---
        global_head_dim_raw = tc.get("global_head_dim")
        global_head_dim: int | None = (
            global_head_dim_raw if isinstance(global_head_dim_raw, int) else None
        )
        global_kv_heads_raw = tc.get("num_global_key_value_heads")
        num_global_kv_heads: int | None = (
            global_kv_heads_raw if isinstance(global_kv_heads_raw, int) else None
        )

        # --- attention_k_eq_v ---
        k_eq_v_raw = tc.get("attention_k_eq_v", False)
        attention_k_eq_v: bool = k_eq_v_raw is True

        # --- kv_per_1k_at_inf (linear-equivalent rate) ---
        kv_per_1k_bytes = 2 * num_layers * num_kv_heads * head_dim * 2 * 1000
        kv_per_1k_at_inf = math.ceil(kv_per_1k_bytes / (1024 * 1024))

        return KVCacheProfile(
            num_sliding_layers=n_sliding,
            num_global_layers=n_global,
            num_kv_shared_layers=num_kv_shared,
            sliding_window=sliding_window,
            head_dim=head_dim,
            global_head_dim=global_head_dim,
            num_kv_heads=num_kv_heads,
            num_global_kv_heads=num_global_kv_heads,
            attention_k_eq_v=attention_k_eq_v,
            kv_per_1k_at_inf=kv_per_1k_at_inf,
        )

    def _estimate_base_vram(
        self,
        gguf_files: list[tuple[str, int]],
        quant: str | None,
        parameters_b: float | None,
    ) -> tuple[int | None, str | None]:
        """Estimate base VRAM from GGUF file sizes or parameter count."""
        if gguf_files:
            chosen = self._pick_gguf_file(gguf_files, quant)
            if chosen is not None:
                filename, size_bytes = chosen
                vram = self._estimate_vram_from_file_size(size_bytes)
                size_gb = size_bytes / (1024**3)
                notes = (
                    f"GGUF repo: {len(gguf_files)} file(s). "
                    f"Estimated from {filename} ({size_gb:.1f} GB)"
                )
                return vram, notes
        elif parameters_b is not None:
            return self._estimate_vram_mb(parameters_b, dtype_bytes=2), None
        return None, None

    @staticmethod
    def _find_gguf_files(info: object) -> list[tuple[str, int]]:
        """Extract GGUF filenames and sizes from HF model info siblings.

        Returns
        -------
        list[tuple[str, int]]
            List of (filename, size_bytes) tuples for .gguf files.
        """
        siblings = getattr(info, "siblings", None)
        if not siblings:
            return []
        result: list[tuple[str, int]] = []
        for s in siblings:
            name = getattr(s, "rfilename", "") or ""
            size = getattr(s, "size", None)
            if name.lower().endswith(".gguf") and size is not None and size > 0:
                result.append((name, size))
        return result

    @staticmethod
    def _extract_quants(gguf_files: list[tuple[str, int]]) -> list[str]:
        """Extract quantization types from GGUF filenames."""
        quants: list[str] = []
        for filename, _ in gguf_files:
            upper = filename.upper()
            for pattern in _GGUF_QUANT_PATTERNS:
                if pattern in upper and pattern not in quants:
                    quants.append(pattern)
                    break
        return quants

    @staticmethod
    def _pick_gguf_file(
        gguf_files: list[tuple[str, int]],
        quant: str | None,
    ) -> tuple[str, int] | None:
        """Pick a GGUF file, optionally filtering by quantization.

        If quant is specified, returns the matching file.
        Otherwise returns the smallest GGUF file as a reasonable default.
        """
        if not gguf_files:
            return None
        if quant is not None:
            quant_upper = quant.upper()
            for filename, size in gguf_files:
                if quant_upper in filename.upper():
                    return (filename, size)
            return None
        return min(gguf_files, key=lambda x: x[1])

    @staticmethod
    def _estimate_vram_from_file_size(file_size_bytes: int) -> int:
        """Estimate VRAM from GGUF file size with 10% overhead."""
        if file_size_bytes == 0:
            return 0
        size_mb = file_size_bytes / (1024 * 1024)
        return math.ceil(size_mb * _VRAM_OVERHEAD_FACTOR)

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
