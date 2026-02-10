"""llama.cpp options knowledge base.

Maintains knowledge about llama-server options for intelligent
preset generation. Documents VRAM impact, recommended values,
and option interactions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class VRAMImpact(Enum):
    """VRAM impact level for an option."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"
    REDUCES = "reduces"  # Option reduces VRAM usage


@dataclass(frozen=True)
class OptionSpec:
    """Specification for a llama.cpp server option.

    Attributes:
        name: Option name (e.g., "n_gpu_layers").
        type: Python type (int, bool, str).
        default: Default value.
        description: Human-readable description.
        vram_impact: How this option affects VRAM usage.
        min_version: llama.cpp version when this option was added.
        deprecated: Whether this option is deprecated.
    """

    name: str
    type: type
    default: Any
    description: str
    vram_impact: VRAMImpact
    min_version: str | None = None
    deprecated: bool = False


@dataclass(frozen=True)
class RecommendedConfig:
    """Recommended llama.cpp configuration for a model.

    Attributes:
        n_gpu_layers: Number of layers to offload (-1 = all).
        ctx_size: Context window size.
        flash_attn: Whether to enable flash attention.
        extra_args: Additional arguments as key-value pairs.
        notes: Explanations for the recommendations.
    """

    n_gpu_layers: int
    ctx_size: int
    flash_attn: bool
    extra_args: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


class LlamaCppOptions:
    """Knowledge base for llama.cpp server options.

    Provides option specifications, recommended configurations based
    on model size and available VRAM, and configuration validation.

    Example:
        >>> options = LlamaCppOptions()
        >>> config = options.get_recommended_config(model_size_b=7)
        >>> print(f"GPU layers: {config.n_gpu_layers}")
    """

    # Documented options with metadata
    _OPTIONS: ClassVar[dict[str, OptionSpec]] = {
        "n_gpu_layers": OptionSpec(
            name="n_gpu_layers",
            type=int,
            default=-1,
            description="Number of layers to offload to GPU. -1 = all layers, 0 = CPU only.",
            vram_impact=VRAMImpact.HIGH,
        ),
        "ctx_size": OptionSpec(
            name="ctx_size",
            type=int,
            default=4096,
            description="Context window size. Larger values use more VRAM for KV cache.",
            vram_impact=VRAMImpact.MEDIUM,
        ),
        "flash_attn": OptionSpec(
            name="flash_attn",
            type=bool,
            default=False,
            description="Enable flash attention for memory-efficient attention computation.",
            vram_impact=VRAMImpact.REDUCES,
        ),
        "batch_size": OptionSpec(
            name="batch_size",
            type=int,
            default=2048,
            description="Batch size for prompt processing.",
            vram_impact=VRAMImpact.LOW,
        ),
        "ubatch_size": OptionSpec(
            name="ubatch_size",
            type=int,
            default=512,
            description="Micro-batch size for prompt processing.",
            vram_impact=VRAMImpact.LOW,
        ),
        "threads": OptionSpec(
            name="threads",
            type=int,
            default=0,
            description="Number of CPU threads. 0 = auto-detect.",
            vram_impact=VRAMImpact.NONE,
        ),
        "jinja": OptionSpec(
            name="jinja",
            type=bool,
            default=False,
            description="Enable Jinja2 template processing for chat templates.",
            vram_impact=VRAMImpact.NONE,
        ),
        "n_cpu_moe": OptionSpec(
            name="n_cpu_moe",
            type=int,
            default=0,
            description="CPU threads for MoE expert layers. Useful for large MoE models.",
            vram_impact=VRAMImpact.REDUCES,
            min_version="b4000",
        ),
        "models_max": OptionSpec(
            name="models_max",
            type=int,
            default=1,
            description="Maximum number of models to load in router mode.",
            vram_impact=VRAMImpact.HIGH,
        ),
        "no_models_autoload": OptionSpec(
            name="no_models_autoload",
            type=bool,
            default=False,
            description="Don't auto-load models on startup in router mode.",
            vram_impact=VRAMImpact.REDUCES,
        ),
    }

    # Model size tiers (in billions of parameters)
    _SIZE_TIERS: ClassVar[dict[str, tuple[float, float]]] = {
        "tiny": (0, 3),  # <3B
        "small": (3, 10),  # 3-10B
        "medium": (10, 35),  # 10-35B
        "large": (35, 80),  # 35-80B
        "xl": (80, float("inf")),  # >80B
    }

    def get_option(self, name: str) -> OptionSpec | None:
        """Get specification for an option.

        Args:
            name: Option name.

        Returns:
            OptionSpec or None if unknown.
        """
        return self._OPTIONS.get(name)

    def get_all_options(self) -> dict[str, OptionSpec]:
        """Get all documented options.

        Returns:
            Dictionary of option name to OptionSpec.
        """
        return dict(self._OPTIONS)

    def get_recommended_config(
        self,
        model_size_b: float,
        *,
        available_vram_mb: int | None = None,
        is_moe: bool = False,
        ctx_size: int = 8192,
    ) -> RecommendedConfig:
        """Get recommended configuration for a model.

        Args:
            model_size_b: Model size in billions of parameters.
            available_vram_mb: Available VRAM in MB (for partial offload calc).
            is_moe: Whether the model uses Mixture of Experts.
            ctx_size: Desired context window size.

        Returns:
            RecommendedConfig with optimal settings.
        """
        notes: list[str] = []

        # Determine size tier
        tier = self._get_size_tier(model_size_b)

        # Default to full GPU offload
        n_gpu_layers = -1
        flash_attn = True
        extra_args: dict[str, Any] = {}

        # Adjust for large models
        if tier in ("large", "xl"):
            if available_vram_mb is not None:
                # Estimate if full offload will fit
                # Rough estimate: 1GB per 1B params at Q4
                estimated_vram = int(model_size_b * 1000)  # Very rough
                if estimated_vram > available_vram_mb:
                    # Need partial offload
                    # Estimate layers based on available VRAM
                    ratio = available_vram_mb / estimated_vram
                    # Assume ~80 layers for large models
                    estimated_layers = int(80 * ratio * 0.9)  # 90% to leave headroom
                    n_gpu_layers = max(1, min(estimated_layers, 80))
                    notes.append(
                        f"Partial offload: {n_gpu_layers} layers to fit {available_vram_mb} MB"
                    )
            else:
                notes.append("Large model - consider partial offload if OOM occurs")

        # MoE specific settings
        if is_moe:
            extra_args["n_cpu_moe"] = 512  # Good default for MoE
            notes.append("MoE model: using n_cpu_moe=512 for expert layers")

        # Adjust context size based on VRAM
        final_ctx = ctx_size
        if available_vram_mb is not None and available_vram_mb < 16000:
            # Reduce context for limited VRAM
            final_ctx = min(ctx_size, 4096)
            if final_ctx < ctx_size:
                notes.append(f"Reduced context to {final_ctx} for VRAM budget")

        return RecommendedConfig(
            n_gpu_layers=n_gpu_layers,
            ctx_size=final_ctx,
            flash_attn=flash_attn,
            extra_args=extra_args,
            notes=tuple(notes),
        )

    def get_moe_config(self, experts: int = 8) -> RecommendedConfig:
        """Get recommended config for MoE models.

        Args:
            experts: Number of MoE experts.

        Returns:
            RecommendedConfig optimized for MoE.
        """
        # Scale n_cpu_moe with expert count
        n_cpu_moe = min(512, experts * 64)

        return RecommendedConfig(
            n_gpu_layers=-1,
            ctx_size=8192,
            flash_attn=True,
            extra_args={"n_cpu_moe": n_cpu_moe},
            notes=(f"MoE with {experts} experts: n_cpu_moe={n_cpu_moe}",),
        )

    def estimate_vram_overhead(
        self,
        ctx_size: int,
        *,
        num_layers: int = 32,
        hidden_size: int = 4096,
        num_kv_heads: int = 8,
    ) -> int:
        """Estimate KV cache VRAM overhead.

        Args:
            ctx_size: Context window size.
            num_layers: Number of transformer layers.
            hidden_size: Model hidden dimension.
            num_kv_heads: Number of KV heads.

        Returns:
            Estimated KV cache overhead in MB.
        """
        # KV cache formula: 2 * num_layers * num_kv_heads * head_dim * ctx_size * 2 bytes
        head_dim = hidden_size // 32  # Approximate
        bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * 2
        total_bytes = bytes_per_token * ctx_size
        return total_bytes // (1024 * 1024)

    def validate_config(self, config: RecommendedConfig) -> list[str]:
        """Validate a configuration.

        Args:
            config: Configuration to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        if config.ctx_size <= 0:
            errors.append("ctx_size must be positive")

        if config.n_gpu_layers < -1:
            errors.append("n_gpu_layers must be >= -1")

        # Check for conflicting options
        # (add more validation as needed)

        return errors

    def _get_size_tier(self, model_size_b: float) -> str:
        """Determine model size tier.

        Args:
            model_size_b: Model size in billions.

        Returns:
            Tier name (tiny, small, medium, large, xl).
        """
        for tier, (min_size, max_size) in self._SIZE_TIERS.items():
            if min_size <= model_size_b < max_size:
                return tier
        return "xl"
