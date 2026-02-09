"""Tests for LlamaCppOptions knowledge base - RED phase first."""

from __future__ import annotations

import pytest

from gpumod.discovery.llamacpp_options import (
    LlamaCppOptions,
    OptionSpec,
    RecommendedConfig,
    VRAMImpact,
)


class TestOptionSpec:
    """Tests for OptionSpec dataclass."""

    def test_option_spec_fields(self) -> None:
        """OptionSpec should have all required fields."""
        opt = OptionSpec(
            name="n_gpu_layers",
            type=int,
            default=-1,
            description="Number of layers to offload to GPU",
            vram_impact=VRAMImpact.HIGH,
            min_version=None,
            deprecated=False,
        )
        assert opt.name == "n_gpu_layers"
        assert opt.type is int
        assert opt.default == -1
        assert opt.vram_impact == VRAMImpact.HIGH


class TestRecommendedConfig:
    """Tests for RecommendedConfig dataclass."""

    def test_recommended_config_is_frozen(self) -> None:
        """RecommendedConfig should be immutable."""
        config = RecommendedConfig(
            n_gpu_layers=-1,
            ctx_size=8192,
            flash_attn=True,
            extra_args={},
            notes=("Full GPU offload",),
        )
        with pytest.raises(AttributeError):
            config.n_gpu_layers = 0  # type: ignore[misc]


class TestLlamaCppOptions:
    """Tests for LlamaCppOptions knowledge base."""

    @pytest.fixture
    def options(self) -> LlamaCppOptions:
        return LlamaCppOptions()

    def test_get_option_n_gpu_layers(self, options: LlamaCppOptions) -> None:
        """get_option('n_gpu_layers') should return OptionSpec."""
        opt = options.get_option("n_gpu_layers")
        assert opt is not None
        assert opt.name == "n_gpu_layers"
        assert opt.type is int
        assert opt.default == -1

    def test_get_option_ctx_size(self, options: LlamaCppOptions) -> None:
        """get_option('ctx_size') should return OptionSpec."""
        opt = options.get_option("ctx_size")
        assert opt is not None
        assert opt.name == "ctx_size"
        assert opt.type is int

    def test_get_option_flash_attn(self, options: LlamaCppOptions) -> None:
        """get_option('flash_attn') should return OptionSpec."""
        opt = options.get_option("flash_attn")
        assert opt is not None
        assert opt.type is bool
        assert opt.vram_impact == VRAMImpact.REDUCES

    def test_get_recommended_config_small_model(
        self,
        options: LlamaCppOptions,
    ) -> None:
        """get_recommended_config for 7B model should return sensible defaults."""
        config = options.get_recommended_config(model_size_b=7)
        assert config.n_gpu_layers == -1  # Full offload for small models
        assert config.ctx_size >= 4096

    def test_get_recommended_config_large_model(
        self,
        options: LlamaCppOptions,
    ) -> None:
        """get_recommended_config for 70B model should include partial offload."""
        config = options.get_recommended_config(
            model_size_b=70,
            available_vram_mb=24000,
        )
        # Large model needs partial offload
        assert config.n_gpu_layers != -1 or "partial" in str(config.notes).lower()

    def test_get_moe_config(self, options: LlamaCppOptions) -> None:
        """get_moe_config should include n_cpu_moe option."""
        config = options.get_moe_config(experts=8)
        assert "n_cpu_moe" in config.extra_args

    def test_estimate_vram_overhead_ctx_size(
        self,
        options: LlamaCppOptions,
    ) -> None:
        """estimate_vram_overhead should return KV cache estimate."""
        overhead = options.estimate_vram_overhead(ctx_size=8192)
        assert overhead > 0

    def test_validate_config_valid(self, options: LlamaCppOptions) -> None:
        """validate_config should pass for valid config."""
        config = RecommendedConfig(
            n_gpu_layers=-1,
            ctx_size=8192,
            flash_attn=True,
            extra_args={},
            notes=(),
        )
        errors = options.validate_config(config)
        assert len(errors) == 0

    def test_validate_config_invalid_ctx_size(
        self,
        options: LlamaCppOptions,
    ) -> None:
        """validate_config should catch invalid ctx_size."""
        config = RecommendedConfig(
            n_gpu_layers=-1,
            ctx_size=-1,  # Invalid
            flash_attn=True,
            extra_args={},
            notes=(),
        )
        errors = options.validate_config(config)
        assert len(errors) > 0
        assert any("ctx_size" in e for e in errors)

    def test_all_options_documented(self, options: LlamaCppOptions) -> None:
        """All common options should be documented."""
        common_options = [
            "n_gpu_layers",
            "ctx_size",
            "flash_attn",
            "batch_size",
            "threads",
            "jinja",
        ]
        for opt_name in common_options:
            opt = options.get_option(opt_name)
            assert opt is not None, f"Option {opt_name} not documented"
            assert opt.description, f"Option {opt_name} missing description"


class TestVRAMImpact:
    """Tests for VRAMImpact enum."""

    def test_vram_impact_values(self) -> None:
        """VRAMImpact should have expected values."""
        assert VRAMImpact.HIGH.value == "high"
        assert VRAMImpact.MEDIUM.value == "medium"
        assert VRAMImpact.LOW.value == "low"
        assert VRAMImpact.NONE.value == "none"
        assert VRAMImpact.REDUCES.value == "reduces"
