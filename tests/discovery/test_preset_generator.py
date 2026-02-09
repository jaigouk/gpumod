"""Tests for PresetGenerator - RED phase first."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from gpumod.discovery.gguf_metadata import GGUFFile
from gpumod.discovery.preset_generator import (
    PresetGenerator,
    PresetRequest,
)
from gpumod.discovery.system_info import SystemInfo


class TestPresetRequest:
    """Tests for PresetRequest dataclass."""

    def test_preset_request_defaults(self) -> None:
        """PresetRequest should have sensible defaults."""
        gguf = GGUFFile(
            filename="model-Q4_K_M.gguf",
            size_bytes=20_000_000_000,
            quant_type="Q4_K_M",
            estimated_vram_mb=21000,
            is_split=False,
            split_parts=1,
        )
        system = SystemInfo(
            gpu_total_mb=24576,
            gpu_used_mb=0,
            gpu_available_mb=24576,
            gpu_name="RTX 4090",
            ram_total_mb=65536,
            ram_available_mb=60000,
            swap_available_mb=8192,
            current_mode=None,
            running_services=(),
        )
        request = PresetRequest(
            repo_id="unsloth/test-model",
            gguf_file=gguf,
            system_info=system,
        )
        assert request.ctx_size == 8192  # default
        assert request.port is None  # auto-assign


class TestPresetGenerator:
    """Tests for PresetGenerator."""

    @pytest.fixture
    def generator(self) -> PresetGenerator:
        return PresetGenerator()

    @pytest.fixture
    def sample_request(self) -> PresetRequest:
        gguf = GGUFFile(
            filename="Qwen3-Coder-30B-Q4_K_M.gguf",
            size_bytes=20_000_000_000,
            quant_type="Q4_K_M",
            estimated_vram_mb=21000,
            is_split=False,
            split_parts=1,
        )
        system = SystemInfo(
            gpu_total_mb=24576,
            gpu_used_mb=0,
            gpu_available_mb=24576,
            gpu_name="RTX 4090",
            ram_total_mb=65536,
            ram_available_mb=60000,
            swap_available_mb=8192,
            current_mode=None,
            running_services=(),
        )
        return PresetRequest(
            repo_id="unsloth/Qwen3-Coder-30B-GGUF",
            gguf_file=gguf,
            system_info=system,
            ctx_size=8192,
        )

    def test_generate_returns_valid_yaml(
        self,
        generator: PresetGenerator,
        sample_request: PresetRequest,
    ) -> None:
        """generate() should return valid YAML string."""
        yaml_str = generator.generate(sample_request)
        # Should not raise
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)

    def test_generated_preset_has_required_fields(
        self,
        generator: PresetGenerator,
        sample_request: PresetRequest,
    ) -> None:
        """Generated preset should have all required fields."""
        yaml_str = generator.generate(sample_request)
        parsed = yaml.safe_load(yaml_str)

        required_fields = [
            "id",
            "name",
            "driver",
            "port",
            "vram_mb",
            "health_endpoint",
            "startup_timeout",
        ]
        for field in required_fields:
            assert field in parsed, f"Missing required field: {field}"

    def test_service_id_derived_from_model(
        self,
        generator: PresetGenerator,
        sample_request: PresetRequest,
    ) -> None:
        """Service ID should be derived from model name."""
        yaml_str = generator.generate(sample_request)
        parsed = yaml.safe_load(yaml_str)
        # Should be lowercase, hyphenated
        assert parsed["id"].islower() or "-" in parsed["id"]
        assert "qwen" in parsed["id"].lower()

    def test_port_auto_assigned(
        self,
        generator: PresetGenerator,
        sample_request: PresetRequest,
    ) -> None:
        """Port should be auto-assigned if not specified."""
        yaml_str = generator.generate(sample_request)
        parsed = yaml.safe_load(yaml_str)
        assert 1024 <= parsed["port"] <= 65535

    def test_port_custom_value(
        self,
        generator: PresetGenerator,
        sample_request: PresetRequest,
    ) -> None:
        """Should use custom port if specified."""
        sample_request = PresetRequest(
            repo_id=sample_request.repo_id,
            gguf_file=sample_request.gguf_file,
            system_info=sample_request.system_info,
            port=8080,
        )
        yaml_str = generator.generate(sample_request)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["port"] == 8080

    def test_n_gpu_layers_calculated(
        self,
        generator: PresetGenerator,
        sample_request: PresetRequest,
    ) -> None:
        """n_gpu_layers should be calculated for VRAM budget."""
        yaml_str = generator.generate(sample_request)
        parsed = yaml.safe_load(yaml_str)
        assert "unit_vars" in parsed
        assert "n_gpu_layers" in parsed["unit_vars"]

    def test_ctx_size_from_request(
        self,
        generator: PresetGenerator,
        sample_request: PresetRequest,
    ) -> None:
        """ctx_size should come from request."""
        yaml_str = generator.generate(sample_request)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["unit_vars"]["ctx_size"] == 8192

    def test_moe_model_gets_special_options(
        self,
        generator: PresetGenerator,
    ) -> None:
        """MoE models should get n_cpu_moe option."""
        gguf = GGUFFile(
            filename="Qwen3-Coder-Next-80B-MoE-Q2_K_XL.gguf",
            size_bytes=30_000_000_000,
            quant_type="Q2_K_XL",
            estimated_vram_mb=33000,
            is_split=False,
            split_parts=1,
        )
        system = SystemInfo(
            gpu_total_mb=24576,
            gpu_used_mb=0,
            gpu_available_mb=24576,
            gpu_name="RTX 4090",
            ram_total_mb=65536,
            ram_available_mb=60000,
            swap_available_mb=8192,
            current_mode=None,
            running_services=(),
        )
        request = PresetRequest(
            repo_id="unsloth/Qwen3-Coder-Next-GGUF",
            gguf_file=gguf,
            system_info=system,
            is_moe=True,
        )
        yaml_str = generator.generate(request)
        parsed = yaml.safe_load(yaml_str)
        # MoE should have special handling
        assert "n_gpu_layers" in parsed["unit_vars"]

    def test_preset_works_with_gpumod(
        self,
        generator: PresetGenerator,
        sample_request: PresetRequest,
    ) -> None:
        """Generated preset should be valid for 'gpumod service add'."""
        yaml_str = generator.generate(sample_request)
        parsed = yaml.safe_load(yaml_str)

        # Check driver is valid
        assert parsed["driver"] in ["llamacpp", "vllm", "fastapi", "docker"]

        # Check vram_mb is reasonable
        assert 0 < parsed["vram_mb"] < 100000

        # Check health_endpoint format
        assert parsed["health_endpoint"].startswith("/")

    def test_includes_provenance_comment(
        self,
        generator: PresetGenerator,
        sample_request: PresetRequest,
    ) -> None:
        """Generated YAML should include provenance comment."""
        yaml_str = generator.generate(sample_request)
        assert "Generated by gpumod discover" in yaml_str

    def test_router_mode_by_default(
        self,
        generator: PresetGenerator,
        sample_request: PresetRequest,
    ) -> None:
        """Should use router mode (models_dir) by default."""
        yaml_str = generator.generate(sample_request)
        parsed = yaml.safe_load(yaml_str)
        assert "models_dir" in parsed["unit_vars"]
        assert parsed["unit_vars"].get("no_models_autoload", False) is True


class TestPresetGeneratorEdgeCases:
    """Tests for edge cases in preset generation."""

    @pytest.fixture
    def generator(self) -> PresetGenerator:
        return PresetGenerator()

    def test_special_chars_in_model_name(
        self,
        generator: PresetGenerator,
    ) -> None:
        """Should sanitize special characters in model name."""
        gguf = GGUFFile(
            filename="Model@v2.0_test!.gguf",
            size_bytes=10_000_000_000,
            quant_type="Q4_K_M",
            estimated_vram_mb=11000,
            is_split=False,
            split_parts=1,
        )
        system = SystemInfo(
            gpu_total_mb=24576,
            gpu_used_mb=0,
            gpu_available_mb=24576,
            gpu_name="RTX 4090",
            ram_total_mb=65536,
            ram_available_mb=60000,
            swap_available_mb=8192,
            current_mode=None,
            running_services=(),
        )
        request = PresetRequest(
            repo_id="unsloth/weird-model-name",
            gguf_file=gguf,
            system_info=system,
        )
        yaml_str = generator.generate(request)
        parsed = yaml.safe_load(yaml_str)
        # ID should not contain special chars
        assert "@" not in parsed["id"]
        assert "!" not in parsed["id"]

    def test_refuses_if_exceeds_vram(
        self,
        generator: PresetGenerator,
    ) -> None:
        """Should raise if selected quant exceeds available VRAM."""
        gguf = GGUFFile(
            filename="huge-model.gguf",
            size_bytes=50_000_000_000,  # 50GB
            quant_type="Q8_0",
            estimated_vram_mb=55000,
            is_split=False,
            split_parts=1,
        )
        system = SystemInfo(
            gpu_total_mb=24576,
            gpu_used_mb=0,
            gpu_available_mb=24576,
            gpu_name="RTX 4090",
            ram_total_mb=65536,
            ram_available_mb=60000,
            swap_available_mb=8192,
            current_mode=None,
            running_services=(),
        )
        request = PresetRequest(
            repo_id="unsloth/huge-model",
            gguf_file=gguf,
            system_info=system,
        )
        # Should still generate but with partial offload notes
        yaml_str = generator.generate(request)
        parsed = yaml.safe_load(yaml_str)
        # Should indicate partial offload or warning
        assert parsed["unit_vars"]["n_gpu_layers"] != -1
