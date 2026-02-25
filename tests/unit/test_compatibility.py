"""Tests for llama.cpp architecture compatibility checker.

TDD tests for gpumod-h1c: Architecture compatibility check before model operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


class TestGetLlamacppVersion:
    """Tests for CompatibilityChecker.get_llamacpp_version() - AC1."""

    async def test_parses_version_from_llama_server_output(self) -> None:
        """Parse build number from 'llama-server --version' output."""
        from gpumod.compatibility import CompatibilityChecker

        # Mock subprocess to return typical llama-server output
        checker = CompatibilityChecker()
        checker._mock_version_output = "version: 7801 (abc123def)"

        version = await checker.get_llamacpp_version()

        assert version == 7801

    async def test_caches_version_on_subsequent_calls(self) -> None:
        """Returns cached value on subsequent calls."""
        from gpumod.compatibility import CompatibilityChecker

        checker = CompatibilityChecker()
        checker._mock_version_output = "version: 8000 (xyz789)"

        # First call
        version1 = await checker.get_llamacpp_version()
        # Change mock - should not affect cached result
        checker._mock_version_output = "version: 9999 (changed)"
        version2 = await checker.get_llamacpp_version()

        assert version1 == 8000
        assert version2 == 8000  # Still cached

    async def test_returns_none_when_llama_server_unavailable(self) -> None:
        """Returns None when llama-server is not installed."""
        from gpumod.compatibility import CompatibilityChecker

        checker = CompatibilityChecker()
        checker._mock_version_output = None  # Simulate unavailable

        version = await checker.get_llamacpp_version()

        assert version is None

    async def test_returns_none_when_output_unparseable(self) -> None:
        """Returns None when version output cannot be parsed."""
        from gpumod.compatibility import CompatibilityChecker

        checker = CompatibilityChecker()
        checker._mock_version_output = "unexpected output format"

        version = await checker.get_llamacpp_version()

        assert version is None

    async def test_logs_warning_on_version_detection_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Logs warning when version detection fails."""
        import logging

        from gpumod.compatibility import CompatibilityChecker

        checker = CompatibilityChecker()
        checker._mock_version_output = None

        with caplog.at_level(logging.WARNING):
            await checker.get_llamacpp_version()

        assert any("version" in record.message.lower() for record in caplog.records)


class TestArchitectureMatrix:
    """Tests for architecture matrix loading - AC2."""

    def test_loads_matrix_from_package_data(self) -> None:
        """Matrix is loaded from arch_support.yaml at instantiation."""
        from gpumod.compatibility import CompatibilityChecker

        checker = CompatibilityChecker()

        # Should have loaded architecture matrix
        assert hasattr(checker, "_matrix")
        assert len(checker._matrix) > 0

    def test_matrix_contains_known_architectures(self) -> None:
        """Matrix contains expected architectures with min_version."""
        from gpumod.compatibility import CompatibilityChecker

        checker = CompatibilityChecker()

        # Key architectures from design doc
        assert "llama" in checker._matrix
        assert "qwen2" in checker._matrix
        assert "qwen35moe" in checker._matrix

        # Each entry has min_version
        assert checker._matrix["qwen35moe"].min_version >= 7900

    def test_matrix_entry_has_required_fields(self) -> None:
        """Each matrix entry has min_version and optional notes."""
        from gpumod.compatibility import ArchSupport, CompatibilityChecker

        checker = CompatibilityChecker()

        for support in checker._matrix.values():
            assert isinstance(support, ArchSupport)
            assert isinstance(support.min_version, int)
            assert support.min_version > 0
            # notes is optional (str | None)

    def test_matrix_extensible_via_yaml(self, tmp_path: pytest.TempPathFactory) -> None:
        """Matrix can be loaded from custom path (extensibility)."""
        import yaml

        from gpumod.compatibility import CompatibilityChecker

        # Create custom matrix
        custom_matrix = tmp_path / "custom_arch.yaml"
        custom_matrix.write_text(
            yaml.dump(
                {
                    "architectures": {
                        "custom_arch": {"min_version": 9999, "notes": "Custom test"},
                    }
                }
            )
        )

        checker = CompatibilityChecker(matrix_path=custom_matrix)

        assert "custom_arch" in checker._matrix
        assert checker._matrix["custom_arch"].min_version == 9999


class TestIsSupported:
    """Tests for CompatibilityChecker.is_supported() - AC3."""

    async def test_returns_true_when_architecture_supported(self) -> None:
        """Returns (True, None) when architecture supported by current version."""
        from gpumod.compatibility import CompatibilityChecker

        checker = CompatibilityChecker()
        checker._mock_version_output = "version: 8000 (abc123)"

        # qwen35moe requires 7900, we have 8000
        supported, reason = await checker.is_supported("qwen35moe")

        assert supported is True
        assert reason is None

    async def test_returns_false_when_architecture_unsupported(self) -> None:
        """Returns (False, reason) when architecture unsupported."""
        from gpumod.compatibility import CompatibilityChecker

        checker = CompatibilityChecker()
        checker._mock_version_output = "version: 7801 (abc123)"

        # qwen35moe requires 7900, we have 7801
        supported, reason = await checker.is_supported("qwen35moe")

        assert supported is False
        assert reason is not None
        assert "7900" in reason  # Required version
        assert "7801" in reason  # Current version

    async def test_reason_format_includes_versions(self) -> None:
        """Error message includes current and required versions."""
        from gpumod.compatibility import CompatibilityChecker

        checker = CompatibilityChecker()
        checker._mock_version_output = "version: 5000 (abc123)"

        supported, reason = await checker.is_supported("qwen35moe")

        assert supported is False
        # Should be human-readable: "qwen35moe requires llama.cpp b7900+ (you have b5000)"
        assert "qwen35moe" in reason
        assert "7900" in reason
        assert "5000" in reason

    async def test_unknown_architecture_returns_true_with_warning(self) -> None:
        """Unknown architectures fail-open with warning message."""
        from gpumod.compatibility import CompatibilityChecker

        checker = CompatibilityChecker()
        checker._mock_version_output = "version: 8000 (abc123)"

        supported, reason = await checker.is_supported("unknown_future_arch")

        assert supported is True
        assert reason is not None
        assert "unknown" in reason.lower()

    async def test_version_detection_failure_returns_true(self) -> None:
        """Fail-open when version detection fails."""
        from gpumod.compatibility import CompatibilityChecker

        checker = CompatibilityChecker()
        checker._mock_version_output = None  # Simulate unavailable

        supported, reason = await checker.is_supported("qwen35moe")

        assert supported is True  # Fail-open
        assert reason is not None  # Warning message

    async def test_uses_provided_version_instead_of_cached(self) -> None:
        """Uses provided version if given, not cached version."""
        from gpumod.compatibility import CompatibilityChecker

        checker = CompatibilityChecker()
        checker._mock_version_output = "version: 8000 (abc123)"

        # Override with explicit lower version
        supported, reason = await checker.is_supported("qwen35moe", version=5000)

        assert supported is False
        assert "5000" in reason

    async def test_uses_cached_version_when_not_provided(self) -> None:
        """Uses cached version when no explicit version given."""
        from gpumod.compatibility import CompatibilityChecker

        checker = CompatibilityChecker()
        checker._mock_version_output = "version: 8000 (abc123)"

        # First call caches version
        await checker.get_llamacpp_version()
        # Change mock - should still use cached
        checker._mock_version_output = "version: 1000 (old)"

        supported, _reason = await checker.is_supported("qwen35moe")

        assert supported is True  # Uses cached 8000, not mocked 1000


class TestHfToLlamacppArchMapping:
    """Tests for HuggingFace model class to llama.cpp architecture mapping."""

    def test_mapping_contains_known_architectures(self) -> None:
        """Mapping includes common HF model classes."""
        from gpumod.compatibility import HF_TO_LLAMACPP_ARCH

        # Key mappings from design doc
        assert HF_TO_LLAMACPP_ARCH["LlamaForCausalLM"] == "llama"
        assert HF_TO_LLAMACPP_ARCH["Qwen2ForCausalLM"] == "qwen2"
        assert HF_TO_LLAMACPP_ARCH["Qwen2MoeForCausalLM"] == "qwen2moe"
        # Qwen 3.5 uses Qwen3MoeForCausalLM
        assert "Qwen3MoeForCausalLM" in HF_TO_LLAMACPP_ARCH

    def test_get_arch_from_hf_class_returns_arch(self) -> None:
        """get_arch_from_hf_class returns llama.cpp architecture string."""
        from gpumod.compatibility import get_arch_from_hf_class

        assert get_arch_from_hf_class("LlamaForCausalLM") == "llama"
        assert get_arch_from_hf_class("Qwen2MoeForCausalLM") == "qwen2moe"

    def test_get_arch_from_hf_class_returns_none_for_unknown(self) -> None:
        """get_arch_from_hf_class returns None for unknown classes."""
        from gpumod.compatibility import get_arch_from_hf_class

        assert get_arch_from_hf_class("UnknownModel") is None
        assert get_arch_from_hf_class("") is None


class TestMcpIntegration:
    """Tests for MCP tool integration - AC4.

    These tests verify the compatibility checker integrates with MCP tools.
    """

    async def test_generate_preset_includes_compatibility_warning(self) -> None:
        """generate_preset includes compatibility_warning when unsupported."""
        from unittest.mock import AsyncMock, patch

        from gpumod.mcp_tools import generate_preset

        # Mock context
        ctx = AsyncMock()
        ctx.info = AsyncMock()

        # Mock compatibility checker to return unsupported
        with patch("gpumod.mcp_tools.get_compatibility_checker") as mock_get:
            checker = AsyncMock()
            checker.is_supported = AsyncMock(
                return_value=(False, "qwen35moe requires llama.cpp b7900+ (you have b7801)")
            )
            mock_get.return_value = checker

            # Also need to mock fetch_model_config to return architecture
            with patch("gpumod.mcp_tools.fetch_model_config") as mock_config:
                mock_config.return_value = {
                    "architectures": ["Qwen3MoeForCausalLM"],
                    "context_length": 32768,
                }

                result = await generate_preset(
                    repo_id="unsloth/Qwen3.5-35B-A3B-GGUF",
                    gguf_file="model-Q4_K_XL.gguf",
                    ctx=ctx,
                )

        # Should still return preset (warn, not block)
        assert "preset" in result
        # Should include compatibility warning
        assert "compatibility_warning" in result
        assert "7900" in result["compatibility_warning"]

    async def test_generate_preset_no_warning_when_supported(self) -> None:
        """generate_preset has no compatibility_warning when supported."""
        from unittest.mock import AsyncMock, patch

        from gpumod.mcp_tools import generate_preset

        ctx = AsyncMock()
        ctx.info = AsyncMock()

        with patch("gpumod.mcp_tools.get_compatibility_checker") as mock_get:
            checker = AsyncMock()
            checker.is_supported = AsyncMock(return_value=(True, None))
            mock_get.return_value = checker

            with patch("gpumod.mcp_tools.fetch_model_config") as mock_config:
                mock_config.return_value = {
                    "architectures": ["LlamaForCausalLM"],
                    "context_length": 32768,
                }

                result = await generate_preset(
                    repo_id="meta-llama/Llama-3-GGUF",
                    gguf_file="model.gguf",
                    ctx=ctx,
                )

        assert "preset" in result
        assert "compatibility_warning" not in result

    async def test_generate_preset_no_warning_when_checker_unavailable(self) -> None:
        """generate_preset works without compatibility checker (graceful degradation)."""
        from unittest.mock import AsyncMock, patch

        from gpumod.mcp_tools import generate_preset

        ctx = AsyncMock()
        ctx.info = AsyncMock()

        # No checker available
        with patch("gpumod.mcp_tools.get_compatibility_checker", return_value=None):
            result = await generate_preset(
                repo_id="unsloth/test-GGUF",
                gguf_file="model.gguf",
                ctx=ctx,
            )

        assert "preset" in result
        # Should work without warning (no checker = no warning)
        assert "compatibility_warning" not in result
