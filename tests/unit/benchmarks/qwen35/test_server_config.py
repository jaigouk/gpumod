"""Tests for llama.cpp server configuration.

Validates server flags match verified 2026 recommendations from:
- https://github.com/ggml-org/llama.cpp/issues/11200
- https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide
"""

from __future__ import annotations

import pytest

from gpumod.benchmarks.qwen35.server_config import (
    DEFAULT_16GB,
    DEFAULT_24GB,
    get_server_config,
)


class TestServerConfigValues:
    """Test that config values match verified recommendations."""

    def test_default_kv_cache_key_quantization(self) -> None:
        """KV cache key should use q8_0 for free performance."""
        assert DEFAULT_24GB.cache_type_k == "q8_0"

    def test_default_kv_cache_value_quantization(self) -> None:
        """KV cache value should use q8_0 for free performance."""
        assert DEFAULT_24GB.cache_type_v == "q8_0"

    def test_default_flash_attention(self) -> None:
        """Flash attention should be enabled by default."""
        assert DEFAULT_24GB.flash_attention is True

    def test_default_fit_mode(self) -> None:
        """Fit mode should be enabled for auto VRAM management."""
        assert DEFAULT_24GB.fit is True

    def test_default_no_mmap(self) -> None:
        """Memory mapping should be disabled."""
        assert DEFAULT_24GB.no_mmap is True

    def test_default_jinja(self) -> None:
        """Jinja templates should be enabled for Qwen."""
        assert DEFAULT_24GB.jinja is True

    def test_default_context_size(self) -> None:
        """Default context should be 32k minimum."""
        assert DEFAULT_24GB.context_size >= 32768


class TestServerConfig16GB:
    """Test 16GB VRAM preset."""

    def test_16gb_context_size(self) -> None:
        """16GB cards may need smaller context."""
        assert DEFAULT_16GB.context_size <= DEFAULT_24GB.context_size

    def test_16gb_has_kv_cache_quantization(self) -> None:
        """16GB cards should also use KV cache quantization."""
        assert DEFAULT_16GB.cache_type_k == "q8_0"
        assert DEFAULT_16GB.cache_type_v == "q8_0"


class TestServerConfigModel:
    """Test ServerConfig dataclass behavior."""

    def test_config_is_frozen(self) -> None:
        """Config should be immutable."""
        with pytest.raises(AttributeError):
            DEFAULT_24GB.context_size = 65536  # type: ignore[misc]

    def test_to_cli_args_returns_list(self) -> None:
        """to_cli_args should return list of strings."""
        args = DEFAULT_24GB.to_cli_args()
        assert isinstance(args, list)
        assert all(isinstance(arg, str) for arg in args)

    def test_to_cli_args_contains_kv_cache_flags(self) -> None:
        """CLI args should include KV cache quantization flags."""
        args = DEFAULT_24GB.to_cli_args()
        assert "-ctk" in args
        assert "q8_0" in args
        assert "-ctv" in args

    def test_to_cli_args_contains_flash_attention(self) -> None:
        """CLI args should include flash attention flag."""
        args = DEFAULT_24GB.to_cli_args()
        assert "-fa" in args

    def test_to_cli_args_contains_fit(self) -> None:
        """CLI args should include fit flag."""
        args = DEFAULT_24GB.to_cli_args()
        assert "--fit" in args

    def test_to_cli_args_contains_context(self) -> None:
        """CLI args should include context size."""
        args = DEFAULT_24GB.to_cli_args()
        assert "-c" in args

    def test_to_cli_args_contains_threads(self) -> None:
        """CLI args should include thread count."""
        args = DEFAULT_24GB.to_cli_args()
        assert "-t" in args


class TestGetServerConfig:
    """Test get_server_config factory function."""

    def test_get_24gb(self) -> None:
        config = get_server_config("24gb")
        assert config == DEFAULT_24GB

    def test_get_16gb(self) -> None:
        config = get_server_config("16gb")
        assert config == DEFAULT_16GB

    def test_get_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown config"):
            get_server_config("invalid")

    def test_case_insensitive(self) -> None:
        """Config names should be case-insensitive."""
        assert get_server_config("24GB") == DEFAULT_24GB
        assert get_server_config("16Gb") == DEFAULT_16GB
