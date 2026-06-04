"""Tests for Qwen3.5 sampler configuration.

Validates that preset values match Qwen official recommendations from:
https://huggingface.co/Qwen/Qwen3.5-35B-A3B
"""

from __future__ import annotations

import pytest

from gpumod.benchmarks.coding.sampler_config import (
    GEMMA_CODING,
    NON_THINKING,
    THINKING_CODING,
    get_config,
)


class TestSamplerConfigValues:
    """Test that config values match Qwen recommendations."""

    def test_thinking_coding_temperature(self) -> None:
        assert THINKING_CODING.temperature == 0.6

    def test_thinking_coding_top_p(self) -> None:
        assert THINKING_CODING.top_p == 0.95

    def test_thinking_coding_top_k(self) -> None:
        assert THINKING_CODING.top_k == 20

    def test_thinking_coding_min_p(self) -> None:
        assert THINKING_CODING.min_p == 0.0

    def test_thinking_coding_presence_penalty(self) -> None:
        assert THINKING_CODING.presence_penalty == 0.0

    def test_thinking_coding_repetition_penalty(self) -> None:
        assert THINKING_CODING.repetition_penalty == 1.0

    def test_non_thinking_temperature(self) -> None:
        assert NON_THINKING.temperature == 0.7

    def test_non_thinking_top_p(self) -> None:
        assert NON_THINKING.top_p == 0.8

    def test_non_thinking_top_k(self) -> None:
        assert NON_THINKING.top_k == 20

    def test_non_thinking_min_p(self) -> None:
        assert NON_THINKING.min_p == 0.0

    def test_non_thinking_presence_penalty(self) -> None:
        assert NON_THINKING.presence_penalty == 1.5

    def test_non_thinking_repetition_penalty(self) -> None:
        assert NON_THINKING.repetition_penalty == 1.0

    # gpumod-h6gs: GEMMA_CODING matches Google's Gemma 4 recommendation
    # (https://huggingface.co/google/gemma-4-12B-it). The same sampler is
    # recommended for instruction-following, coding, and reasoning — unlike
    # Qwen, which differentiates between thinking-coding and non-thinking.
    def test_gemma_coding_temperature(self) -> None:
        assert GEMMA_CODING.temperature == 1.0

    def test_gemma_coding_top_p(self) -> None:
        assert GEMMA_CODING.top_p == 0.95

    def test_gemma_coding_top_k(self) -> None:
        assert GEMMA_CODING.top_k == 64

    def test_gemma_coding_min_p(self) -> None:
        assert GEMMA_CODING.min_p == 0.0

    def test_gemma_coding_presence_penalty(self) -> None:
        assert GEMMA_CODING.presence_penalty == 0.0

    def test_gemma_coding_repetition_penalty(self) -> None:
        # gpumod-eods: 1.05 (not the Google default 1.0) to break degeneration
        # loops at temp=1.0 — see iter_06 in gpumod-h6gs run 2 where the model
        # got stuck repeating "If __ename__ import requests" until the response
        # budget was exhausted. 1.05 is the lowest value that breaks the loop
        # without measurably suppressing legitimate code repetition.
        assert GEMMA_CODING.repetition_penalty == 1.05


class TestSamplerConfigModel:
    """Test SamplerConfig dataclass behavior."""

    def test_config_is_frozen(self) -> None:
        """Config should be immutable."""
        with pytest.raises(AttributeError):
            THINKING_CODING.temperature = 0.9  # type: ignore[misc]

    def test_config_to_dict(self) -> None:
        """Config should convert to dict for API calls."""
        d = THINKING_CODING.to_dict()
        assert d["temperature"] == 0.6
        assert d["top_p"] == 0.95
        assert d["top_k"] == 20

    def test_config_to_dict_excludes_defaults(self) -> None:
        """to_dict with exclude_defaults skips repetition_penalty=1.0."""
        d = THINKING_CODING.to_dict(exclude_defaults=True)
        assert "repetition_penalty" not in d


class TestGetConfig:
    """Test get_config factory function."""

    def test_get_thinking_coding(self) -> None:
        config = get_config("thinking_coding")
        assert config == THINKING_CODING

    def test_get_non_thinking(self) -> None:
        config = get_config("non_thinking")
        assert config == NON_THINKING

    def test_get_gemma_coding(self) -> None:
        config = get_config("gemma_coding")
        assert config == GEMMA_CODING

    def test_get_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown config"):
            get_config("invalid")
