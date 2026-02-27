"""Tests for prompt categories."""

from __future__ import annotations

from gpumod.benchmarks.qwen35.prompt_categories import (
    PromptCategory,
    generate_prompt,
)


class TestPromptCategory:
    """Test PromptCategory enum."""

    def test_short_exists(self) -> None:
        assert PromptCategory.SHORT is not None

    def test_medium_exists(self) -> None:
        assert PromptCategory.MEDIUM is not None

    def test_long_exists(self) -> None:
        assert PromptCategory.LONG is not None

    def test_multi_turn_exists(self) -> None:
        assert PromptCategory.MULTI_TURN is not None

    def test_short_target_tokens(self) -> None:
        """SHORT should target ~100 tokens."""
        assert PromptCategory.SHORT.target_tokens == 100

    def test_medium_target_tokens(self) -> None:
        """MEDIUM should target ~500 tokens."""
        assert PromptCategory.MEDIUM.target_tokens == 500

    def test_long_target_tokens(self) -> None:
        """LONG should target ~2000 tokens."""
        assert PromptCategory.LONG.target_tokens == 2000

    def test_multi_turn_target_tokens(self) -> None:
        """MULTI_TURN should target ~1000 tokens cumulative."""
        assert PromptCategory.MULTI_TURN.target_tokens == 1000


class TestGeneratePrompt:
    """Test prompt generation."""

    def test_generate_short_returns_string(self) -> None:
        prompt = generate_prompt(PromptCategory.SHORT)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_generate_medium_returns_string(self) -> None:
        prompt = generate_prompt(PromptCategory.MEDIUM)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_generate_long_returns_string(self) -> None:
        prompt = generate_prompt(PromptCategory.LONG)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_generate_multi_turn_returns_list(self) -> None:
        """MULTI_TURN should return list of conversation turns."""
        result = generate_prompt(PromptCategory.MULTI_TURN)
        assert isinstance(result, list)
        assert len(result) >= 2  # At least 2 turns

    def test_short_prompt_length_reasonable(self) -> None:
        """SHORT prompt should be roughly 100 tokens (~400 chars)."""
        prompt = generate_prompt(PromptCategory.SHORT)
        # Rough estimate: 1 token ≈ 4 chars
        assert 200 <= len(prompt) <= 800

    def test_medium_prompt_longer_than_short(self) -> None:
        """MEDIUM prompt should be longer than SHORT."""
        short = generate_prompt(PromptCategory.SHORT)
        medium = generate_prompt(PromptCategory.MEDIUM)
        assert len(medium) > len(short)

    def test_long_prompt_longer_than_medium(self) -> None:
        """LONG prompt should be longer than MEDIUM."""
        medium = generate_prompt(PromptCategory.MEDIUM)
        long_prompt = generate_prompt(PromptCategory.LONG)
        assert len(long_prompt) > len(medium)
