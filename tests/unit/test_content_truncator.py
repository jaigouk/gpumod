"""Tests for ContentTruncator — token-aware content truncation.

TDD: RED phase - these tests define the ContentTruncator interface.

Root cause fix for gpumod-v7k:
- No size limits on output (88KB response exceeds MCP token limits)

SOLID principles applied:
- SRP: ContentTruncator only handles truncation
- OCP: Different truncation strategies possible (token-based, char-based)
- DIP: DriverDocsFetcher depends on Truncator protocol
"""

from __future__ import annotations

import pytest

# These imports will fail until we implement them (RED phase)
try:
    from gpumod.discovery.content_truncator import (
        CharTruncator,
        ContentTruncator,
        TokenTruncator,
        TruncationResult,
    )
except ImportError:
    # Expected during RED phase
    ContentTruncator = None
    TokenTruncator = None
    CharTruncator = None
    TruncationResult = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# ~200 tokens worth of content
SHORT_CONTENT = """\
# Server Options

This section describes the server command-line options.

## Basic Options

| Flag | Description |
|------|-------------|
| `--port` | Port to listen on (default: 8080) |
| `--host` | Host to bind to (default: localhost) |
"""

# Large content that would exceed typical token limits
LARGE_CONTENT = SHORT_CONTENT * 100  # ~20,000 tokens


@pytest.fixture
def token_truncator():
    """Create TokenTruncator with 1000 token limit."""
    if TokenTruncator is None:
        pytest.skip("TokenTruncator not implemented yet (RED phase)")
    return TokenTruncator(max_tokens=1000)


@pytest.fixture
def char_truncator():
    """Create CharTruncator with 5000 char limit."""
    if CharTruncator is None:
        pytest.skip("CharTruncator not implemented yet (RED phase)")
    return CharTruncator(max_chars=5000)


# ---------------------------------------------------------------------------
# TruncationResult dataclass
# ---------------------------------------------------------------------------


class TestTruncationResult:
    """Tests for TruncationResult dataclass."""

    def test_result_fields(self):
        """TruncationResult has all required fields."""
        if TruncationResult is None:
            pytest.skip("TruncationResult not implemented yet (RED phase)")

        result = TruncationResult(
            content="truncated content...",
            truncated=True,
            original_length=10000,
            result_length=1000,
            unit="tokens",
        )

        assert result.content == "truncated content..."
        assert result.truncated is True
        assert result.original_length == 10000
        assert result.result_length == 1000
        assert result.unit == "tokens"

    def test_result_not_truncated(self):
        """TruncationResult for content that fits."""
        if TruncationResult is None:
            pytest.skip("TruncationResult not implemented yet (RED phase)")

        result = TruncationResult(
            content="short content",
            truncated=False,
            original_length=100,
            result_length=100,
            unit="tokens",
        )

        assert result.truncated is False
        assert result.original_length == result.result_length


# ---------------------------------------------------------------------------
# TokenTruncator
# ---------------------------------------------------------------------------


class TestTokenTruncator:
    """Tests for token-aware truncation."""

    def test_truncator_respects_token_limit(self, token_truncator):
        """Output does not exceed max_tokens."""
        result = token_truncator.truncate(LARGE_CONTENT)

        assert result.truncated is True
        assert result.result_length <= 1000

    def test_truncator_no_truncation_for_short_content(self, token_truncator):
        """Short content is not truncated."""
        result = token_truncator.truncate(SHORT_CONTENT)

        assert result.truncated is False
        assert result.content == SHORT_CONTENT

    def test_truncator_adds_indicator(self, token_truncator):
        """Truncated output includes indicator."""
        result = token_truncator.truncate(LARGE_CONTENT)

        assert result.truncated is True
        assert "[...truncated" in result.content or "truncated" in result.content.lower()

    def test_truncator_indicator_shows_counts(self, token_truncator):
        """Truncation indicator shows token counts."""
        result = token_truncator.truncate(LARGE_CONTENT)

        # Should show something like "[...truncated, showing 1000 of 20000 tokens]"
        assert result.truncated is True
        indicator = result.content.split("[")[-1] if "[" in result.content else ""
        # Indicator should mention token counts
        assert "token" in indicator.lower() or str(result.result_length) in result.content

    def test_truncator_preserves_section_start(self, token_truncator):
        """Truncation preserves the beginning of content (most important)."""
        result = token_truncator.truncate(LARGE_CONTENT)

        # First line should be preserved
        assert "# Server Options" in result.content

    def test_truncator_breaks_at_paragraph_boundary(self):
        """Truncation happens at paragraph/section boundary when possible."""
        if TokenTruncator is None:
            pytest.skip("TokenTruncator not implemented yet (RED phase)")

        # Create truncator with limit that would cut mid-table
        truncator = TokenTruncator(max_tokens=50)
        result = truncator.truncate(SHORT_CONTENT)

        # Should not cut in the middle of a word or table row
        if result.truncated:
            assert not result.content.endswith("|")  # Don't cut mid-table
            # Content should end cleanly (newline or indicator)

    def test_truncator_configurable_limit(self):
        """Token limit is configurable."""
        if TokenTruncator is None:
            pytest.skip("TokenTruncator not implemented yet (RED phase)")

        truncator_small = TokenTruncator(max_tokens=100)
        truncator_large = TokenTruncator(max_tokens=50000)

        result_small = truncator_small.truncate(LARGE_CONTENT)
        result_large = truncator_large.truncate(LARGE_CONTENT)

        assert result_small.result_length < result_large.result_length

    def test_truncator_default_limit(self):
        """Default limit is 20000 tokens (MCP best practice)."""
        if TokenTruncator is None:
            pytest.skip("TokenTruncator not implemented yet (RED phase)")

        truncator = TokenTruncator()
        assert truncator.max_tokens == 20000


# ---------------------------------------------------------------------------
# CharTruncator (simpler fallback)
# ---------------------------------------------------------------------------


class TestCharTruncator:
    """Tests for character-based truncation (fallback when tiktoken unavailable)."""

    def test_char_truncator_respects_limit(self, char_truncator):
        """Output does not exceed max_chars."""
        result = char_truncator.truncate(LARGE_CONTENT)

        assert result.truncated is True
        assert len(result.content) <= 5000 + 100  # Allow for indicator

    def test_char_truncator_unit_is_chars(self, char_truncator):
        """CharTruncator reports unit as 'chars'."""
        result = char_truncator.truncate(LARGE_CONTENT)

        assert result.unit == "chars"


# ---------------------------------------------------------------------------
# Integration with DriverDocsFetcher
# ---------------------------------------------------------------------------


class TestTruncatorIntegration:
    """Integration tests for truncator with DriverDocsFetcher."""

    async def test_fetch_includes_truncation_metadata(self):
        """fetch_driver_docs response includes truncation metadata."""
        pytest.skip("Integration test - implement after GREEN phase")

    async def test_fetch_with_max_tokens_parameter(self):
        """fetch_driver_docs accepts max_tokens parameter."""
        pytest.skip("Integration test - implement after GREEN phase")

    async def test_fetch_default_truncation_enabled(self):
        """Truncation is enabled by default with 20000 token limit."""
        pytest.skip("Integration test - implement after GREEN phase")
