"""Tests for SectionFilter — fuzzy section matching with explicit failure.

TDD: RED phase - these tests define the new section filtering interface.

Root cause fix for gpumod-v7k:
1. Silent fallback returns full doc when section not found (violates fail-fast)
2. Exact match only - 'server' doesn't match 'Server options'
3. No size limits on output

SOLID principles applied:
- SRP: SectionFilter only does section extraction
- OCP: SectionMatcher protocol allows new matching strategies
- DIP: DriverDocsFetcher depends on abstractions (SectionMatcher protocol)
"""

from __future__ import annotations

import pytest

from gpumod.discovery.docs_fetcher import DriverDocs

# These imports will fail until we implement them (RED phase)
try:
    from gpumod.discovery.section_filter import (
        ExactMatcher,
        FuzzyMatcher,
        SectionFilter,
        SectionMatch,
        SectionNotFoundError,
    )
except ImportError:
    # Expected during RED phase - tests should fail with import error
    SectionFilter = None
    SectionNotFoundError = None
    FuzzyMatcher = None
    ExactMatcher = None
    SectionMatch = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_README = """\
# llama.cpp Server

## Usage

Run the server with:

```bash
llama-server -m model.gguf
```

## Server options

| Flag | Description |
|------|-------------|
| `--port` | Port to listen on |
| `--n-gpu-layers` | Number of GPU layers |
| `--flash-attn` | Enable flash attention |
| `--ctx-size` | Context size |

## Advanced

Expert offloading with `--n-cpu-moe`.

## Build

Instructions for building from source.
"""


@pytest.fixture
def sample_docs():
    """Create sample DriverDocs for testing."""
    return DriverDocs(
        driver="llamacpp",
        version="b7999",
        source_url="https://example.com/README.md",
        content=SAMPLE_README,
        sections=["llama.cpp Server", "Usage", "Server options", "Advanced", "Build"],
    )


@pytest.fixture
def section_filter():
    """Create SectionFilter with default FuzzyMatcher."""
    if SectionFilter is None:
        pytest.skip("SectionFilter not implemented yet (RED phase)")
    return SectionFilter()


@pytest.fixture
def fuzzy_matcher():
    """Create FuzzyMatcher for testing."""
    if FuzzyMatcher is None:
        pytest.skip("FuzzyMatcher not implemented yet (RED phase)")
    return FuzzyMatcher()


@pytest.fixture
def exact_matcher():
    """Create ExactMatcher for testing."""
    if ExactMatcher is None:
        pytest.skip("ExactMatcher not implemented yet (RED phase)")
    return ExactMatcher()


# ---------------------------------------------------------------------------
# SectionNotFoundError - explicit failure mode
# ---------------------------------------------------------------------------


class TestSectionNotFoundError:
    """Tests for explicit error when section not found."""

    def test_error_includes_available_sections(self, section_filter, sample_docs):
        """SectionNotFoundError includes list of available sections."""
        with pytest.raises(SectionNotFoundError) as exc_info:
            section_filter.filter(sample_docs, "Nonexistent Section")

        error = exc_info.value
        assert hasattr(error, "available_sections")
        assert "Server options" in error.available_sections
        assert "Usage" in error.available_sections

    def test_error_includes_best_match(self, section_filter, sample_docs):
        """SectionNotFoundError includes best fuzzy match with score."""
        # Use a query that's close but below threshold
        with pytest.raises(SectionNotFoundError) as exc_info:
            section_filter.filter(sample_docs, "configuration")  # Not close to any section

        error = exc_info.value
        assert hasattr(error, "best_match")
        # May or may not have a best match depending on threshold

    def test_error_includes_hint(self, section_filter, sample_docs):
        """SectionNotFoundError includes helpful hint."""
        with pytest.raises(SectionNotFoundError) as exc_info:
            section_filter.filter(sample_docs, "nonexistent xyz")

        error = exc_info.value
        assert hasattr(error, "hint")
        # Hint should mention available sections
        assert "Available sections" in error.hint or any(
            s in error.hint for s in sample_docs.sections
        )

    def test_error_message_is_descriptive(self, section_filter, sample_docs):
        """Error message clearly describes what went wrong."""
        with pytest.raises(SectionNotFoundError, match="Section 'xyz' not found"):
            section_filter.filter(sample_docs, "xyz")


# ---------------------------------------------------------------------------
# FuzzyMatcher - partial/substring matching
# ---------------------------------------------------------------------------


class TestFuzzyMatcher:
    """Tests for fuzzy section matching with Levenshtein distance."""

    def test_fuzzy_matches_partial_name(self, fuzzy_matcher):
        """'server' fuzzy-matches 'Server options'."""
        sections = ["Usage", "Server options", "Advanced", "Build"]

        match = fuzzy_matcher.match("server", sections)

        assert match is not None
        assert match.section == "Server options"
        assert match.score >= 0.6  # Reasonable threshold

    def test_fuzzy_matches_case_insensitive(self, fuzzy_matcher):
        """Matching is case-insensitive."""
        sections = ["Usage", "Server options", "Advanced"]

        match = fuzzy_matcher.match("SERVER OPTIONS", sections)

        assert match is not None
        assert match.section == "Server options"
        assert match.score >= 0.9  # High score for case-only difference

    def test_fuzzy_handles_typos(self, fuzzy_matcher):
        """Fuzzy matching handles common typos."""
        sections = ["Usage", "Server options", "Advanced"]

        match = fuzzy_matcher.match("Servr options", sections)  # Typo

        assert match is not None
        assert match.section == "Server options"
        assert match.score >= 0.8

    def test_fuzzy_returns_none_for_poor_match(self, fuzzy_matcher):
        """Returns None when no section matches above threshold."""
        sections = ["Usage", "Server options", "Advanced"]

        match = fuzzy_matcher.match("completely unrelated", sections)

        assert match is None

    def test_fuzzy_returns_best_match_among_multiple(self, fuzzy_matcher):
        """Returns highest-scoring match when multiple partial matches exist."""
        sections = ["Server options", "Server config", "Advanced server"]

        match = fuzzy_matcher.match("server options", sections)

        assert match is not None
        assert match.section == "Server options"  # Exact match wins

    def test_fuzzy_match_includes_score(self, fuzzy_matcher):
        """SectionMatch includes confidence score."""
        sections = ["Usage", "Server options"]

        match = fuzzy_matcher.match("server", sections)

        assert match is not None
        assert 0.0 <= match.score <= 1.0


# ---------------------------------------------------------------------------
# ExactMatcher - backward-compatible exact matching
# ---------------------------------------------------------------------------


class TestExactMatcher:
    """Tests for exact section matching (case-insensitive)."""

    def test_exact_matches_case_insensitive(self, exact_matcher):
        """Exact matching is case-insensitive."""
        sections = ["Usage", "Server options", "Advanced"]

        match = exact_matcher.match("server options", sections)

        assert match is not None
        assert match.section == "Server options"
        assert match.score == 1.0

    def test_exact_returns_none_for_partial(self, exact_matcher):
        """Exact matcher does not match partial names."""
        sections = ["Usage", "Server options", "Advanced"]

        match = exact_matcher.match("server", sections)

        assert match is None

    def test_exact_returns_none_for_typos(self, exact_matcher):
        """Exact matcher does not handle typos."""
        sections = ["Usage", "Server options", "Advanced"]

        match = exact_matcher.match("Servr options", sections)

        assert match is None


# ---------------------------------------------------------------------------
# SectionFilter - integration
# ---------------------------------------------------------------------------


class TestSectionFilter:
    """Tests for SectionFilter with fuzzy matching."""

    def test_filter_with_fuzzy_match(self, section_filter, sample_docs):
        """'server' fuzzy-matches 'Server options' and returns that section."""
        # With fuzzy=True (default), partial match should work
        result = section_filter.filter(sample_docs, "server", fuzzy=True)

        assert "Server options" in result.content
        assert "--port" in result.content
        assert "Expert offloading" not in result.content  # Not in Advanced section

    def test_filter_with_exact_match_enabled(self, sample_docs):
        """With fuzzy=False, only exact matches work."""
        if SectionFilter is None:
            pytest.skip("SectionFilter not implemented yet (RED phase)")

        filter_exact = SectionFilter(matcher=ExactMatcher())

        # Exact match should work
        result = filter_exact.filter(sample_docs, "Server options", fuzzy=False)
        assert "Server options" in result.content

        # Partial match should raise error
        with pytest.raises(SectionNotFoundError):
            filter_exact.filter(sample_docs, "server", fuzzy=False)

    def test_filter_returns_filtered_docs(self, section_filter, sample_docs):
        """Filtered result is a DriverDocs with filtered content."""
        result = section_filter.filter(sample_docs, "server", fuzzy=True)

        # Result should be DriverDocs with only the matched section
        assert result.sections == ["Server options"]
        assert "Server options" in result.content

    def test_filter_preserves_section_boundaries(self, section_filter, sample_docs):
        """Filtered content ends at next section header."""
        result = section_filter.filter(sample_docs, "Server options", fuzzy=True)

        # Should include Server options content
        assert "--port" in result.content
        assert "--flash-attn" in result.content

        # Should NOT include Advanced section
        assert "Expert offloading" not in result.content

        # Should NOT include Build section
        assert "building from source" not in result.content


# ---------------------------------------------------------------------------
# SectionMatch dataclass
# ---------------------------------------------------------------------------


class TestSectionMatch:
    """Tests for SectionMatch dataclass."""

    def test_section_match_fields(self):
        """SectionMatch has required fields."""
        if SectionMatch is None:
            pytest.skip("SectionMatch not implemented yet (RED phase)")

        match = SectionMatch(section="Server options", score=0.85)

        assert match.section == "Server options"
        assert match.score == 0.85

    def test_section_match_comparable(self):
        """SectionMatch instances can be compared by score."""
        if SectionMatch is None:
            pytest.skip("SectionMatch not implemented yet (RED phase)")

        match1 = SectionMatch(section="Server options", score=0.85)
        match2 = SectionMatch(section="Usage", score=0.60)

        # Higher score should be "greater"
        assert match1 > match2


# ---------------------------------------------------------------------------
# Integration with DriverDocsFetcher
# ---------------------------------------------------------------------------


class TestDriverDocsFetcherIntegration:
    """Tests for DriverDocsFetcher using new SectionFilter."""

    async def test_fetch_with_fuzzy_section(self):
        """fetch_driver_docs uses fuzzy matching by default."""
        pytest.skip("Integration test - implement after GREEN phase")

    async def test_fetch_section_not_found_returns_error_dict(self):
        """MCP tool returns error dict with available_sections on mismatch."""
        pytest.skip("Integration test - implement after GREEN phase")
