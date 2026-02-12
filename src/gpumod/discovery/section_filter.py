"""Section filtering with fuzzy matching and explicit failure mode.

Implements SRP by separating section filtering from doc fetching.
Implements OCP via SectionMatcher protocol for swappable match strategies.

Part of fix for gpumod-v7k: silent fallback on section not found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

from gpumod.discovery.docs_fetcher import DriverDocs


@dataclass(frozen=True, order=True)
class SectionMatch:
    """Result of section matching with confidence score.

    Attributes:
        score: Match confidence (0.0-1.0), used for ordering.
        section: Matched section header text.
    """

    score: float
    section: str


class SectionNotFoundError(Exception):
    """Raised when requested section is not found in document.

    Attributes:
        query: The section name that was searched for.
        available_sections: List of section headers in the document.
        best_match: The closest matching section (if any).
        hint: Helpful suggestion for the user.
    """

    def __init__(
        self,
        query: str,
        available_sections: list[str],
        best_match: SectionMatch | None = None,
    ) -> None:
        self.query = query
        self.available_sections = available_sections
        self.best_match = best_match

        # Build helpful hint
        if best_match and best_match.score > 0.4:
            self.hint = f"Did you mean '{best_match.section}'?"
        else:
            sections_preview = ", ".join(available_sections[:5])
            if len(available_sections) > 5:
                sections_preview += f" (and {len(available_sections) - 5} more)"
            self.hint = f"Available sections: {sections_preview}"

        message = f"Section '{query}' not found. {self.hint}"
        super().__init__(message)


class SectionMatcher(Protocol):
    """Protocol for section matching strategies (OCP)."""

    def match(self, query: str, sections: list[str]) -> SectionMatch | None:
        """Find best matching section for query.

        Args:
            query: Section name to search for.
            sections: List of available section headers.

        Returns:
            SectionMatch with section name and confidence score,
            or None if no match above threshold.
        """
        ...


class FuzzyMatcher:
    """Fuzzy section matcher using difflib SequenceMatcher.

    Handles partial matches, typos, and case differences.
    Uses Levenshtein-like ratio for scoring.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize with match threshold.

        Args:
            threshold: Minimum score (0.0-1.0) to consider a match.
                Default 0.5 allows partial matches like "server" -> "Server options".
        """
        self.threshold = threshold

    def match(self, query: str, sections: list[str]) -> SectionMatch | None:
        """Find best fuzzy match for query among sections."""
        if not sections:
            return None

        query_lower = query.lower()
        best_match: SectionMatch | None = None

        for section in sections:
            section_lower = section.lower()

            # Check for exact match first (fast path)
            if query_lower == section_lower:
                return SectionMatch(score=1.0, section=section)

            # Check if query is substring (e.g., "server" in "Server options")
            if query_lower in section_lower:
                # Score based on how much of the section the query covers
                substring_score = len(query_lower) / len(section_lower)
                # Boost substring matches to be competitive with ratio
                score = max(substring_score, 0.6 + (substring_score * 0.4))
            else:
                # Use SequenceMatcher ratio for fuzzy comparison
                score = SequenceMatcher(None, query_lower, section_lower).ratio()

            if score >= self.threshold:
                candidate = SectionMatch(score=score, section=section)
                if best_match is None or candidate > best_match:
                    best_match = candidate

        return best_match


class ExactMatcher:
    """Exact section matcher (case-insensitive).

    For backward compatibility with existing behavior.
    """

    def match(self, query: str, sections: list[str]) -> SectionMatch | None:
        """Find exact case-insensitive match."""
        query_lower = query.lower()

        for section in sections:
            if section.lower() == query_lower:
                return SectionMatch(score=1.0, section=section)

        return None


class SectionFilter:
    """Filters document content to specific sections.

    Uses strategy pattern (SectionMatcher) for matching behavior.
    Raises SectionNotFoundError with helpful info on mismatch.
    """

    def __init__(self, matcher: SectionMatcher | None = None) -> None:
        """Initialize with matching strategy.

        Args:
            matcher: Section matching strategy. Defaults to FuzzyMatcher.
        """
        self.matcher = matcher or FuzzyMatcher()

    def filter(
        self,
        docs: DriverDocs,
        section: str,
        fuzzy: bool = True,
    ) -> DriverDocs:
        """Filter document to requested section.

        Args:
            docs: Source document to filter.
            section: Section header to extract.
            fuzzy: If True, use fuzzy matching. If False, use exact match.

        Returns:
            New DriverDocs with only the matched section's content.

        Raises:
            SectionNotFoundError: If section not found, with available sections
                and best match info.
        """
        # Use appropriate matcher
        matcher = self.matcher if fuzzy else ExactMatcher()

        # Find best match
        match = matcher.match(section, docs.sections)

        if match is None:
            # Get best fuzzy match for hint even if using exact matcher
            fuzzy_matcher = FuzzyMatcher(threshold=0.3)
            best_fuzzy = fuzzy_matcher.match(section, docs.sections)

            raise SectionNotFoundError(
                query=section,
                available_sections=docs.sections,
                best_match=best_fuzzy,
            )

        # Extract section content
        filtered_content = self._extract_section(docs.content, match.section)

        # Return new DriverDocs with filtered content
        return DriverDocs(
            driver=docs.driver,
            version=docs.version,
            source_url=docs.source_url,
            content=filtered_content,
            sections=[match.section],
            cached_at=docs.cached_at,
        )

    @staticmethod
    def _extract_section(content: str, section_name: str) -> str:
        """Extract content for a specific section header.

        Includes content from the matched header until the next header
        of equal or higher level.
        """
        lines = content.splitlines()
        section_lower = section_name.lower()
        start_idx: int | None = None
        start_level: int = 0

        for i, line in enumerate(lines):
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                header_level = len(match.group(1))
                header_text = match.group(2).strip()

                if start_idx is not None and header_level <= start_level:
                    # Found end of section
                    return "\n".join(lines[start_idx:i])

                if header_text.lower() == section_lower:
                    start_idx = i
                    start_level = header_level

        if start_idx is not None:
            # Section extends to end of document
            return "\n".join(lines[start_idx:])

        # Section not found - shouldn't happen if match was valid
        return content
