"""Token-aware content truncation for MCP tool responses.

Implements SRP by separating truncation from doc fetching.
Aligns with MCP best practices: token-based limits (default 20k tokens).

Part of fix for gpumod-v7k: 88KB response exceeds MCP limits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


# Approximate tokens per character ratio for English markdown
# Conservative estimate: ~4 chars per token (actual varies by content)
_CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass(frozen=True)
class TruncationResult:
    """Result of content truncation.

    Attributes:
        content: Truncated content (with indicator if truncated).
        truncated: Whether content was truncated.
        original_length: Original content length (in unit).
        result_length: Result content length (in unit).
        unit: Unit of measurement ("tokens" or "chars").
    """

    content: str
    truncated: bool
    original_length: int
    result_length: int
    unit: str


class Truncator(Protocol):
    """Protocol for truncation strategies."""

    def truncate(self, content: str) -> TruncationResult:
        """Truncate content if it exceeds limit.

        Args:
            content: Content to potentially truncate.

        Returns:
            TruncationResult with truncated content and metadata.
        """
        ...


class TokenTruncator:
    """Token-aware truncation using character estimation.

    Uses conservative character-to-token ratio for estimation.
    Breaks at paragraph boundaries when possible.
    """

    def __init__(self, max_tokens: int = 20000) -> None:
        """Initialize with token limit.

        Args:
            max_tokens: Maximum tokens in output. Default 20000 per MCP best practices.
        """
        self.max_tokens = max_tokens
        self._max_chars = max_tokens * _CHARS_PER_TOKEN_ESTIMATE

    def truncate(self, content: str) -> TruncationResult:
        """Truncate content to fit within token limit."""
        original_tokens = self._estimate_tokens(content)

        if original_tokens <= self.max_tokens:
            return TruncationResult(
                content=content,
                truncated=False,
                original_length=original_tokens,
                result_length=original_tokens,
                unit="tokens",
            )

        # Find truncation point at paragraph boundary
        truncated_content = self._truncate_at_boundary(content)
        result_tokens = self._estimate_tokens(truncated_content)

        # Add truncation indicator
        indicator = (
            f"\n\n[...truncated, showing ~{result_tokens} of ~{original_tokens} tokens]"
        )
        truncated_content += indicator

        return TruncationResult(
            content=truncated_content,
            truncated=True,
            original_length=original_tokens,
            result_length=result_tokens,
            unit="tokens",
        )

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count from character count."""
        return len(content) // _CHARS_PER_TOKEN_ESTIMATE

    def _truncate_at_boundary(self, content: str) -> str:
        """Truncate at paragraph or section boundary.

        Tries to break at:
        1. Double newline (paragraph)
        2. Single newline
        3. Space
        4. Hard cutoff
        """
        # Reserve space for indicator
        target_chars = self._max_chars - 100

        if len(content) <= target_chars:
            return content

        # Try to find paragraph boundary near target
        search_start = max(0, target_chars - 500)
        search_region = content[search_start:target_chars]

        # Look for double newline (paragraph break)
        para_break = search_region.rfind("\n\n")
        if para_break != -1:
            return content[: search_start + para_break]

        # Look for section header before target
        lines = content[:target_chars].splitlines()
        for i in range(len(lines) - 1, max(0, len(lines) - 20), -1):
            if re.match(r"^#{1,6}\s+", lines[i]):
                # Found header - truncate before it
                return "\n".join(lines[:i])

        # Look for single newline
        newline_pos = search_region.rfind("\n")
        if newline_pos != -1:
            return content[: search_start + newline_pos]

        # Look for space
        space_pos = content[:target_chars].rfind(" ")
        if space_pos != -1:
            return content[:space_pos]

        # Hard cutoff
        return content[:target_chars]


class CharTruncator:
    """Simple character-based truncation (fallback).

    Use when token estimation isn't needed.
    """

    def __init__(self, max_chars: int = 80000) -> None:
        """Initialize with character limit.

        Args:
            max_chars: Maximum characters in output.
        """
        self.max_chars = max_chars

    def truncate(self, content: str) -> TruncationResult:
        """Truncate content to fit within character limit."""
        original_chars = len(content)

        if original_chars <= self.max_chars:
            return TruncationResult(
                content=content,
                truncated=False,
                original_length=original_chars,
                result_length=original_chars,
                unit="chars",
            )

        # Reserve space for indicator
        target = self.max_chars - 100

        # Try to break at paragraph
        truncated = self._truncate_at_boundary(content, target)

        # Add indicator
        indicator = (
            f"\n\n[...truncated, showing {len(truncated)} of {original_chars} chars]"
        )
        truncated += indicator

        return TruncationResult(
            content=truncated,
            truncated=True,
            original_length=original_chars,
            result_length=len(truncated) - len(indicator),
            unit="chars",
        )

    def _truncate_at_boundary(self, content: str, target: int) -> str:
        """Truncate at paragraph boundary."""
        if len(content) <= target:
            return content

        # Look for paragraph break
        search_start = max(0, target - 500)
        search_region = content[search_start:target]

        para_break = search_region.rfind("\n\n")
        if para_break != -1:
            return content[: search_start + para_break]

        newline_pos = search_region.rfind("\n")
        if newline_pos != -1:
            return content[: search_start + newline_pos]

        return content[:target]


# Default truncator instance
ContentTruncator = TokenTruncator
