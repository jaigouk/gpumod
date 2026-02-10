"""Protocols and data types for model discovery.

Defines the ModelSearcher protocol (ISP) and SearchResult dataclass
for consistent model discovery across different sources.

SOLID Principles:
- ISP: ModelSearcher is a minimal interface for search operations
- DIP: MCP tools depend on this protocol, not concrete implementations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True)
class SearchResult:
    """Immutable model search result.

    Contains model metadata with format detection and driver hints.

    Attributes:
        repo_id: Full HuggingFace repo ID (e.g., "unsloth/Qwen-GGUF").
        name: Human-readable model name.
        description: Model description from repo card, or None.
        last_modified: Last modification timestamp.
        tags: Tuple of tags from the repo.
        model_format: Detected format ("gguf", "safetensors", "unknown").
        driver_hint: Recommended driver ("llamacpp", "vllm", None).
    """

    repo_id: str
    name: str
    description: str | None
    last_modified: datetime
    tags: tuple[str, ...]
    model_format: str
    driver_hint: str | None


@runtime_checkable
class ModelSearcher(Protocol):
    """Protocol for model search implementations.

    All model searchers must implement this interface, enabling
    dependency inversion in MCP tools.

    SOLID:
    - ISP: Single method interface focused on search
    - LSP: Any implementation can substitute another
    """

    async def search(
        self,
        *,
        query: str,
        author: str | None = None,
        driver: str | None = None,
        limit: int = 50,
        force_refresh: bool = False,
    ) -> list[SearchResult]:
        """Search for models matching the query.

        Args:
            query: Search query (model name, keywords).
            author: Optional HuggingFace organization filter.
            driver: Optional driver filter ("llamacpp", "vllm", "any").
            limit: Maximum results to return.
            force_refresh: Bypass cache and fetch fresh data.

        Returns:
            List of SearchResult matching the query.
        """
        ...
