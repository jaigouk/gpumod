"""HuggingFace model searcher.

Searches the full HuggingFace Hub for models, detecting format
and providing driver hints for both GGUF and Safetensors models.

SOLID Principles:
- SRP: Only handles HuggingFace Hub search
- OCP: New formats can be added to detection without modification
- LSP: Implements ModelSearcher protocol
- DIP: Depends on HfApi abstraction
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import UTC, datetime

from huggingface_hub import HfApi

from gpumod.discovery.protocols import ModelSearcher, SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format Detection Functions
# ---------------------------------------------------------------------------


def detect_model_format(tags: list[str]) -> str:
    """Detect model format from repository tags.

    Args:
        tags: List of tags from HuggingFace repository.

    Returns:
        Detected format: "gguf", "safetensors", or "unknown".
    """
    tags_lower = [t.lower() for t in tags]

    # GGUF takes priority (more specific quantized format)
    if "gguf" in tags_lower:
        return "gguf"

    # Safetensors is the standard vLLM format
    if "safetensors" in tags_lower:
        return "safetensors"

    return "unknown"


def get_driver_hint(model_format: str) -> str | None:
    """Map model format to recommended driver.

    Args:
        model_format: Model format ("gguf", "safetensors", "unknown").

    Returns:
        Driver name ("llamacpp", "vllm") or None for unknown.
    """
    mapping = {
        "gguf": "llamacpp",
        "safetensors": "vllm",
    }
    return mapping.get(model_format)


# ---------------------------------------------------------------------------
# HuggingFaceSearcher Implementation
# ---------------------------------------------------------------------------


class HuggingFaceSearcher:
    """Searches HuggingFace Hub for models with format detection.

    Implements ModelSearcher protocol for unified model discovery.
    Supports filtering by driver type (llamacpp for GGUF, vllm for Safetensors).

    Example:
        >>> searcher = HuggingFaceSearcher()
        >>> results = await searcher.search(query="deepseek ocr")
        >>> for r in results:
        ...     print(f"{r.name}: {r.driver_hint}")

        >>> # Filter to GGUF only
        >>> gguf_models = await searcher.search(query="llama", driver="llamacpp")
    """

    # Pattern for extracting model name from repo ID
    _NAME_PATTERN = re.compile(r"^[^/]+/(.+?)(?:-GGUF|-gguf)?$", re.IGNORECASE)

    def __init__(
        self,
        *,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        """Initialize the searcher.

        Args:
            cache_ttl_seconds: Cache TTL in seconds (default 1 hour).
        """
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, list[SearchResult]] = {}
        self._cache_time: dict[str, float] = {}

    async def search(
        self,
        *,
        query: str,
        author: str | None = None,
        driver: str | None = None,
        limit: int = 50,
        force_refresh: bool = False,
    ) -> list[SearchResult]:
        """Search HuggingFace Hub for models.

        Args:
            query: Search query (model name, keywords).
            author: Optional HuggingFace organization filter.
            driver: Optional driver filter ("llamacpp", "vllm", "any", None).
            limit: Maximum results to return.
            force_refresh: Bypass cache and fetch fresh data.

        Returns:
            List of SearchResult matching the query.
        """
        # Build cache key
        cache_key = f"{query}:{author or 'all'}:{driver or 'any'}"

        # Check cache
        if not force_refresh and cache_key in self._cache:
            cache_time = self._cache_time.get(cache_key, 0)
            if time.monotonic() - cache_time < self._cache_ttl:
                return self._cache[cache_key][:limit]

        # Fetch from HuggingFace
        results = await self._fetch_models(
            query=query,
            author=author,
            driver=driver,
            limit=limit,
        )

        # Cache results
        self._cache[cache_key] = results
        self._cache_time[cache_key] = time.monotonic()

        return results[:limit]

    async def _fetch_models(  # noqa: C901, PLR0912
        self,
        *,
        query: str,
        author: str | None,
        driver: str | None,
        limit: int,
    ) -> list[SearchResult]:
        """Fetch models from HuggingFace Hub API.

        Args:
            query: Search query.
            author: Optional organization filter.
            driver: Optional driver filter.
            limit: Maximum results.

        Returns:
            List of SearchResult.
        """
        api = HfApi()

        # Build API kwargs
        kwargs: dict[str, object] = {
            "search": query,
            "limit": limit * 2,  # Fetch more to account for filtering
        }

        if author:
            kwargs["author"] = author

        # Apply format filter based on driver
        if driver == "llamacpp":
            kwargs["filter"] = "gguf"
        elif driver == "vllm":
            kwargs["filter"] = "safetensors"
        # For "any" or None, don't filter by format at API level

        logger.debug("HuggingFace API query: %s", kwargs)

        # Run in thread since huggingface_hub is sync
        raw_models = await asyncio.to_thread(
            api.list_models,
            **kwargs,
        )

        results: list[SearchResult] = []

        for model in raw_models:
            repo_id = model.id or getattr(model, "modelId", None)
            if not repo_id:
                continue

            tags = list(model.tags or [])
            model_format = detect_model_format(tags)
            driver_hint = get_driver_hint(model_format)

            # Apply driver filter (post-fetch for "any" searches)
            if driver and driver != "any":
                if driver == "llamacpp" and model_format != "gguf":
                    continue
                if driver == "vllm" and model_format != "safetensors":
                    continue

            # Skip unknown formats
            if model_format == "unknown":
                continue

            # Extract name
            name = self._extract_name(repo_id)

            # Get description
            description = None
            if hasattr(model, "cardData") and model.cardData and isinstance(model.cardData, dict):
                description = model.cardData.get("description")

            # Get last modified
            last_modified = model.lastModified
            if last_modified is None:
                last_modified = datetime.now(tz=UTC)
            elif not last_modified.tzinfo:
                last_modified = last_modified.replace(tzinfo=UTC)

            results.append(
                SearchResult(
                    repo_id=repo_id,
                    name=name,
                    description=description,
                    last_modified=last_modified,
                    tags=tuple(tags),
                    model_format=model_format,
                    driver_hint=driver_hint,
                )
            )

            if len(results) >= limit:
                break

        logger.info("Found %d models for query '%s'", len(results), query)
        return results

    def _extract_name(self, repo_id: str) -> str:
        """Extract human-readable name from repo ID.

        Args:
            repo_id: Full repo ID like "unsloth/Qwen3-Coder-GGUF".

        Returns:
            Human-readable name like "Qwen3 Coder".
        """
        match = self._NAME_PATTERN.match(repo_id)
        raw_name = match.group(1) if match else repo_id.split("/")[-1]

        # Remove common suffixes
        for suffix in ("-GGUF", "-gguf", "_GGUF", "_gguf"):
            raw_name = raw_name.removesuffix(suffix)

        # Convert hyphens/underscores to spaces
        return raw_name.replace("-", " ").replace("_", " ")

    def clear_cache(self) -> None:
        """Clear the search cache."""
        self._cache.clear()
        self._cache_time.clear()


# Ensure HuggingFaceSearcher satisfies the protocol
_: ModelSearcher = HuggingFaceSearcher()
