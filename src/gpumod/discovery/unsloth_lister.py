"""HuggingFace model lister.

Lists GGUF models from HuggingFace Hub with support for:
- Organization filtering (--author)
- Keyword search (--search)
- Task filtering (--task)
Provides caching and filtering capabilities for model discovery.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar

logger = logging.getLogger(__name__)


class HuggingFaceAPIError(Exception):
    """Raised when HuggingFace API call fails."""


@dataclass(frozen=True)
class UnslothModel:
    """Immutable model metadata from HuggingFace repos.

    Named UnslothModel for backwards compatibility, but supports any HF org.

    Attributes:
        repo_id: Full repo ID (e.g., "unsloth/Qwen3-Coder-Next-GGUF").
        name: Human-readable model name.
        description: Model description from repo card, or None.
        last_modified: Last modification timestamp.
        tags: Tuple of tags from the repo.
        has_gguf: Whether the repo contains .gguf files.
    """

    repo_id: str
    name: str
    description: str | None
    last_modified: datetime
    tags: tuple[str, ...]
    has_gguf: bool


# Alias for clarity
HFModel = UnslothModel


class UnslothModelLister:
    """Lists GGUF models from HuggingFace organizations.

    Provides caching to avoid repeated API calls and supports filtering
    by task type, keyword search, and organization.

    Example:
        >>> lister = UnslothModelLister()
        >>> models = await lister.list_models(task="code")
        >>> for model in models:
        ...     print(f"{model.name}: {model.repo_id}")

        >>> # Search by keyword
        >>> lister = UnslothModelLister(author=None)
        >>> models = await lister.list_models(search="deepseek")
    """

    # Pattern for extracting model name from repo ID
    _NAME_PATTERN = re.compile(r"^[^/]+/(.+?)(?:-GGUF)?$", re.IGNORECASE)

    # Task keywords for filtering
    _TASK_KEYWORDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "code": ("code", "coder", "coding", "starcoder", "codellama"),
        "chat": ("chat", "instruct", "assistant"),
        "embed": ("embed", "embedding", "bge", "e5"),
        "reasoning": ("reason", "math", "r1", "deepseek"),
    }

    def __init__(
        self,
        *,
        cache_ttl_seconds: int = 3600,
        author: str | None = "unsloth",
    ) -> None:
        """Initialize the lister.

        Args:
            cache_ttl_seconds: Cache TTL in seconds (default 1 hour).
            author: HuggingFace organization to list from. None for search mode.
        """
        self._cache_ttl = cache_ttl_seconds
        self._author = author
        self._cache: dict[str, list[UnslothModel]] = {}  # Cache by search key
        self._cache_time: dict[str, float] = {}

    async def list_models(
        self,
        *,
        task: str | None = None,
        search: str | None = None,
        force_refresh: bool = False,
    ) -> list[UnslothModel]:
        """List GGUF models from HuggingFace.

        Args:
            task: Optional task filter (code, chat, embed, reasoning).
            search: Optional model name search query (e.g., "deepseek", "kimi").
            force_refresh: Bypass cache and fetch fresh data.

        Returns:
            List of UnslothModel matching the filter.

        Raises:
            HuggingFaceAPIError: If the API call fails.
        """
        # Create cache key from search params
        cache_key = f"{self._author or 'all'}:{search or ''}"

        # Check cache
        if not force_refresh and cache_key in self._cache:
            cache_time = self._cache_time.get(cache_key, 0)
            if time.monotonic() - cache_time < self._cache_ttl:
                return self._filter_by_task(self._cache[cache_key], task)

        # Fetch from HuggingFace
        try:
            models = await self._fetch_models(search=search)
            self._cache[cache_key] = models
            self._cache_time[cache_key] = time.monotonic()
        except Exception as exc:
            # If we have cached data, return it on error
            if cache_key in self._cache:
                logger.warning("HuggingFace API error, returning cached data: %s", exc)
                return self._filter_by_task(self._cache[cache_key], task)
            raise HuggingFaceAPIError(str(exc)) from exc

        return self._filter_by_task(models, task)

    async def _fetch_models(self, *, search: str | None = None) -> list[UnslothModel]:
        """Fetch models from HuggingFace Hub API.

        Args:
            search: Optional search query for model names.

        Returns:
            List of UnslothModel for GGUF repos.
        """
        import time as time_module

        from huggingface_hub import HfApi

        api = HfApi()

        # Build search kwargs
        kwargs: dict[str, object] = {}
        if self._author:
            kwargs["author"] = self._author
        if search:
            # HuggingFace search query - search in model name
            kwargs["search"] = search
        # Filter for GGUF models using tags filter
        kwargs["filter"] = "gguf"

        logger.debug("HuggingFace API query: %s", kwargs)
        start = time_module.monotonic()

        # Run in thread since huggingface_hub is sync
        raw_models = await asyncio.to_thread(
            api.list_models,
            **kwargs,
        )

        elapsed = time_module.monotonic() - start
        logger.debug("HuggingFace API returned in %.2fs", elapsed)

        models: list[UnslothModel] = []
        for model in raw_models:
            repo_id = model.id or model.modelId
            if not repo_id:
                continue

            # Check if this is a GGUF repo (by name or tags)
            tags = tuple(model.tags or [])
            is_gguf = "gguf" in tags or "GGUF" in repo_id or "gguf" in repo_id.lower()

            if not is_gguf:
                continue

            # Extract name from repo ID
            name = self._extract_name(repo_id)

            # Get description from card data
            description = None
            if hasattr(model, "cardData") and model.cardData and isinstance(model.cardData, dict):
                description = model.cardData.get("model_name") or model.cardData.get("description")

            # Get last modified
            last_modified = model.lastModified
            if last_modified is None:
                last_modified = datetime.now(tz=UTC)
            elif not last_modified.tzinfo:
                last_modified = last_modified.replace(tzinfo=UTC)

            models.append(
                UnslothModel(
                    repo_id=repo_id,
                    name=name,
                    description=description,
                    last_modified=last_modified,
                    tags=tags,
                    has_gguf=True,
                )
            )

        logger.info("Found %d GGUF models from %s", len(models), self._author)
        return models

    def _extract_name(self, repo_id: str) -> str:
        """Extract human-readable name from repo ID.

        Args:
            repo_id: Full repo ID like "unsloth/Qwen3-Coder-Next-GGUF".

        Returns:
            Human-readable name like "Qwen3 Coder Next".
        """
        match = self._NAME_PATTERN.match(repo_id)
        # Fallback: just use the part after the slash
        raw_name = match.group(1) if match else repo_id.split("/")[-1]

        # Remove common suffixes
        for suffix in ("-GGUF", "-gguf", "_GGUF", "_gguf"):
            raw_name = raw_name.removesuffix(suffix)

        # Convert hyphens/underscores to spaces and title case
        return raw_name.replace("-", " ").replace("_", " ")

    def _filter_by_task(
        self,
        models: list[UnslothModel],
        task: str | None,
    ) -> list[UnslothModel]:
        """Filter models by task type.

        Args:
            models: List of models to filter.
            task: Task type (code, chat, embed, reasoning) or None.

        Returns:
            Filtered list of models.
        """
        if task is None:
            return models

        keywords = self._TASK_KEYWORDS.get(task.lower(), ())
        if not keywords:
            logger.warning("Unknown task type: %s", task)
            return models

        result = []
        for model in models:
            # Check name and tags for keywords
            name_lower = model.name.lower()
            tags_lower = [t.lower() for t in model.tags]

            for keyword in keywords:
                if keyword in name_lower or keyword in tags_lower:
                    result.append(model)
                    break

        return result

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._cache.clear()
        self._cache_time.clear()
