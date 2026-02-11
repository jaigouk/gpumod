"""Config.json fetcher from HuggingFace repos.

Fetches and parses model configuration files to detect architecture,
context length, MoE status, and other model properties.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from gpumod.discovery.gguf_metadata import RepoNotFoundError

logger = logging.getLogger(__name__)


class ConfigNotFoundError(Exception):
    """Raised when config.json is not found in the repo."""


@dataclass(frozen=True)
class ModelConfig:
    """Immutable model configuration parsed from config.json.

    Attributes:
        repo_id: HuggingFace repository ID.
        architectures: List of architecture names (e.g., ["LlamaForCausalLM"]).
        total_params: Estimated total parameters, or None if unknown.
        is_moe: Whether this is a Mixture of Experts model.
        num_experts: Number of experts if MoE, else None.
        context_length: Maximum context length (max_position_embeddings).
        vocab_size: Vocabulary size.
        raw_config: Original config.json dict for advanced use.
    """

    repo_id: str
    architectures: list[str]
    total_params: int | None
    is_moe: bool
    num_experts: int | None
    context_length: int | None
    vocab_size: int | None
    raw_config: dict[str, Any]


class ConfigFetcher:
    """Fetches and caches config.json from HuggingFace repos.

    Uses TTL-based caching to minimize API calls. Default TTL is 1 hour.

    Example:
        >>> fetcher = ConfigFetcher()
        >>> config = await fetcher.fetch("meta-llama/Llama-3.1-8B")
        >>> print(config.context_length)
        131072
    """

    # URL template for fetching config.json from HuggingFace
    _URL_TEMPLATE = "https://huggingface.co/{repo_id}/raw/main/config.json"

    def __init__(self, cache_ttl_seconds: int = 3600) -> None:
        """Initialize the fetcher.

        Args:
            cache_ttl_seconds: Cache time-to-live in seconds. Default 1 hour.
        """
        self._cache: dict[str, tuple[ModelConfig, float]] = {}
        self._ttl = cache_ttl_seconds

    async def fetch(self, repo_id: str) -> ModelConfig:
        """Fetch and parse config.json from a HuggingFace repo.

        Args:
            repo_id: HuggingFace repo ID (e.g., "meta-llama/Llama-3.1-8B").

        Returns:
            ModelConfig with parsed configuration data.

        Raises:
            ConfigNotFoundError: If config.json doesn't exist in the repo.
            RepoNotFoundError: If the repo doesn't exist.
        """

        # Check cache first
        if repo_id in self._cache:
            config, timestamp = self._cache[repo_id]
            if time.monotonic() - timestamp < self._ttl:
                logger.debug("Cache hit for %s", repo_id)
                return config

        # Fetch from HuggingFace
        url = self._URL_TEMPLATE.format(repo_id=repo_id)
        logger.debug("Fetching config.json from %s", url)

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url)

                if response.status_code == 404:
                    # Could be repo not found or config.json not found
                    raise ConfigNotFoundError(f"config.json not found for {repo_id}")

                response.raise_for_status()
                raw_config = response.json()

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    raise ConfigNotFoundError(
                        f"config.json not found for {repo_id}"
                    ) from exc
                raise
            except httpx.RequestError as exc:
                if "not found" in str(exc).lower():
                    raise RepoNotFoundError(f"Repo not found: {repo_id}") from exc
                raise

        # Parse the config
        config = self._parse_config(repo_id, raw_config)

        # Cache the result
        self._cache[repo_id] = (config, time.monotonic())
        logger.debug("Cached config for %s", repo_id)

        return config

    def _parse_config(self, repo_id: str, raw_config: dict[str, Any]) -> ModelConfig:
        """Parse raw config.json into ModelConfig.

        Args:
            repo_id: Repository ID.
            raw_config: Raw config.json dictionary.

        Returns:
            Parsed ModelConfig.
        """
        # Parse architectures
        architectures = raw_config.get("architectures", [])
        if not isinstance(architectures, list):
            architectures = []

        # Parse context length (try multiple field names)
        context_length = (
            raw_config.get("max_position_embeddings")
            or raw_config.get("max_sequence_length")
            or raw_config.get("n_positions")
        )

        # Parse vocab size
        vocab_size = raw_config.get("vocab_size")

        # Detect MoE model and expert count
        is_moe, num_experts = self._detect_moe(raw_config)

        # Estimate total params (if available)
        total_params = raw_config.get("num_parameters")

        return ModelConfig(
            repo_id=repo_id,
            architectures=architectures,
            total_params=total_params,
            is_moe=is_moe,
            num_experts=num_experts,
            context_length=context_length,
            vocab_size=vocab_size,
            raw_config=raw_config,
        )

    def _detect_moe(self, raw_config: dict[str, Any]) -> tuple[bool, int | None]:
        """Detect if model is MoE and count experts.

        Different MoE architectures use different field names:
        - Mixtral: num_local_experts
        - DeepSeek: n_routed_experts
        - Qwen MoE: num_experts

        Args:
            raw_config: Raw config.json dictionary.

        Returns:
            Tuple of (is_moe, num_experts).
        """
        # Check various MoE indicator fields
        moe_fields = [
            "num_local_experts",  # Mixtral
            "n_routed_experts",  # DeepSeek
            "num_experts",  # Generic
            "moe_num_experts",  # Alternative
        ]

        for field in moe_fields:
            if field in raw_config:
                num_experts = raw_config[field]
                if isinstance(num_experts, int) and num_experts > 1:
                    return True, num_experts

        # Also check architecture name for MoE indicators
        architectures = raw_config.get("architectures", [])
        for arch in architectures:
            if isinstance(arch, str) and "moe" in arch.lower():
                # MoE architecture but couldn't find expert count
                return True, None

        return False, None
