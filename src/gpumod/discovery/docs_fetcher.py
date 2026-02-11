"""Driver documentation fetcher for llama.cpp and vLLM.

Fetches and caches driver documentation from GitHub/official sources
to provide up-to-date command-line flags and configuration options.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import ClassVar

import httpx

logger = logging.getLogger(__name__)

# 24 hours in seconds
_DEFAULT_TTL_SECONDS = 86400


class DocsNotFoundError(Exception):
    """Raised when driver docs cannot be fetched from any URL."""


class DriverNotInstalledError(Exception):
    """Raised when the driver binary is not installed."""


@dataclass(frozen=True)
class DriverDocs:
    """Immutable driver documentation result.

    Attributes:
        driver: Driver name ("llamacpp" or "vllm").
        version: Detected or explicitly provided version string.
        source_url: The URL the content was fetched from.
        content: Raw markdown content.
        sections: List of parsed section headers.
        cached_at: When this result was cached.
    """

    driver: str
    version: str
    source_url: str
    content: str
    sections: list[str] = field(default_factory=list)
    cached_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


class DriverDocsFetcher:
    """Fetches and caches driver documentation from upstream sources.

    Supports llama.cpp and vLLM with automatic version detection,
    fallback URL chains, and TTL-based caching.

    Example:
        >>> fetcher = DriverDocsFetcher()
        >>> docs = await fetcher.fetch("llamacpp")
        >>> print(docs.sections)
        ['Usage', 'Server options', ...]
    """

    URL_TEMPLATES: ClassVar[dict[str, list[str]]] = {
        "llamacpp": [
            "https://raw.githubusercontent.com/ggml-org/llama.cpp/{version}/tools/server/README.md",
        ],
        "vllm": [
            "https://raw.githubusercontent.com/vllm-project/vllm/v{version}/docs/configuration/engine_args.md",
            "https://raw.githubusercontent.com/vllm-project/vllm/v{version}/docs/serving/engine_args.md",
        ],
    }

    # Fallback branch names when version detection fails
    _FALLBACK_BRANCHES: ClassVar[dict[str, str]] = {
        "llamacpp": "master",
        "vllm": "main",
    }

    _VALID_DRIVERS: frozenset[str] = frozenset({"llamacpp", "vllm"})

    def __init__(self, cache_ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self._cache: dict[str, tuple[DriverDocs, float]] = {}
        self._ttl = cache_ttl_seconds

    async def detect_version(self, driver: str) -> str | None:
        """Auto-detect installed version of a driver.

        Args:
            driver: "llamacpp" or "vllm".

        Returns:
            Version string (e.g., "b7999" for llamacpp, "0.15.1" for vllm),
            or None if the driver is not installed.
        """
        if driver == "llamacpp":
            return await self._detect_llamacpp_version()
        if driver == "vllm":
            return await self._detect_vllm_version()
        return None

    async def fetch(
        self,
        driver: str,
        version: str | None = None,
        section: str | None = None,
    ) -> DriverDocs:
        """Fetch driver documentation with caching and fallback URLs.

        Args:
            driver: "llamacpp" or "vllm".
            version: Explicit version string. Auto-detected if None.
            section: Optional section header to filter content by.

        Returns:
            DriverDocs with fetched content.

        Raises:
            DocsNotFoundError: If all URL templates fail.
            ValueError: If driver is not supported.
        """
        if driver not in self._VALID_DRIVERS:
            msg = f"Unsupported driver: {driver!r}. Must be one of: {sorted(self._VALID_DRIVERS)}"
            raise ValueError(msg)

        # Resolve version
        if version is None:
            version = await self.detect_version(driver)
        if version is None:
            version = self._FALLBACK_BRANCHES[driver]

        # Check cache
        cache_key = f"{driver}:{version}"
        if cache_key in self._cache:
            docs, timestamp = self._cache[cache_key]
            if time.monotonic() - timestamp < self._ttl:
                logger.debug("Cache hit for %s", cache_key)
                if section is not None:
                    return self._filter_section(docs, section)
                return docs

        # Try each URL template
        templates = self.URL_TEMPLATES.get(driver, [])
        last_error: Exception | None = None

        for template in templates:
            url = template.format(version=version)
            logger.debug("Trying URL: %s", url)

            try:
                content = await self._fetch_url(url)
                sections = self._parse_sections(content)
                docs = DriverDocs(
                    driver=driver,
                    version=version,
                    source_url=url,
                    content=content,
                    sections=sections,
                )

                # Cache the result
                self._cache[cache_key] = (docs, time.monotonic())
                logger.debug("Cached docs for %s", cache_key)

                if section is not None:
                    return self._filter_section(docs, section)
                return docs

            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.debug("Failed to fetch %s: %s", url, exc)
                last_error = exc
                continue

        msg = f"Could not fetch docs for {driver} v{version} from any URL"
        raise DocsNotFoundError(msg) from last_error

    async def _fetch_url(self, url: str) -> str:
        """Fetch raw content from a URL.

        Raises httpx errors on failure.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    @staticmethod
    def _parse_sections(content: str) -> list[str]:
        """Parse markdown section headers from content.

        Returns list of header text (without the ## prefix).
        """
        sections: list[str] = []
        for line in content.splitlines():
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                sections.append(match.group(2).strip())
        return sections

    @staticmethod
    def _filter_section(docs: DriverDocs, section: str) -> DriverDocs:
        """Return a new DriverDocs with only the requested section's content.

        Matches section headers case-insensitively. Includes content from
        the matched header until the next header of equal or higher level.
        """
        lines = docs.content.splitlines()
        section_lower = section.lower()
        start_idx: int | None = None
        start_level: int = 0

        for i, line in enumerate(lines):
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                header_level = len(match.group(1))
                header_text = match.group(2).strip()
                if start_idx is not None and header_level <= start_level:
                    # Found the end of our section
                    filtered_content = "\n".join(lines[start_idx:i])
                    return DriverDocs(
                        driver=docs.driver,
                        version=docs.version,
                        source_url=docs.source_url,
                        content=filtered_content,
                        sections=[section],
                        cached_at=docs.cached_at,
                    )
                if header_text.lower() == section_lower:
                    start_idx = i
                    start_level = header_level

        if start_idx is not None:
            # Section extends to end of document
            filtered_content = "\n".join(lines[start_idx:])
            return DriverDocs(
                driver=docs.driver,
                version=docs.version,
                source_url=docs.source_url,
                content=filtered_content,
                sections=[section],
                cached_at=docs.cached_at,
            )

        # Section not found — return full docs with empty section list
        return DriverDocs(
            driver=docs.driver,
            version=docs.version,
            source_url=docs.source_url,
            content=docs.content,
            sections=docs.sections,
            cached_at=docs.cached_at,
        )

    async def _detect_llamacpp_version(self) -> str | None:
        """Detect llama.cpp version via `llama-server --version`."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "llama-server",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            output = stdout.decode().strip()

            # Parse "version: 7999 (abc123)" → "b7999"
            match = re.search(r"version:\s*(\d+)", output)
            if match:
                return f"b{match.group(1)}"
            return None
        except (FileNotFoundError, TimeoutError, OSError):
            return None

    async def _detect_vllm_version(self) -> str | None:
        """Detect vLLM version via `pip show vllm`."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "pip",
                "show",
                "vllm",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            output = stdout.decode()

            # Parse "Version: 0.15.1" → "0.15.1"
            match = re.search(r"^Version:\s*(.+)$", output, re.MULTILINE)
            if match:
                return match.group(1).strip()
            return None
        except (FileNotFoundError, TimeoutError, OSError):
            return None
