"""Tests for DriverDocsFetcher — fetches driver docs from GitHub.

TDD: RED phase - these tests define the DriverDocsFetcher interface.
Tests cover version detection, HTTP fetching, caching, section filtering,
fallback URL chains, and SEC-V1 input validation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from gpumod.discovery.docs_fetcher import (
    DocsNotFoundError,
    DriverDocs,
    DriverDocsFetcher,
)

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

## Advanced

Expert offloading with `--n-cpu-moe`.
"""

VLLM_DOCS = """\
# Engine Arguments

## Model arguments

- `--model` Model name or path
- `--tensor-parallel-size` Number of GPUs

## Serving arguments

- `--port` Port for API
- `--gpu-memory-utilization` GPU memory fraction
"""


@pytest.fixture
def llamacpp_readme():
    return SAMPLE_README


@pytest.fixture
def vllm_docs():
    return VLLM_DOCS


@pytest.fixture
def fetcher():
    return DriverDocsFetcher(cache_ttl_seconds=60)


# ---------------------------------------------------------------------------
# DriverDocs dataclass
# ---------------------------------------------------------------------------


class TestDriverDocs:
    """Tests for DriverDocs dataclass."""

    def test_driver_docs_fields(self):
        """DriverDocs has all required fields."""
        docs = DriverDocs(
            driver="llamacpp",
            version="b7999",
            source_url="https://example.com/README.md",
            content="# README",
            sections=["README"],
        )
        assert docs.driver == "llamacpp"
        assert docs.version == "b7999"
        assert docs.source_url == "https://example.com/README.md"
        assert docs.content == "# README"
        assert docs.sections == ["README"]
        assert docs.cached_at is not None

    def test_driver_docs_is_frozen(self):
        """DriverDocs is immutable."""
        docs = DriverDocs(
            driver="llamacpp",
            version="b7999",
            source_url="https://example.com",
            content="test",
        )
        with pytest.raises(AttributeError):
            docs.driver = "vllm"


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


class TestVersionDetection:
    """Tests for version auto-detection (llamacpp, vllm)."""

    async def test_detect_llamacpp_version(self, fetcher):
        """Detects llamacpp version from 'llama-server --version' output."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                b"version: 7999 (abc123def)",
                b"",
            )
            mock_exec.return_value = mock_proc

            version = await fetcher.detect_version("llamacpp")

            assert version == "b7999"
            mock_exec.assert_called_once_with(
                "llama-server",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

    async def test_detect_vllm_version(self, fetcher):
        """Detects vllm version from 'pip show vllm' output."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                b"Name: vllm\nVersion: 0.15.1\nSummary: ...\n",
                b"",
            )
            mock_exec.return_value = mock_proc

            version = await fetcher.detect_version("vllm")

            assert version == "0.15.1"

    async def test_detect_llamacpp_not_installed(self, fetcher):
        """Returns None when llama-server is not installed."""
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            version = await fetcher.detect_version("llamacpp")
            assert version is None

    async def test_detect_vllm_not_installed(self, fetcher):
        """Returns None when vllm is not installed."""
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            version = await fetcher.detect_version("vllm")
            assert version is None

    async def test_detect_timeout(self, fetcher):
        """Returns None when version detection times out."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.side_effect = asyncio.TimeoutError
            mock_exec.return_value = mock_proc

            version = await fetcher.detect_version("llamacpp")
            assert version is None

    async def test_detect_invalid_driver(self, fetcher):
        """Returns None for unknown driver names."""
        version = await fetcher.detect_version("unknown_driver")
        assert version is None

    async def test_detect_unparseable_output(self, fetcher):
        """Returns None when version output cannot be parsed."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                b"some unexpected output format",
                b"",
            )
            mock_exec.return_value = mock_proc

            version = await fetcher.detect_version("llamacpp")
            assert version is None


# ---------------------------------------------------------------------------
# Fetch with auto-detect version
# ---------------------------------------------------------------------------


class TestFetchAutoDetect:
    """Tests for fetch() with automatic version detection."""

    async def test_fetch_with_auto_detect(self, fetcher, llamacpp_readme):
        """fetch() auto-detects version then fetches docs."""
        with (
            patch.object(fetcher, "detect_version", return_value="b7999"),
            patch.object(fetcher, "_fetch_url", return_value=llamacpp_readme),
        ):
            docs = await fetcher.fetch("llamacpp")

            assert docs.driver == "llamacpp"
            assert docs.version == "b7999"
            assert "llama.cpp Server" in docs.content
            assert len(docs.sections) > 0

    async def test_fetch_with_explicit_version(self, fetcher, llamacpp_readme):
        """fetch() uses explicit version when provided."""
        with patch.object(fetcher, "_fetch_url", return_value=llamacpp_readme):
            docs = await fetcher.fetch("llamacpp", version="b8000")

            assert docs.version == "b8000"

    async def test_fetch_falls_back_to_master_when_not_installed(self, fetcher, llamacpp_readme):
        """fetch() uses 'master' branch when driver not installed."""
        with (
            patch.object(fetcher, "detect_version", return_value=None),
            patch.object(fetcher, "_fetch_url", return_value=llamacpp_readme),
        ):
            docs = await fetcher.fetch("llamacpp")

            assert docs.version == "master"

    async def test_fetch_vllm_falls_back_to_main(self, fetcher, vllm_docs):
        """fetch() uses 'main' branch for vllm when not installed."""
        with (
            patch.object(fetcher, "detect_version", return_value=None),
            patch.object(fetcher, "_fetch_url", return_value=vllm_docs),
        ):
            docs = await fetcher.fetch("vllm")

            assert docs.version == "main"


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


class TestCacheBehavior:
    """Tests for TTL-based caching."""

    async def test_cache_hit(self, fetcher, llamacpp_readme):
        """Second call within TTL returns cached result without HTTP request."""
        with (
            patch.object(fetcher, "detect_version", return_value="b7999"),
            patch.object(fetcher, "_fetch_url", return_value=llamacpp_readme) as mock_fetch,
        ):
            docs1 = await fetcher.fetch("llamacpp")
            docs2 = await fetcher.fetch("llamacpp")

            # Only one HTTP fetch should occur
            assert mock_fetch.call_count == 1
            assert docs1.content == docs2.content

    async def test_cache_miss_after_ttl_expiry(self, llamacpp_readme):
        """Refetches after TTL expires."""
        fetcher = DriverDocsFetcher(cache_ttl_seconds=0)  # Immediate expiry

        with (
            patch.object(fetcher, "detect_version", return_value="b7999"),
            patch.object(fetcher, "_fetch_url", return_value=llamacpp_readme) as mock_fetch,
        ):
            await fetcher.fetch("llamacpp")
            await asyncio.sleep(0.01)
            await fetcher.fetch("llamacpp")

            assert mock_fetch.call_count == 2

    async def test_different_drivers_have_separate_cache(
        self, fetcher, llamacpp_readme, vllm_docs
    ):
        """Different drivers have separate cache entries."""
        with (
            patch.object(fetcher, "detect_version", return_value="b7999"),
            patch.object(fetcher, "_fetch_url") as mock_fetch,
        ):
            mock_fetch.side_effect = [llamacpp_readme, vllm_docs]

            docs1 = await fetcher.fetch("llamacpp")
            docs2 = await fetcher.fetch("vllm")

            assert docs1.driver == "llamacpp"
            assert docs2.driver == "vllm"
            assert mock_fetch.call_count == 2

    async def test_different_versions_have_separate_cache(self, fetcher, llamacpp_readme):
        """Different versions have separate cache entries."""
        with patch.object(fetcher, "_fetch_url", return_value=llamacpp_readme) as mock_fetch:
            await fetcher.fetch("llamacpp", version="b7999")
            await fetcher.fetch("llamacpp", version="b8000")

            assert mock_fetch.call_count == 2


# ---------------------------------------------------------------------------
# Section filtering
# ---------------------------------------------------------------------------


class TestSectionFiltering:
    """Tests for section filtering by header."""

    async def test_filter_existing_section(self, fetcher, llamacpp_readme):
        """Filtering by existing section returns only that section's content."""
        with (
            patch.object(fetcher, "detect_version", return_value="b7999"),
            patch.object(fetcher, "_fetch_url", return_value=llamacpp_readme),
        ):
            docs = await fetcher.fetch("llamacpp", section="Server options")

            assert "Server options" in docs.content
            assert "--port" in docs.content
            # Should not include content from other sections
            assert "Expert offloading" not in docs.content

    async def test_filter_section_case_insensitive(self, fetcher, llamacpp_readme):
        """Section filtering is case-insensitive."""
        with (
            patch.object(fetcher, "detect_version", return_value="b7999"),
            patch.object(fetcher, "_fetch_url", return_value=llamacpp_readme),
        ):
            docs = await fetcher.fetch("llamacpp", section="server options")

            assert "Server options" in docs.content

    async def test_filter_nonexistent_section_returns_full(self, fetcher, llamacpp_readme):
        """Filtering by non-existent section returns full content."""
        with (
            patch.object(fetcher, "detect_version", return_value="b7999"),
            patch.object(fetcher, "_fetch_url", return_value=llamacpp_readme),
        ):
            docs = await fetcher.fetch("llamacpp", section="Nonexistent Section")

            # Returns full content when section not found
            assert "llama.cpp Server" in docs.content

    async def test_filter_last_section(self, fetcher, llamacpp_readme):
        """Section at end of document includes content to EOF."""
        with (
            patch.object(fetcher, "detect_version", return_value="b7999"),
            patch.object(fetcher, "_fetch_url", return_value=llamacpp_readme),
        ):
            docs = await fetcher.fetch("llamacpp", section="Advanced")

            assert "Expert offloading" in docs.content

    def test_parse_sections_extracts_headers(self):
        """_parse_sections extracts ## headers from content."""
        sections = DriverDocsFetcher._parse_sections(SAMPLE_README)

        assert "llama.cpp Server" in sections
        assert "Usage" in sections
        assert "Server options" in sections
        assert "Advanced" in sections


# ---------------------------------------------------------------------------
# Fallback URL chain
# ---------------------------------------------------------------------------


class TestFallbackURLChain:
    """Tests for fallback URL chain when first URL fails."""

    async def test_first_url_fails_second_succeeds(self, fetcher, vllm_docs):
        """When first URL returns 404, tries next URL in chain."""
        call_count = 0

        async def mock_fetch_url(url):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPStatusError(
                    "404",
                    request=MagicMock(),
                    response=MagicMock(status_code=404),
                )
            return vllm_docs

        with (
            patch.object(fetcher, "detect_version", return_value="0.15.1"),
            patch.object(fetcher, "_fetch_url", side_effect=mock_fetch_url),
        ):
            docs = await fetcher.fetch("vllm")

            assert docs.content == vllm_docs
            assert call_count == 2

    async def test_all_fallbacks_fail_raises_docs_not_found(self, fetcher):
        """DocsNotFoundError raised when all URLs fail."""
        with (
            patch.object(fetcher, "detect_version", return_value="0.15.1"),
            patch.object(
                fetcher,
                "_fetch_url",
                side_effect=httpx.HTTPStatusError(
                    "404",
                    request=MagicMock(),
                    response=MagicMock(status_code=404),
                ),
            ),
            pytest.raises(DocsNotFoundError, match="Could not fetch docs"),
        ):
            await fetcher.fetch("vllm")

    async def test_network_error_tries_next_url(self, fetcher, vllm_docs):
        """Network errors (RequestError) also trigger fallback."""
        call_count = 0

        async def mock_fetch_url(url):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.RequestError("Connection refused", request=MagicMock())
            return vllm_docs

        with (
            patch.object(fetcher, "detect_version", return_value="0.15.1"),
            patch.object(fetcher, "_fetch_url", side_effect=mock_fetch_url),
        ):
            docs = await fetcher.fetch("vllm")
            assert docs.content == vllm_docs


# ---------------------------------------------------------------------------
# Validation and error handling
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for input validation and error handling."""

    async def test_invalid_driver_raises_value_error(self, fetcher):
        """Unsupported driver name raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported driver"):
            await fetcher.fetch("invalid_driver")

    async def test_empty_driver_raises_value_error(self, fetcher):
        """Empty driver string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported driver"):
            await fetcher.fetch("")


# ---------------------------------------------------------------------------
# fetch_driver_docs MCP tool integration
# ---------------------------------------------------------------------------


def _make_mock_ctx():
    """Create a mock FastMCP Context."""
    ctx = MagicMock()
    ctx.fastmcp._lifespan_result = {}
    return ctx


class TestFetchDriverDocsTool:
    """Tests for fetch_driver_docs MCP tool (mcp_tools.py)."""

    async def test_fetch_returns_dict(self, llamacpp_readme):
        """Tool returns serializable dict with expected keys."""
        from gpumod.mcp_tools import fetch_driver_docs

        docs = DriverDocs(
            driver="llamacpp",
            version="b7999",
            source_url="https://example.com/README.md",
            content=llamacpp_readme,
            sections=["Usage", "Server options", "Advanced"],
        )

        with patch("gpumod.mcp_tools.DriverDocsFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.fetch.return_value = docs
            fetcher_cls.return_value = mock_fetcher

            result = await fetch_driver_docs(
                driver="llamacpp",
                ctx=_make_mock_ctx(),
            )

            assert "driver" in result
            assert "version" in result
            assert "content" in result
            assert "sections" in result
            assert "source_url" in result
            assert result["driver"] == "llamacpp"

    async def test_fetch_validates_driver(self):
        """Invalid driver returns VALIDATION_ERROR."""
        from gpumod.mcp_tools import fetch_driver_docs

        result = await fetch_driver_docs(
            driver="invalid",
            ctx=_make_mock_ctx(),
        )

        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_fetch_validates_version_format(self):
        """Invalid version format returns VALIDATION_ERROR."""
        from gpumod.mcp_tools import fetch_driver_docs

        result = await fetch_driver_docs(
            driver="llamacpp",
            version="; rm -rf /",
            ctx=_make_mock_ctx(),
        )

        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_fetch_validates_section_length(self):
        """Overly long section returns VALIDATION_ERROR."""
        from gpumod.mcp_tools import fetch_driver_docs

        result = await fetch_driver_docs(
            driver="llamacpp",
            section="x" * 300,
            ctx=_make_mock_ctx(),
        )

        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_fetch_validates_section_dangerous_patterns(self):
        """Section with injection patterns returns VALIDATION_ERROR."""
        from gpumod.mcp_tools import fetch_driver_docs

        result = await fetch_driver_docs(
            driver="llamacpp",
            section="section; rm -rf /",
            ctx=_make_mock_ctx(),
        )

        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_fetch_handles_docs_not_found(self):
        """DocsNotFoundError returns NOT_FOUND error."""
        from gpumod.mcp_tools import fetch_driver_docs

        with patch("gpumod.mcp_tools.DriverDocsFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.fetch.side_effect = DocsNotFoundError("not found")
            fetcher_cls.return_value = mock_fetcher

            result = await fetch_driver_docs(
                driver="llamacpp",
                ctx=_make_mock_ctx(),
            )

            assert "error" in result
            assert result["code"] == "NOT_FOUND"

    async def test_fetch_with_explicit_version(self, llamacpp_readme):
        """Tool passes explicit version to fetcher."""
        from gpumod.mcp_tools import fetch_driver_docs

        docs = DriverDocs(
            driver="llamacpp",
            version="b8000",
            source_url="https://example.com",
            content=llamacpp_readme,
            sections=[],
        )

        with patch("gpumod.mcp_tools.DriverDocsFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.fetch.return_value = docs
            fetcher_cls.return_value = mock_fetcher

            result = await fetch_driver_docs(
                driver="llamacpp",
                version="b8000",
                ctx=_make_mock_ctx(),
            )

            mock_fetcher.fetch.assert_awaited_once_with(
                driver="llamacpp", version="b8000", section=None
            )
            assert result["version"] == "b8000"

    async def test_fetch_with_section_filter(self, llamacpp_readme):
        """Tool uses SectionFilter for section filtering (gpumod-v7k refactor)."""
        from gpumod.mcp_tools import fetch_driver_docs

        # Full docs returned by fetcher (section=None)
        full_docs = DriverDocs(
            driver="llamacpp",
            version="b7999",
            source_url="https://example.com",
            content=llamacpp_readme,
            sections=["llama.cpp Server", "Usage", "Server options", "Advanced"],
        )

        with patch("gpumod.mcp_tools.DriverDocsFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.fetch.return_value = full_docs
            fetcher_cls.return_value = mock_fetcher

            result = await fetch_driver_docs(
                driver="llamacpp",
                section="Server options",
                ctx=_make_mock_ctx(),
            )

            # Fetcher is called without section (SectionFilter handles filtering)
            mock_fetcher.fetch.assert_awaited_once_with(
                driver="llamacpp", version=None, section=None
            )
            # Result should have filtered section
            assert result["sections"] == ["Server options"]
            assert "--port" in result["content"]
            # Should include truncation metadata
            assert "metadata" in result


# ---------------------------------------------------------------------------
# SEC-V1: Security input validation
# ---------------------------------------------------------------------------


class TestDocsFetcherSecurity:
    """SEC-V1 security tests for fetch_driver_docs tool."""

    async def test_version_rejects_shell_injection(self):
        """Version with shell metacharacters is rejected."""
        from gpumod.mcp_tools import fetch_driver_docs

        payloads = [
            "; rm -rf /",
            "$(whoami)",
            "`id`",
            "v1.0 && ls",
        ]
        for payload in payloads:
            result = await fetch_driver_docs(
                driver="llamacpp",
                version=payload,
                ctx=_make_mock_ctx(),
            )
            assert "error" in result, f"Shell injection not rejected: {payload}"
            assert result["code"] == "VALIDATION_ERROR"

    async def test_section_rejects_path_traversal(self):
        """Section with path traversal is rejected."""
        from gpumod.mcp_tools import fetch_driver_docs

        payloads = [
            "../../etc/passwd",
            "section/../../../etc",
        ]
        for payload in payloads:
            result = await fetch_driver_docs(
                driver="llamacpp",
                section=payload,
                ctx=_make_mock_ctx(),
            )
            assert "error" in result, f"Path traversal not rejected: {payload}"
            assert result["code"] == "VALIDATION_ERROR"

    async def test_section_rejects_template_injection(self):
        """Section with Jinja2 template syntax is rejected."""
        from gpumod.mcp_tools import fetch_driver_docs

        payloads = [
            "{{7*7}}",
            "{%import os%}",
        ]
        for payload in payloads:
            result = await fetch_driver_docs(
                driver="llamacpp",
                section=payload,
                ctx=_make_mock_ctx(),
            )
            assert "error" in result, f"Template injection not rejected: {payload}"
            assert result["code"] == "VALIDATION_ERROR"

    async def test_driver_rejects_injection(self):
        """Only 'llamacpp' and 'vllm' accepted."""
        from gpumod.mcp_tools import fetch_driver_docs

        payloads = [
            "llamacpp; rm -rf /",
            "' OR 1=1 --",
            "../../../etc/passwd",
        ]
        for payload in payloads:
            result = await fetch_driver_docs(
                driver=payload,
                ctx=_make_mock_ctx(),
            )
            assert "error" in result, f"Driver injection not rejected: {payload}"
            assert result["code"] == "VALIDATION_ERROR"
