"""Tests for gpumod.cli_discover -- Discover CLI command."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer.testing

from gpumod.cli import app
from gpumod.cli_discover import _format_vram
from gpumod.discovery.gguf_metadata import GGUFFile
from gpumod.discovery.system_info import SystemInfo
from gpumod.discovery.unsloth_lister import UnslothModel

runner = typer.testing.CliRunner()

# ANSI escape code pattern for stripping Rich formatting
_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text (Rich formatting)."""
    return _ANSI_PATTERN.sub("", text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_system_info() -> SystemInfo:
    """Standard RTX 4090 system info."""
    return SystemInfo(
        gpu_total_mb=24576,
        gpu_used_mb=512,
        gpu_available_mb=24064,
        gpu_name="NVIDIA RTX 4090",
        ram_total_mb=65536,
        ram_available_mb=58000,
        swap_available_mb=8000,
        current_mode=None,
        running_services=[],
    )


@pytest.fixture
def mock_models() -> list[UnslothModel]:
    """Sample Unsloth models."""
    return [
        UnslothModel(
            repo_id="unsloth/Qwen3-Coder-Next-GGUF",
            name="Qwen3 Coder Next",
            description="A code generation model",
            last_modified=datetime.now(tz=UTC),
            tags=("gguf", "code"),
            has_gguf=True,
        ),
        UnslothModel(
            repo_id="unsloth/Nemotron-3-Nano-GGUF",
            name="Nemotron 3 Nano",
            description="A chat model",
            last_modified=datetime.now(tz=UTC),
            tags=("gguf", "chat"),
            has_gguf=True,
        ),
    ]


@pytest.fixture
def mock_gguf_files() -> list[GGUFFile]:
    """Sample GGUF files that fit in VRAM."""
    return [
        GGUFFile(
            filename="model-Q4_K_M.gguf",
            size_bytes=20_000_000_000,
            quant_type="Q4_K_M",
            estimated_vram_mb=21000,
            is_split=False,
            split_parts=1,
        ),
        GGUFFile(
            filename="model-Q8_0.gguf",
            size_bytes=40_000_000_000,
            quant_type="Q8_0",
            estimated_vram_mb=42000,
            is_split=False,
            split_parts=1,
        ),
    ]


# ---------------------------------------------------------------------------
# TestDiscoverHelp
# ---------------------------------------------------------------------------


class TestDiscoverHelp:
    """Tests for discover command help."""

    def test_discover_help(self) -> None:
        """discover --help shows usage information."""
        result = runner.invoke(app, ["discover", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "discover" in output.lower()
        assert "--task" in output
        assert "--vram" in output
        assert "--context" in output

    def test_discover_help_shows_search_option(self) -> None:
        """--help shows --search option with description."""
        result = runner.invoke(app, ["discover", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--search" in output
        assert "-s" in output
        # Check description mentions model name search
        assert "search" in output.lower()

    def test_discover_help_shows_author_option(self) -> None:
        """--help shows --author option with default value."""
        result = runner.invoke(app, ["discover", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--author" in output
        assert "-a" in output
        # Check default is unsloth
        assert "unsloth" in output.lower()

    def test_discover_help_shows_verbose_option(self) -> None:
        """--help shows --verbose option for debug output."""
        result = runner.invoke(app, ["discover", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--verbose" in output
        assert "-v" in output

    def test_discover_help_shows_dry_run_option(self) -> None:
        """--help shows --dry-run option."""
        result = runner.invoke(app, ["discover", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--dry-run" in output

    def test_discover_help_shows_json_option(self) -> None:
        """--help shows --json option for non-interactive output."""
        result = runner.invoke(app, ["discover", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--json" in output

    def test_discover_help_shows_no_cache_option(self) -> None:
        """--help shows --no-cache option."""
        result = runner.invoke(app, ["discover", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--no-cache" in output

    def test_discover_help_shows_examples(self) -> None:
        """--help shows usage examples."""
        result = runner.invoke(app, ["discover", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "Examples:" in output
        # Verify key example patterns are shown
        assert "gpumod discover" in output
        assert "--search deepseek" in output
        assert "--author bartowski" in output
        assert "--task code" in output


# ---------------------------------------------------------------------------
# TestDiscoverJson
# ---------------------------------------------------------------------------


class TestDiscoverJson:
    """Tests for discover --json output."""

    def test_discover_json_output(
        self,
        mock_system_info: SystemInfo,
        mock_models: list[UnslothModel],
        mock_gguf_files: list[GGUFFile],
    ) -> None:
        """--json outputs valid JSON with compatible models."""
        # Only use files that fit in VRAM
        fitting_files = [f for f in mock_gguf_files if f.estimated_vram_mb <= 24064]

        with (
            patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls,
            patch("gpumod.cli_discover.UnslothModelLister") as mock_lister_cls,
            patch("gpumod.cli_discover.GGUFMetadataFetcher") as mock_fetcher_cls,
        ):
            # Setup mocks
            collector = MagicMock()
            collector.get_system_info = AsyncMock(return_value=mock_system_info)
            mock_collector_cls.return_value = collector

            lister = MagicMock()
            lister.list_models = AsyncMock(return_value=mock_models)
            mock_lister_cls.return_value = lister

            fetcher = MagicMock()
            fetcher.list_gguf_files = AsyncMock(return_value=fitting_files)
            mock_fetcher_cls.return_value = fetcher

            result = runner.invoke(app, ["discover", "--json"])

        assert result.exit_code == 0
        # Parse JSON output
        output = json.loads(result.output)
        assert isinstance(output, list)
        if output:
            assert "model" in output[0]
            assert "files" in output[0]

    def test_discover_json_no_models(
        self,
        mock_system_info: SystemInfo,
    ) -> None:
        """--json with no models returns empty list gracefully."""
        with (
            patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls,
            patch("gpumod.cli_discover.UnslothModelLister") as mock_lister_cls,
        ):
            collector = MagicMock()
            collector.get_system_info = AsyncMock(return_value=mock_system_info)
            mock_collector_cls.return_value = collector

            lister = MagicMock()
            lister.list_models = AsyncMock(return_value=[])
            mock_lister_cls.return_value = lister

            result = runner.invoke(app, ["discover", "--json"])

        # Should exit cleanly when no models found
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# TestDiscoverVramFilter
# ---------------------------------------------------------------------------


class TestDiscoverVramFilter:
    """Tests for VRAM filtering."""

    def test_discover_filters_by_vram(
        self,
        mock_system_info: SystemInfo,
        mock_models: list[UnslothModel],
    ) -> None:
        """Models exceeding VRAM budget are filtered out."""
        # Create files that don't fit
        large_files = [
            GGUFFile(
                filename="model-Q8_0.gguf",
                size_bytes=50_000_000_000,
                quant_type="Q8_0",
                estimated_vram_mb=52000,  # Exceeds 24GB
                is_split=False,
                split_parts=1,
            ),
        ]

        with (
            patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls,
            patch("gpumod.cli_discover.UnslothModelLister") as mock_lister_cls,
            patch("gpumod.cli_discover.GGUFMetadataFetcher") as mock_fetcher_cls,
        ):
            collector = MagicMock()
            collector.get_system_info = AsyncMock(return_value=mock_system_info)
            mock_collector_cls.return_value = collector

            lister = MagicMock()
            lister.list_models = AsyncMock(return_value=mock_models)
            mock_lister_cls.return_value = lister

            fetcher = MagicMock()
            fetcher.list_gguf_files = AsyncMock(return_value=large_files)
            mock_fetcher_cls.return_value = fetcher

            result = runner.invoke(app, ["discover", "--json"])

        # Should show no compatible models message
        assert result.exit_code == 0

    def test_discover_custom_vram_budget(
        self,
        mock_system_info: SystemInfo,
        mock_models: list[UnslothModel],
        mock_gguf_files: list[GGUFFile],
    ) -> None:
        """--vram overrides detected VRAM budget."""
        with (
            patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls,
            patch("gpumod.cli_discover.UnslothModelLister") as mock_lister_cls,
            patch("gpumod.cli_discover.GGUFMetadataFetcher") as mock_fetcher_cls,
        ):
            collector = MagicMock()
            collector.get_system_info = AsyncMock(return_value=mock_system_info)
            mock_collector_cls.return_value = collector

            lister = MagicMock()
            lister.list_models = AsyncMock(return_value=mock_models)
            mock_lister_cls.return_value = lister

            fetcher = MagicMock()
            fetcher.list_gguf_files = AsyncMock(return_value=mock_gguf_files)
            mock_fetcher_cls.return_value = fetcher

            # Use small VRAM budget that filters out all files
            result = runner.invoke(app, ["discover", "--vram", "1000", "--json"])

        # Should show no compatible models (all exceed 1GB)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# TestDiscoverTaskFilter
# ---------------------------------------------------------------------------


class TestDiscoverTaskFilter:
    """Tests for task filtering."""

    def test_discover_task_filter(
        self,
        mock_system_info: SystemInfo,
        mock_gguf_files: list[GGUFFile],
    ) -> None:
        """--task filters models by task type."""
        code_model = UnslothModel(
            repo_id="unsloth/Qwen3-Coder-GGUF",
            name="Qwen3 Coder",
            description="A code generation model",
            last_modified=datetime.now(tz=UTC),
            tags=("gguf", "code"),
            has_gguf=True,
        )

        with (
            patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls,
            patch("gpumod.cli_discover.UnslothModelLister") as mock_lister_cls,
            patch("gpumod.cli_discover.GGUFMetadataFetcher") as mock_fetcher_cls,
        ):
            collector = MagicMock()
            collector.get_system_info = AsyncMock(return_value=mock_system_info)
            mock_collector_cls.return_value = collector

            lister = MagicMock()
            lister.list_models = AsyncMock(return_value=[code_model])
            mock_lister_cls.return_value = lister

            fetcher = MagicMock()
            fitting_files = [f for f in mock_gguf_files if f.estimated_vram_mb <= 24064]
            fetcher.list_gguf_files = AsyncMock(return_value=fitting_files)
            mock_fetcher_cls.return_value = fetcher

            result = runner.invoke(app, ["discover", "--task", "code", "--json"])

        assert result.exit_code == 0
        # Verify list_models was called with task filter
        lister.list_models.assert_awaited_once()
        call_kwargs = lister.list_models.call_args
        assert call_kwargs.kwargs.get("task") == "code"


# ---------------------------------------------------------------------------
# TestDiscoverErrors
# ---------------------------------------------------------------------------


class TestDiscoverErrors:
    """Tests for error handling."""

    def test_discover_nvidia_unavailable(self) -> None:
        """Shows error when nvidia-smi unavailable."""
        from gpumod.discovery.system_info import NvidiaSmiUnavailableError

        with patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls:
            collector = MagicMock()
            collector.get_system_info = AsyncMock(
                side_effect=NvidiaSmiUnavailableError("nvidia-smi not found")
            )
            mock_collector_cls.return_value = collector

            result = runner.invoke(app, ["discover"])

        assert result.exit_code == 1
        assert "nvidia-smi" in result.output.lower()

    def test_discover_hf_api_error(
        self,
        mock_system_info: SystemInfo,
    ) -> None:
        """Shows error when HuggingFace API fails."""
        from gpumod.discovery.unsloth_lister import HuggingFaceAPIError

        with (
            patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls,
            patch("gpumod.cli_discover.UnslothModelLister") as mock_lister_cls,
        ):
            collector = MagicMock()
            collector.get_system_info = AsyncMock(return_value=mock_system_info)
            mock_collector_cls.return_value = collector

            lister = MagicMock()
            lister.list_models = AsyncMock(side_effect=HuggingFaceAPIError("API rate limited"))
            mock_lister_cls.return_value = lister

            result = runner.invoke(app, ["discover"])

        assert result.exit_code == 1
        assert "error" in result.output.lower()


# ---------------------------------------------------------------------------
# TestDiscoverDryRun
# ---------------------------------------------------------------------------


class TestDiscoverDryRun:
    """Tests for --dry-run mode."""

    def test_discover_dry_run_no_write(
        self,
        mock_system_info: SystemInfo,
        mock_models: list[UnslothModel],
        mock_gguf_files: list[GGUFFile],
    ) -> None:
        """--dry-run shows preview without writing file."""
        fitting_files = [f for f in mock_gguf_files if f.estimated_vram_mb <= 24064]

        with (
            patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls,
            patch("gpumod.cli_discover.UnslothModelLister") as mock_lister_cls,
            patch("gpumod.cli_discover.GGUFMetadataFetcher") as mock_fetcher_cls,
            patch("gpumod.cli_discover.IntPrompt") as mock_prompt,
            patch("gpumod.cli_discover.Path") as mock_path,
        ):
            collector = MagicMock()
            collector.get_system_info = AsyncMock(return_value=mock_system_info)
            mock_collector_cls.return_value = collector

            lister = MagicMock()
            lister.list_models = AsyncMock(return_value=mock_models)
            mock_lister_cls.return_value = lister

            fetcher = MagicMock()
            fetcher.list_gguf_files = AsyncMock(return_value=fitting_files)
            mock_fetcher_cls.return_value = fetcher

            # User selects first model
            mock_prompt.ask.return_value = 1

            result = runner.invoke(app, ["discover", "--dry-run"])

        assert result.exit_code == 0
        assert "dry-run" in result.output.lower()
        # Should NOT write file in dry-run mode
        mock_path.return_value.write_text.assert_not_called()


# ---------------------------------------------------------------------------
# TestDiscoverSearchOption
# ---------------------------------------------------------------------------


class TestDiscoverSearchOption:
    """Tests for --search option."""

    def test_discover_search_option(
        self,
        mock_system_info: SystemInfo,
        mock_models: list[UnslothModel],
        mock_gguf_files: list[GGUFFile],
    ) -> None:
        """--search passes search query to model lister."""
        fitting_files = [f for f in mock_gguf_files if f.estimated_vram_mb <= 24064]

        with (
            patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls,
            patch("gpumod.cli_discover.UnslothModelLister") as mock_lister_cls,
            patch("gpumod.cli_discover.GGUFMetadataFetcher") as mock_fetcher_cls,
        ):
            collector = MagicMock()
            collector.get_system_info = AsyncMock(return_value=mock_system_info)
            mock_collector_cls.return_value = collector

            lister = MagicMock()
            lister.list_models = AsyncMock(return_value=mock_models)
            mock_lister_cls.return_value = lister

            fetcher = MagicMock()
            fetcher.list_gguf_files = AsyncMock(return_value=fitting_files)
            mock_fetcher_cls.return_value = fetcher

            result = runner.invoke(app, ["discover", "--search", "deepseek", "--json"])

        assert result.exit_code == 0
        # Verify list_models was called with search parameter
        lister.list_models.assert_awaited_once()
        call_kwargs = lister.list_models.call_args
        assert call_kwargs.kwargs.get("search") == "deepseek"

    def test_discover_search_short_flag(
        self,
        mock_system_info: SystemInfo,
        mock_models: list[UnslothModel],
        mock_gguf_files: list[GGUFFile],
    ) -> None:
        """Short flag -s works for search."""
        fitting_files = [f for f in mock_gguf_files if f.estimated_vram_mb <= 24064]

        with (
            patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls,
            patch("gpumod.cli_discover.UnslothModelLister") as mock_lister_cls,
            patch("gpumod.cli_discover.GGUFMetadataFetcher") as mock_fetcher_cls,
        ):
            collector = MagicMock()
            collector.get_system_info = AsyncMock(return_value=mock_system_info)
            mock_collector_cls.return_value = collector

            lister = MagicMock()
            lister.list_models = AsyncMock(return_value=mock_models)
            mock_lister_cls.return_value = lister

            fetcher = MagicMock()
            fetcher.list_gguf_files = AsyncMock(return_value=fitting_files)
            mock_fetcher_cls.return_value = fetcher

            result = runner.invoke(app, ["discover", "-s", "kimi", "--json"])

        assert result.exit_code == 0
        call_kwargs = lister.list_models.call_args
        assert call_kwargs.kwargs.get("search") == "kimi"


# ---------------------------------------------------------------------------
# TestDiscoverAuthorOption
# ---------------------------------------------------------------------------


class TestDiscoverAuthorOption:
    """Tests for --author option."""

    def test_discover_author_option(
        self,
        mock_system_info: SystemInfo,
        mock_models: list[UnslothModel],
        mock_gguf_files: list[GGUFFile],
    ) -> None:
        """--author sets the HuggingFace organization."""
        fitting_files = [f for f in mock_gguf_files if f.estimated_vram_mb <= 24064]

        with (
            patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls,
            patch("gpumod.cli_discover.UnslothModelLister") as mock_lister_cls,
            patch("gpumod.cli_discover.GGUFMetadataFetcher") as mock_fetcher_cls,
        ):
            collector = MagicMock()
            collector.get_system_info = AsyncMock(return_value=mock_system_info)
            mock_collector_cls.return_value = collector

            lister = MagicMock()
            lister.list_models = AsyncMock(return_value=mock_models)
            mock_lister_cls.return_value = lister

            fetcher = MagicMock()
            fetcher.list_gguf_files = AsyncMock(return_value=fitting_files)
            mock_fetcher_cls.return_value = fetcher

            result = runner.invoke(app, ["discover", "--author", "bartowski", "--json"])

        assert result.exit_code == 0
        # Verify UnslothModelLister was initialized with author
        mock_lister_cls.assert_called_once_with(author="bartowski")

    def test_discover_author_short_flag(
        self,
        mock_system_info: SystemInfo,
        mock_models: list[UnslothModel],
        mock_gguf_files: list[GGUFFile],
    ) -> None:
        """Short flag -a works for author."""
        fitting_files = [f for f in mock_gguf_files if f.estimated_vram_mb <= 24064]

        with (
            patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls,
            patch("gpumod.cli_discover.UnslothModelLister") as mock_lister_cls,
            patch("gpumod.cli_discover.GGUFMetadataFetcher") as mock_fetcher_cls,
        ):
            collector = MagicMock()
            collector.get_system_info = AsyncMock(return_value=mock_system_info)
            mock_collector_cls.return_value = collector

            lister = MagicMock()
            lister.list_models = AsyncMock(return_value=mock_models)
            mock_lister_cls.return_value = lister

            fetcher = MagicMock()
            fetcher.list_gguf_files = AsyncMock(return_value=fitting_files)
            mock_fetcher_cls.return_value = fetcher

            result = runner.invoke(app, ["discover", "-a", "deepseek-ai", "--json"])

        assert result.exit_code == 0
        mock_lister_cls.assert_called_once_with(author="deepseek-ai")

    def test_discover_author_default_is_unsloth(
        self,
        mock_system_info: SystemInfo,
        mock_models: list[UnslothModel],
        mock_gguf_files: list[GGUFFile],
    ) -> None:
        """Default author is unsloth."""
        fitting_files = [f for f in mock_gguf_files if f.estimated_vram_mb <= 24064]

        with (
            patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls,
            patch("gpumod.cli_discover.UnslothModelLister") as mock_lister_cls,
            patch("gpumod.cli_discover.GGUFMetadataFetcher") as mock_fetcher_cls,
        ):
            collector = MagicMock()
            collector.get_system_info = AsyncMock(return_value=mock_system_info)
            mock_collector_cls.return_value = collector

            lister = MagicMock()
            lister.list_models = AsyncMock(return_value=mock_models)
            mock_lister_cls.return_value = lister

            fetcher = MagicMock()
            fetcher.list_gguf_files = AsyncMock(return_value=fitting_files)
            mock_fetcher_cls.return_value = fetcher

            result = runner.invoke(app, ["discover", "--json"])

        assert result.exit_code == 0
        # Default author should be unsloth
        mock_lister_cls.assert_called_once_with(author="unsloth")


# ---------------------------------------------------------------------------
# TestDiscoverCombinedOptions
# ---------------------------------------------------------------------------


class TestDiscoverCombinedOptions:
    """Tests for combined option usage."""

    def test_discover_search_with_author(
        self,
        mock_system_info: SystemInfo,
        mock_models: list[UnslothModel],
        mock_gguf_files: list[GGUFFile],
    ) -> None:
        """--search and --author work together."""
        fitting_files = [f for f in mock_gguf_files if f.estimated_vram_mb <= 24064]

        with (
            patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls,
            patch("gpumod.cli_discover.UnslothModelLister") as mock_lister_cls,
            patch("gpumod.cli_discover.GGUFMetadataFetcher") as mock_fetcher_cls,
        ):
            collector = MagicMock()
            collector.get_system_info = AsyncMock(return_value=mock_system_info)
            mock_collector_cls.return_value = collector

            lister = MagicMock()
            lister.list_models = AsyncMock(return_value=mock_models)
            mock_lister_cls.return_value = lister

            fetcher = MagicMock()
            fetcher.list_gguf_files = AsyncMock(return_value=fitting_files)
            mock_fetcher_cls.return_value = fetcher

            result = runner.invoke(
                app,
                ["discover", "--author", "bartowski", "--search", "llama", "--json"],
            )

        assert result.exit_code == 0
        # Verify author was set
        mock_lister_cls.assert_called_once_with(author="bartowski")
        # Verify search was passed
        call_kwargs = lister.list_models.call_args
        assert call_kwargs.kwargs.get("search") == "llama"

    def test_discover_search_author_task(
        self,
        mock_system_info: SystemInfo,
        mock_gguf_files: list[GGUFFile],
    ) -> None:
        """All three filters work together: --search --author --task."""
        code_model = UnslothModel(
            repo_id="bartowski/Qwen3-Coder-GGUF",
            name="Qwen3 Coder",
            description="A code generation model",
            last_modified=datetime.now(tz=UTC),
            tags=("gguf", "code"),
            has_gguf=True,
        )
        fitting_files = [f for f in mock_gguf_files if f.estimated_vram_mb <= 24064]

        with (
            patch("gpumod.cli_discover.SystemInfoCollector") as mock_collector_cls,
            patch("gpumod.cli_discover.UnslothModelLister") as mock_lister_cls,
            patch("gpumod.cli_discover.GGUFMetadataFetcher") as mock_fetcher_cls,
        ):
            collector = MagicMock()
            collector.get_system_info = AsyncMock(return_value=mock_system_info)
            mock_collector_cls.return_value = collector

            lister = MagicMock()
            lister.list_models = AsyncMock(return_value=[code_model])
            mock_lister_cls.return_value = lister

            fetcher = MagicMock()
            fetcher.list_gguf_files = AsyncMock(return_value=fitting_files)
            mock_fetcher_cls.return_value = fetcher

            result = runner.invoke(
                app,
                [
                    "discover",
                    "--author",
                    "bartowski",
                    "--search",
                    "qwen",
                    "--task",
                    "code",
                    "--json",
                ],
            )

        assert result.exit_code == 0
        mock_lister_cls.assert_called_once_with(author="bartowski")
        call_kwargs = lister.list_models.call_args
        assert call_kwargs.kwargs.get("search") == "qwen"
        assert call_kwargs.kwargs.get("task") == "code"


# ---------------------------------------------------------------------------
# TestVramFormatting
# ---------------------------------------------------------------------------


class TestVramFormatting:
    """Tests for VRAM display formatting."""

    def test_format_vram_small_mb(self) -> None:
        """Small values should display in MB."""
        assert _format_vram(512) == "512 MB"
        assert _format_vram(100) == "100 MB"
        assert _format_vram(1) == "1 MB"

    def test_format_vram_large_gb(self) -> None:
        """Large values (>= 1024 MB) should display in GB."""
        assert _format_vram(1024) == "1.0 GB"
        assert _format_vram(2048) == "2.0 GB"
        assert _format_vram(24000) == "23.4 GB"

    def test_format_vram_fractional_gb(self) -> None:
        """Fractional GB values should show one decimal place."""
        assert _format_vram(1536) == "1.5 GB"
        assert _format_vram(17276) == "16.9 GB"
        assert _format_vram(23217) == "22.7 GB"

    def test_format_vram_boundary(self) -> None:
        """Test boundary between MB and GB display."""
        assert _format_vram(1023) == "1023 MB"
        assert _format_vram(1024) == "1.0 GB"

    def test_format_vram_zero(self) -> None:
        """Zero VRAM should display as MB."""
        assert _format_vram(0) == "0 MB"
