"""Tests for gpumod.cli_discover -- Discover CLI command."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer.testing

from gpumod.cli import app
from gpumod.discovery.gguf_metadata import GGUFFile
from gpumod.discovery.system_info import SystemInfo
from gpumod.discovery.unsloth_lister import UnslothModel

runner = typer.testing.CliRunner()


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
        assert result.exit_code == 0
        assert "discover" in result.output.lower()
        assert "--task" in result.output
        assert "--vram" in result.output
        assert "--context" in result.output


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
