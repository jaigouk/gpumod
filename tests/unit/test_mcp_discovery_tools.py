"""Tests for MCP discovery tools — search_hf_models, list_gguf_files, generate_preset.

TDD: RED phase - these tests should FAIL until the tools are implemented.
Follows SOLID principles:
- SRP: Each test class covers one tool
- OCP: Tests use fixtures for extensibility
- LSP: Mock objects substitute real dependencies
- ISP: Tools have focused, single-purpose interfaces
- DIP: Tests depend on abstractions (interfaces), not concrete implementations
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpumod.discovery.gguf_metadata import GGUFFile
from gpumod.discovery.unsloth_lister import UnslothModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_models() -> list[UnslothModel]:
    """Sample HuggingFace models for testing."""
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
            repo_id="unsloth/DeepSeek-R1-GGUF",
            name="DeepSeek R1",
            description="Reasoning model",
            last_modified=datetime.now(tz=UTC),
            tags=("gguf", "reasoning"),
            has_gguf=True,
        ),
        UnslothModel(
            repo_id="bartowski/Llama-3.1-8B-GGUF",
            name="Llama 3.1 8B",
            description="Chat model",
            last_modified=datetime.now(tz=UTC),
            tags=("gguf", "chat"),
            has_gguf=True,
        ),
    ]


@pytest.fixture
def mock_gguf_files() -> list[GGUFFile]:
    """Sample GGUF files for testing."""
    return [
        GGUFFile(
            filename="model-Q4_K_M.gguf",
            size_bytes=8_000_000_000,
            quant_type="Q4_K_M",
            estimated_vram_mb=8500,
            is_split=False,
            split_parts=1,
        ),
        GGUFFile(
            filename="model-Q8_0.gguf",
            size_bytes=16_000_000_000,
            quant_type="Q8_0",
            estimated_vram_mb=17000,
            is_split=False,
            split_parts=1,
        ),
        GGUFFile(
            filename="model-Q2_K.gguf",
            size_bytes=4_000_000_000,
            quant_type="Q2_K",
            estimated_vram_mb=4300,
            is_split=False,
            split_parts=1,
        ),
    ]


def _make_mock_ctx() -> MagicMock:
    """Create a mock FastMCP Context (discovery tools don't need lifespan deps)."""
    ctx = MagicMock()
    ctx.fastmcp._lifespan_result = {}
    return ctx


# ---------------------------------------------------------------------------
# TestSearchHFModels - search HuggingFace for GGUF models
# ---------------------------------------------------------------------------


class TestSearchHFModels:
    """Tests for search_hf_models MCP tool."""

    async def test_search_returns_models_list(self, mock_models: list[UnslothModel]) -> None:
        """Basic search returns a list of models."""
        from gpumod.mcp_tools import search_hf_models

        with patch("gpumod.discovery.unsloth_lister.UnslothModelLister") as lister_cls:
            mock_lister = AsyncMock()
            mock_lister.list_models.return_value = mock_models
            lister_cls.return_value = mock_lister

            result = await search_hf_models(ctx=_make_mock_ctx())

            assert "models" in result
            assert len(result["models"]) == 3

    async def test_search_with_author_filter(self, mock_models: list[UnslothModel]) -> None:
        """Search with author parameter filters to that organization."""
        from gpumod.mcp_tools import search_hf_models

        with patch("gpumod.discovery.unsloth_lister.UnslothModelLister") as lister_cls:
            mock_lister = AsyncMock()
            mock_lister.list_models.return_value = mock_models[:2]  # unsloth only
            lister_cls.return_value = mock_lister

            result = await search_hf_models(author="unsloth", ctx=_make_mock_ctx())

            lister_cls.assert_called_once_with(author="unsloth", cache_ttl_seconds=3600)
            assert len(result["models"]) == 2

    async def test_search_with_keyword(self, mock_models: list[UnslothModel]) -> None:
        """Search with keyword filters by model name."""
        from gpumod.mcp_tools import search_hf_models

        with patch("gpumod.discovery.unsloth_lister.UnslothModelLister") as lister_cls:
            mock_lister = AsyncMock()
            mock_lister.list_models.return_value = [mock_models[1]]  # deepseek
            lister_cls.return_value = mock_lister

            await search_hf_models(search="deepseek", ctx=_make_mock_ctx())

            mock_lister.list_models.assert_awaited_once()
            call_kwargs = mock_lister.list_models.call_args.kwargs
            assert call_kwargs.get("search") == "deepseek"

    async def test_search_with_task_filter(self, mock_models: list[UnslothModel]) -> None:
        """Search with task parameter filters by model type."""
        from gpumod.mcp_tools import search_hf_models

        with patch("gpumod.discovery.unsloth_lister.UnslothModelLister") as lister_cls:
            mock_lister = AsyncMock()
            mock_lister.list_models.return_value = [mock_models[0]]  # code model
            lister_cls.return_value = mock_lister

            await search_hf_models(task="code", ctx=_make_mock_ctx())

            call_kwargs = mock_lister.list_models.call_args.kwargs
            assert call_kwargs.get("task") == "code"

    async def test_search_with_limit(self, mock_models: list[UnslothModel]) -> None:
        """Search with limit parameter caps results."""
        from gpumod.mcp_tools import search_hf_models

        with patch("gpumod.discovery.unsloth_lister.UnslothModelLister") as lister_cls:
            mock_lister = AsyncMock()
            mock_lister.list_models.return_value = mock_models
            lister_cls.return_value = mock_lister

            result = await search_hf_models(limit=2, ctx=_make_mock_ctx())

            assert len(result["models"]) == 2

    async def test_search_validates_limit_range(self) -> None:
        """Limit must be positive and reasonable."""
        from gpumod.mcp_tools import search_hf_models

        result = await search_hf_models(limit=0, ctx=_make_mock_ctx())
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

        result = await search_hf_models(limit=-5, ctx=_make_mock_ctx())
        assert "error" in result

    async def test_search_validates_limit_max(self) -> None:
        """Limit cannot exceed maximum (100)."""
        from gpumod.mcp_tools import search_hf_models

        result = await search_hf_models(limit=500, ctx=_make_mock_ctx())
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_search_validates_task_value(self) -> None:
        """Task must be a valid task type."""
        from gpumod.mcp_tools import search_hf_models

        result = await search_hf_models(task="invalid_task", ctx=_make_mock_ctx())
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_search_handles_api_error(self) -> None:
        """API errors are caught and returned as error response."""
        from gpumod.discovery.unsloth_lister import HuggingFaceAPIError
        from gpumod.mcp_tools import search_hf_models

        with patch("gpumod.discovery.unsloth_lister.UnslothModelLister") as lister_cls:
            mock_lister = AsyncMock()
            mock_lister.list_models.side_effect = HuggingFaceAPIError("API rate limit")
            lister_cls.return_value = mock_lister

            result = await search_hf_models(ctx=_make_mock_ctx())

            assert "error" in result
            assert "API" in result["error"] or "rate" in result["error"].lower()

    async def test_search_returns_model_fields(self, mock_models: list[UnslothModel]) -> None:
        """Result includes all essential model fields."""
        from gpumod.mcp_tools import search_hf_models

        with patch("gpumod.discovery.unsloth_lister.UnslothModelLister") as lister_cls:
            mock_lister = AsyncMock()
            mock_lister.list_models.return_value = [mock_models[0]]
            lister_cls.return_value = mock_lister

            result = await search_hf_models(ctx=_make_mock_ctx())

            model = result["models"][0]
            assert "repo_id" in model
            assert "name" in model
            assert "description" in model
            assert "tags" in model
            assert model["repo_id"] == "unsloth/Qwen3-Coder-Next-GGUF"

    async def test_search_no_cache_bypasses_cache(self, mock_models: list[UnslothModel]) -> None:
        """no_cache parameter forces fresh fetch."""
        from gpumod.mcp_tools import search_hf_models

        with patch("gpumod.discovery.unsloth_lister.UnslothModelLister") as lister_cls:
            mock_lister = AsyncMock()
            mock_lister.list_models.return_value = mock_models
            lister_cls.return_value = mock_lister

            await search_hf_models(no_cache=True, ctx=_make_mock_ctx())

            call_kwargs = mock_lister.list_models.call_args.kwargs
            assert call_kwargs.get("force_refresh") is True


# ---------------------------------------------------------------------------
# TestListGGUFFiles - get GGUF file metadata from a repo
# ---------------------------------------------------------------------------


class TestListGGUFFiles:
    """Tests for list_gguf_files MCP tool."""

    async def test_list_returns_gguf_files(self, mock_gguf_files: list[GGUFFile]) -> None:
        """Basic list returns GGUF file metadata."""
        from gpumod.mcp_tools import list_gguf_files

        with patch("gpumod.discovery.gguf_metadata.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.return_value = mock_gguf_files
            fetcher_cls.return_value = mock_fetcher

            result = await list_gguf_files(
                repo_id="unsloth/Qwen3-Coder-Next-GGUF",
                ctx=_make_mock_ctx(),
            )

            assert "files" in result
            assert len(result["files"]) == 3

    async def test_list_validates_repo_id_required(self) -> None:
        """repo_id is required."""
        from gpumod.mcp_tools import list_gguf_files

        result = await list_gguf_files(repo_id="", ctx=_make_mock_ctx())
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_list_validates_repo_id_format(self) -> None:
        """repo_id must be org/name format."""
        from gpumod.mcp_tools import list_gguf_files

        result = await list_gguf_files(repo_id="invalid", ctx=_make_mock_ctx())
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_list_handles_repo_not_found(self) -> None:
        """Non-existent repos return NOT_FOUND error."""
        from gpumod.discovery.gguf_metadata import RepoNotFoundError
        from gpumod.mcp_tools import list_gguf_files

        with patch("gpumod.discovery.gguf_metadata.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.side_effect = RepoNotFoundError("Not found")
            fetcher_cls.return_value = mock_fetcher

            result = await list_gguf_files(
                repo_id="fake/nonexistent-repo",
                ctx=_make_mock_ctx(),
            )

            assert "error" in result
            assert result["code"] == "NOT_FOUND"

    async def test_list_returns_file_fields(self, mock_gguf_files: list[GGUFFile]) -> None:
        """Result includes all essential file fields."""
        from gpumod.mcp_tools import list_gguf_files

        with patch("gpumod.discovery.gguf_metadata.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.return_value = [mock_gguf_files[0]]
            fetcher_cls.return_value = mock_fetcher

            result = await list_gguf_files(
                repo_id="unsloth/Test-GGUF",
                ctx=_make_mock_ctx(),
            )

            file = result["files"][0]
            assert "filename" in file
            assert "size_bytes" in file
            assert "quant_type" in file
            assert "estimated_vram_mb" in file
            assert "is_split" in file

    async def test_list_with_vram_budget_filter(self, mock_gguf_files: list[GGUFFile]) -> None:
        """vram_budget_mb filters to files that fit."""
        from gpumod.mcp_tools import list_gguf_files

        with patch("gpumod.discovery.gguf_metadata.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.return_value = mock_gguf_files
            fetcher_cls.return_value = mock_fetcher

            # Only Q2_K (4300 MB) and Q4_K_M (8500 MB) should fit in 10GB
            result = await list_gguf_files(
                repo_id="unsloth/Test-GGUF",
                vram_budget_mb=10000,
                ctx=_make_mock_ctx(),
            )

            assert len(result["files"]) == 2
            for f in result["files"]:
                assert f["estimated_vram_mb"] <= 10000

    async def test_list_validates_vram_budget_positive(self) -> None:
        """vram_budget_mb must be positive."""
        from gpumod.mcp_tools import list_gguf_files

        result = await list_gguf_files(
            repo_id="unsloth/Test-GGUF",
            vram_budget_mb=-1000,
            ctx=_make_mock_ctx(),
        )
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_list_includes_repo_metadata(self, mock_gguf_files: list[GGUFFile]) -> None:
        """Result includes repo_id in response."""
        from gpumod.mcp_tools import list_gguf_files

        with patch("gpumod.discovery.gguf_metadata.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.return_value = mock_gguf_files
            fetcher_cls.return_value = mock_fetcher

            result = await list_gguf_files(
                repo_id="unsloth/Test-GGUF",
                ctx=_make_mock_ctx(),
            )

            assert result["repo_id"] == "unsloth/Test-GGUF"

    async def test_list_empty_repo_returns_empty_list(self) -> None:
        """Repo with no GGUF files returns empty list."""
        from gpumod.mcp_tools import list_gguf_files

        with patch("gpumod.discovery.gguf_metadata.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.return_value = []
            fetcher_cls.return_value = mock_fetcher

            result = await list_gguf_files(
                repo_id="unsloth/Empty-Repo",
                ctx=_make_mock_ctx(),
            )

            assert result["files"] == []
            assert result["count"] == 0


# ---------------------------------------------------------------------------
# TestGeneratePreset - generate preset YAML for a model (optional tool)
# ---------------------------------------------------------------------------


class TestGeneratePreset:
    """Tests for generate_preset MCP tool (optional feature)."""

    async def test_generate_returns_yaml_string(self) -> None:
        """Generate returns valid YAML preset string."""
        from gpumod.mcp_tools import generate_preset

        result = await generate_preset(
            repo_id="unsloth/Qwen3-Coder-Next-GGUF",
            gguf_file="model-Q4_K_M.gguf",
            ctx=_make_mock_ctx(),
        )

        assert "preset" in result
        assert "yaml" in result or isinstance(result["preset"], str)
        assert "llama.cpp" in result["preset"].lower() or "llamacpp" in result["preset"].lower()

    async def test_generate_validates_repo_id(self) -> None:
        """repo_id must be valid format."""
        from gpumod.mcp_tools import generate_preset

        result = await generate_preset(
            repo_id="",
            gguf_file="model.gguf",
            ctx=_make_mock_ctx(),
        )
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_generate_validates_gguf_file(self) -> None:
        """gguf_file must end with .gguf."""
        from gpumod.mcp_tools import generate_preset

        result = await generate_preset(
            repo_id="unsloth/Test-GGUF",
            gguf_file="model.bin",
            ctx=_make_mock_ctx(),
        )
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_generate_with_context_size(self) -> None:
        """context_size parameter is included in preset."""
        from gpumod.mcp_tools import generate_preset

        result = await generate_preset(
            repo_id="unsloth/Test-GGUF",
            gguf_file="model-Q4_K_M.gguf",
            context_size=32768,
            ctx=_make_mock_ctx(),
        )

        assert "32768" in result["preset"] or "32k" in result["preset"].lower()

    async def test_generate_with_service_id(self) -> None:
        """service_id parameter customizes the service name."""
        from gpumod.mcp_tools import generate_preset

        result = await generate_preset(
            repo_id="unsloth/Test-GGUF",
            gguf_file="model-Q4_K_M.gguf",
            service_id="my-custom-service",
            ctx=_make_mock_ctx(),
        )

        assert "my-custom-service" in result["preset"]

    async def test_generate_validates_context_size_range(self) -> None:
        """context_size must be within valid range."""
        from gpumod.mcp_tools import generate_preset

        result = await generate_preset(
            repo_id="unsloth/Test-GGUF",
            gguf_file="model.gguf",
            context_size=0,
            ctx=_make_mock_ctx(),
        )
        assert "error" in result

        result = await generate_preset(
            repo_id="unsloth/Test-GGUF",
            gguf_file="model.gguf",
            context_size=1000000,  # Too large
            ctx=_make_mock_ctx(),
        )
        assert "error" in result

    async def test_generate_includes_model_path(self) -> None:
        """Preset includes correct model path from repo."""
        from gpumod.mcp_tools import generate_preset

        result = await generate_preset(
            repo_id="unsloth/Qwen3-Coder-Next-GGUF",
            gguf_file="model-Q4_K_M.gguf",
            ctx=_make_mock_ctx(),
        )

        # Should include HuggingFace path reference
        assert "unsloth/Qwen3-Coder-Next-GGUF" in result["preset"]
        assert "model-Q4_K_M.gguf" in result["preset"]


# ---------------------------------------------------------------------------
# TestDiscoveryToolsIntegration - cross-tool scenarios
# ---------------------------------------------------------------------------


class TestDiscoveryToolsIntegration:
    """Integration tests for discovery tool workflows."""

    async def test_search_then_list_workflow(
        self,
        mock_models: list[UnslothModel],
        mock_gguf_files: list[GGUFFile],
    ) -> None:
        """Typical workflow: search models, then list files for chosen repo."""
        from gpumod.mcp_tools import list_gguf_files, search_hf_models

        # Step 1: Search for models
        with patch("gpumod.discovery.unsloth_lister.UnslothModelLister") as lister_cls:
            mock_lister = AsyncMock()
            mock_lister.list_models.return_value = mock_models
            lister_cls.return_value = mock_lister

            search_result = await search_hf_models(
                search="qwen",
                ctx=_make_mock_ctx(),
            )

        assert len(search_result["models"]) > 0
        chosen_repo = search_result["models"][0]["repo_id"]

        # Step 2: List GGUF files for chosen repo
        with patch("gpumod.discovery.gguf_metadata.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.return_value = mock_gguf_files
            fetcher_cls.return_value = mock_fetcher

            files_result = await list_gguf_files(
                repo_id=chosen_repo,
                ctx=_make_mock_ctx(),
            )

        assert len(files_result["files"]) > 0
        assert files_result["repo_id"] == chosen_repo

    async def test_list_then_generate_workflow(
        self,
        mock_gguf_files: list[GGUFFile],
    ) -> None:
        """Workflow: list files, then generate preset for chosen file."""
        from gpumod.mcp_tools import generate_preset, list_gguf_files

        # Step 1: List GGUF files
        with patch("gpumod.discovery.gguf_metadata.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.return_value = mock_gguf_files
            fetcher_cls.return_value = mock_fetcher

            files_result = await list_gguf_files(
                repo_id="unsloth/Test-GGUF",
                vram_budget_mb=10000,
                ctx=_make_mock_ctx(),
            )

        # Pick smallest file that fits
        chosen_file = files_result["files"][0]["filename"]

        # Step 2: Generate preset
        preset_result = await generate_preset(
            repo_id="unsloth/Test-GGUF",
            gguf_file=chosen_file,
            ctx=_make_mock_ctx(),
        )

        assert "preset" in preset_result
        assert chosen_file in preset_result["preset"]
