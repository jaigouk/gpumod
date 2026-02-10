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


class TestSearchHFModelsDriverFilter:
    """Tests for driver parameter in search_hf_models MCP tool."""

    async def test_search_with_driver_llamacpp(self) -> None:
        """driver='llamacpp' filters to GGUF models only."""
        from gpumod.discovery.protocols import SearchResult
        from gpumod.mcp_tools import search_hf_models

        gguf_result = SearchResult(
            repo_id="user/model-GGUF",
            name="Model GGUF",
            description="GGUF model",
            last_modified=datetime.now(tz=UTC),
            tags=("gguf",),
            model_format="gguf",
            driver_hint="llamacpp",
        )

        with patch("gpumod.mcp_tools.HuggingFaceSearcher") as searcher_cls:
            mock_searcher = AsyncMock()
            mock_searcher.search.return_value = [gguf_result]
            searcher_cls.return_value = mock_searcher

            result = await search_hf_models(
                search="model", driver="llamacpp", ctx=_make_mock_ctx()
            )

            mock_searcher.search.assert_awaited_once()
            call_kwargs = mock_searcher.search.call_args.kwargs
            assert call_kwargs.get("driver") == "llamacpp"
            assert "models" in result
            assert all(m.get("driver_hint") == "llamacpp" for m in result["models"])

    async def test_search_with_driver_vllm(self) -> None:
        """driver='vllm' filters to Safetensors models only."""
        from gpumod.discovery.protocols import SearchResult
        from gpumod.mcp_tools import search_hf_models

        vllm_result = SearchResult(
            repo_id="user/model",
            name="Model",
            description="Safetensors model",
            last_modified=datetime.now(tz=UTC),
            tags=("safetensors", "transformers"),
            model_format="safetensors",
            driver_hint="vllm",
        )

        with patch("gpumod.mcp_tools.HuggingFaceSearcher") as searcher_cls:
            mock_searcher = AsyncMock()
            mock_searcher.search.return_value = [vllm_result]
            searcher_cls.return_value = mock_searcher

            result = await search_hf_models(search="model", driver="vllm", ctx=_make_mock_ctx())

            call_kwargs = mock_searcher.search.call_args.kwargs
            assert call_kwargs.get("driver") == "vllm"
            assert all(m.get("driver_hint") == "vllm" for m in result["models"])

    async def test_search_validates_driver_value(self) -> None:
        """driver must be a valid driver type."""
        from gpumod.mcp_tools import search_hf_models

        result = await search_hf_models(driver="invalid_driver", ctx=_make_mock_ctx())
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_search_returns_model_format_field(self) -> None:
        """Results include model_format field when driver param used."""
        from gpumod.discovery.protocols import SearchResult
        from gpumod.mcp_tools import search_hf_models

        result_model = SearchResult(
            repo_id="user/model-GGUF",
            name="Model",
            description=None,
            last_modified=datetime.now(tz=UTC),
            tags=("gguf",),
            model_format="gguf",
            driver_hint="llamacpp",
        )

        with patch("gpumod.mcp_tools.HuggingFaceSearcher") as searcher_cls:
            mock_searcher = AsyncMock()
            mock_searcher.search.return_value = [result_model]
            searcher_cls.return_value = mock_searcher

            result = await search_hf_models(
                search="model", driver="llamacpp", ctx=_make_mock_ctx()
            )

            model = result["models"][0]
            assert "model_format" in model
            assert model["model_format"] == "gguf"

    async def test_search_driver_any_returns_all_formats(self) -> None:
        """driver='any' returns both GGUF and Safetensors models."""
        from gpumod.discovery.protocols import SearchResult
        from gpumod.mcp_tools import search_hf_models

        models = [
            SearchResult(
                repo_id="user/model-GGUF",
                name="GGUF Model",
                description=None,
                last_modified=datetime.now(tz=UTC),
                tags=("gguf",),
                model_format="gguf",
                driver_hint="llamacpp",
            ),
            SearchResult(
                repo_id="user/model",
                name="Safetensors Model",
                description=None,
                last_modified=datetime.now(tz=UTC),
                tags=("safetensors",),
                model_format="safetensors",
                driver_hint="vllm",
            ),
        ]

        with patch("gpumod.mcp_tools.HuggingFaceSearcher") as searcher_cls:
            mock_searcher = AsyncMock()
            mock_searcher.search.return_value = models
            searcher_cls.return_value = mock_searcher

            result = await search_hf_models(search="model", driver="any", ctx=_make_mock_ctx())

            call_kwargs = mock_searcher.search.call_args.kwargs
            assert call_kwargs.get("driver") == "any"
            assert len(result["models"]) == 2


# ---------------------------------------------------------------------------
# TestListGGUFFiles - get GGUF file metadata from a repo
# ---------------------------------------------------------------------------


class TestListGGUFFiles:
    """Tests for list_gguf_files MCP tool."""

    async def test_list_returns_gguf_files(self, mock_gguf_files: list[GGUFFile]) -> None:
        """Basic list returns GGUF file metadata."""
        from gpumod.mcp_tools import list_gguf_files

        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
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

        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
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

        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
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

        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
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

        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
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

        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
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
# TestListModelFiles - get model files from a repo (unified GGUF + Safetensors)
# ---------------------------------------------------------------------------


class TestListModelFiles:
    """Tests for list_model_files MCP tool (unified format support)."""

    async def test_list_model_files_exists(self) -> None:
        """list_model_files function is importable."""
        from gpumod.mcp_tools import list_model_files

        assert list_model_files is not None

    async def test_list_model_files_gguf(self, mock_gguf_files: list[GGUFFile]) -> None:
        """list_model_files returns GGUF files with format field."""
        from gpumod.mcp_tools import list_model_files

        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.return_value = mock_gguf_files
            fetcher_cls.return_value = mock_fetcher

            result = await list_model_files(
                repo_id="unsloth/Test-GGUF",
                ctx=_make_mock_ctx(),
            )

        assert "files" in result
        assert "model_format" in result
        assert result["model_format"] == "gguf"
        assert len(result["files"]) == 3

    async def test_list_model_files_includes_driver_hint(
        self, mock_gguf_files: list[GGUFFile]
    ) -> None:
        """list_model_files includes driver_hint for GGUF."""
        from gpumod.mcp_tools import list_model_files

        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.return_value = mock_gguf_files
            fetcher_cls.return_value = mock_fetcher

            result = await list_model_files(
                repo_id="unsloth/Test-GGUF",
                ctx=_make_mock_ctx(),
            )

        assert "driver_hint" in result
        assert result["driver_hint"] == "llamacpp"

    async def test_list_model_files_validates_repo_id(self) -> None:
        """repo_id must be valid format."""
        from gpumod.mcp_tools import list_model_files

        result = await list_model_files(
            repo_id="invalid",
            ctx=_make_mock_ctx(),
        )
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_list_model_files_vram_budget(self, mock_gguf_files: list[GGUFFile]) -> None:
        """vram_budget_mb filters files that fit."""
        from gpumod.mcp_tools import list_model_files

        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.return_value = mock_gguf_files
            fetcher_cls.return_value = mock_fetcher

            result = await list_model_files(
                repo_id="unsloth/Test-GGUF",
                vram_budget_mb=10000,
                ctx=_make_mock_ctx(),
            )

        # Only Q2_K (4300) and Q4_K_M (8500) fit in 10GB
        assert len(result["files"]) == 2


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
        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
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
        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
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


# ---------------------------------------------------------------------------
# Edge Case Tests - Robustness and boundary conditions
# ---------------------------------------------------------------------------


class TestSearchHFModelsEdgeCases:
    """Edge case tests for search_hf_models robustness."""

    async def test_search_empty_query_with_driver(self) -> None:
        """Empty search query with driver param should work."""
        from gpumod.discovery.protocols import SearchResult
        from gpumod.mcp_tools import search_hf_models

        result_model = SearchResult(
            repo_id="user/model-GGUF",
            name="Model",
            description=None,
            last_modified=datetime.now(tz=UTC),
            tags=("gguf",),
            model_format="gguf",
            driver_hint="llamacpp",
        )

        with patch("gpumod.mcp_tools.HuggingFaceSearcher") as searcher_cls:
            mock_searcher = AsyncMock()
            mock_searcher.search.return_value = [result_model]
            searcher_cls.return_value = mock_searcher

            result = await search_hf_models(search="", driver="llamacpp", ctx=_make_mock_ctx())

            assert "models" in result
            assert result["count"] >= 0

    async def test_search_combined_filters(self) -> None:
        """Combining author + driver + search should work."""
        from gpumod.discovery.protocols import SearchResult
        from gpumod.mcp_tools import search_hf_models

        result_model = SearchResult(
            repo_id="unsloth/model-GGUF",
            name="Model",
            description=None,
            last_modified=datetime.now(tz=UTC),
            tags=("gguf",),
            model_format="gguf",
            driver_hint="llamacpp",
        )

        with patch("gpumod.mcp_tools.HuggingFaceSearcher") as searcher_cls:
            mock_searcher = AsyncMock()
            mock_searcher.search.return_value = [result_model]
            searcher_cls.return_value = mock_searcher

            await search_hf_models(
                search="qwen",
                author="unsloth",
                driver="llamacpp",
                limit=10,
                ctx=_make_mock_ctx(),
            )

            call_kwargs = mock_searcher.search.call_args.kwargs
            assert call_kwargs.get("author") == "unsloth"
            assert call_kwargs.get("driver") == "llamacpp"
            assert call_kwargs.get("query") == "qwen"

    async def test_search_unicode_query(self) -> None:
        """Unicode characters in search query should not crash."""
        from gpumod.mcp_tools import search_hf_models

        with patch("gpumod.discovery.unsloth_lister.UnslothModelLister") as lister_cls:
            mock_lister = AsyncMock()
            mock_lister.list_models.return_value = []
            lister_cls.return_value = mock_lister

            result = await search_hf_models(search="模型 モデル 🤖", ctx=_make_mock_ctx())

            assert "models" in result or "error" in result

    async def test_search_very_long_query(self) -> None:
        """Very long search queries should not crash."""
        from gpumod.mcp_tools import search_hf_models

        with patch("gpumod.discovery.unsloth_lister.UnslothModelLister") as lister_cls:
            mock_lister = AsyncMock()
            mock_lister.list_models.return_value = []
            lister_cls.return_value = mock_lister

            long_query = "a" * 1000
            result = await search_hf_models(search=long_query, ctx=_make_mock_ctx())

            assert "models" in result or "error" in result

    async def test_search_empty_results_from_api(self) -> None:
        """Empty results from API should return empty list, not error."""
        from gpumod.mcp_tools import search_hf_models

        with patch("gpumod.mcp_tools.HuggingFaceSearcher") as searcher_cls:
            mock_searcher = AsyncMock()
            mock_searcher.search.return_value = []
            searcher_cls.return_value = mock_searcher

            result = await search_hf_models(
                search="nonexistent-model-xyz", driver="llamacpp", ctx=_make_mock_ctx()
            )

            assert "models" in result
            assert result["count"] == 0
            assert result["models"] == []

    async def test_search_special_characters_in_author(self) -> None:
        """Special characters in author should be handled."""
        from gpumod.mcp_tools import search_hf_models

        with patch("gpumod.discovery.unsloth_lister.UnslothModelLister") as lister_cls:
            mock_lister = AsyncMock()
            mock_lister.list_models.return_value = []
            lister_cls.return_value = mock_lister

            # Author with hyphens and numbers (valid HF org names)
            result = await search_hf_models(author="meta-llama", ctx=_make_mock_ctx())

            assert "models" in result

    async def test_search_limit_boundary_values(self) -> None:
        """Test limit at exact boundary values."""
        from gpumod.mcp_tools import search_hf_models

        with patch("gpumod.discovery.unsloth_lister.UnslothModelLister") as lister_cls:
            mock_lister = AsyncMock()
            mock_lister.list_models.return_value = []
            lister_cls.return_value = mock_lister

            # Minimum valid limit
            result = await search_hf_models(limit=1, ctx=_make_mock_ctx())
            assert "models" in result

            # Maximum valid limit
            result = await search_hf_models(limit=100, ctx=_make_mock_ctx())
            assert "models" in result

            # Just over max
            result = await search_hf_models(limit=101, ctx=_make_mock_ctx())
            assert "error" in result


class TestListModelFilesEdgeCases:
    """Edge case tests for list_model_files robustness."""

    async def test_list_empty_repo_returns_unknown_format(self) -> None:
        """Repo with no GGUF files returns unknown format."""
        from gpumod.mcp_tools import list_model_files

        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.return_value = []
            fetcher_cls.return_value = mock_fetcher

            result = await list_model_files(
                repo_id="user/empty-repo",
                ctx=_make_mock_ctx(),
            )

            assert result["files"] == []
            assert result["model_format"] == "unknown"
            assert result["driver_hint"] is None

    async def test_list_repo_id_with_multiple_slashes(self) -> None:
        """repo_id with multiple slashes should fail validation."""
        from gpumod.mcp_tools import list_model_files

        result = await list_model_files(
            repo_id="org/sub/repo",
            ctx=_make_mock_ctx(),
        )
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_list_repo_id_with_empty_org(self) -> None:
        """repo_id with empty org should fail validation."""
        from gpumod.mcp_tools import list_model_files

        result = await list_model_files(
            repo_id="/repo",
            ctx=_make_mock_ctx(),
        )
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_list_repo_id_with_empty_name(self) -> None:
        """repo_id with empty name should fail validation."""
        from gpumod.mcp_tools import list_model_files

        result = await list_model_files(
            repo_id="org/",
            ctx=_make_mock_ctx(),
        )
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_list_zero_vram_budget(self) -> None:
        """Zero VRAM budget should fail validation."""
        from gpumod.mcp_tools import list_model_files

        result = await list_model_files(
            repo_id="user/repo",
            vram_budget_mb=0,
            ctx=_make_mock_ctx(),
        )
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_list_very_large_vram_budget(self, mock_gguf_files: list[GGUFFile]) -> None:
        """Very large VRAM budget should return all files."""
        from gpumod.mcp_tools import list_model_files

        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.return_value = mock_gguf_files
            fetcher_cls.return_value = mock_fetcher

            result = await list_model_files(
                repo_id="user/repo",
                vram_budget_mb=1_000_000,  # 1TB
                ctx=_make_mock_ctx(),
            )

            assert len(result["files"]) == len(mock_gguf_files)

    async def test_list_whitespace_repo_id(self) -> None:
        """Whitespace-only repo_id should fail validation."""
        from gpumod.mcp_tools import list_model_files

        result = await list_model_files(
            repo_id="   ",
            ctx=_make_mock_ctx(),
        )
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"


class TestListGGUFFilesEdgeCases:
    """Edge case tests for list_gguf_files robustness."""

    async def test_list_split_files(self) -> None:
        """Split GGUF files should be properly identified."""
        from gpumod.discovery.gguf_metadata import GGUFFile
        from gpumod.mcp_tools import list_gguf_files

        split_file = GGUFFile(
            filename="model-Q8_0-00001-of-00003.gguf",
            size_bytes=20_000_000_000,
            quant_type="Q8_0",
            estimated_vram_mb=21000,
            is_split=True,
            split_parts=3,
        )

        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.return_value = [split_file]
            fetcher_cls.return_value = mock_fetcher

            result = await list_gguf_files(
                repo_id="user/large-model",
                ctx=_make_mock_ctx(),
            )

            assert result["files"][0]["is_split"] is True
            assert result["files"][0]["split_parts"] == 3

    async def test_list_vram_budget_filters_all(self, mock_gguf_files: list[GGUFFile]) -> None:
        """VRAM budget that's too small filters out all files."""
        from gpumod.mcp_tools import list_gguf_files

        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            mock_fetcher.list_gguf_files.return_value = mock_gguf_files
            fetcher_cls.return_value = mock_fetcher

            result = await list_gguf_files(
                repo_id="user/repo",
                vram_budget_mb=100,  # Very small, nothing fits
                ctx=_make_mock_ctx(),
            )

            assert result["files"] == []
            assert result["count"] == 0


class TestGeneratePresetEdgeCases:
    """Edge case tests for generate_preset robustness."""

    async def test_generate_empty_gguf_file(self) -> None:
        """Empty gguf_file should fail validation."""
        from gpumod.mcp_tools import generate_preset

        result = await generate_preset(
            repo_id="user/repo",
            gguf_file="",
            ctx=_make_mock_ctx(),
        )
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    async def test_generate_gguf_file_without_extension(self) -> None:
        """gguf_file without .gguf extension should fail."""
        from gpumod.mcp_tools import generate_preset

        result = await generate_preset(
            repo_id="user/repo",
            gguf_file="model",
            ctx=_make_mock_ctx(),
        )
        assert "error" in result

    async def test_generate_context_size_boundary_min(self) -> None:
        """Context size at minimum boundary should work."""
        from gpumod.mcp_tools import generate_preset

        result = await generate_preset(
            repo_id="user/repo",
            gguf_file="model.gguf",
            context_size=512,  # Minimum valid
            ctx=_make_mock_ctx(),
        )
        assert "preset" in result
        assert "512" in result["preset"]

    async def test_generate_context_size_boundary_max(self) -> None:
        """Context size at maximum boundary should work."""
        from gpumod.mcp_tools import generate_preset

        result = await generate_preset(
            repo_id="user/repo",
            gguf_file="model.gguf",
            context_size=262144,  # Maximum valid (256k)
            ctx=_make_mock_ctx(),
        )
        assert "preset" in result
        assert "262144" in result["preset"]

    async def test_generate_special_chars_in_repo_name(self) -> None:
        """Repo names with hyphens and numbers should work."""
        from gpumod.mcp_tools import generate_preset

        result = await generate_preset(
            repo_id="meta-llama/Llama-3.1-8B-GGUF",
            gguf_file="model-Q4_K_M.gguf",
            ctx=_make_mock_ctx(),
        )
        assert "preset" in result
        assert "meta-llama/Llama-3.1-8B-GGUF" in result["preset"]

    async def test_generate_auto_service_id_from_repo(self) -> None:
        """Auto-generated service_id should be clean and valid."""
        from gpumod.mcp_tools import generate_preset

        result = await generate_preset(
            repo_id="unsloth/Qwen3-Coder-Next-GGUF",
            gguf_file="model.gguf",
            ctx=_make_mock_ctx(),
        )
        assert "service_id" in result
        # Should not contain GGUF suffix
        assert "gguf" not in result["service_id"].lower()
        # Should be lowercase and clean
        assert result["service_id"] == result["service_id"].lower()


class TestHuggingFaceSearcherEdgeCases:
    """Edge case tests for HuggingFaceSearcher implementation."""

    async def test_searcher_cache_miss_then_hit(self) -> None:
        """Second search should use cached results."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        searcher = HuggingFaceSearcher(cache_ttl_seconds=3600)

        with patch("gpumod.discovery.hf_searcher.HfApi") as api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = []
            api_cls.return_value = mock_api

            # First call - cache miss
            await searcher.search(query="test", driver="llamacpp")
            assert mock_api.list_models.call_count == 1

            # Second call - cache hit
            await searcher.search(query="test", driver="llamacpp")
            assert mock_api.list_models.call_count == 1  # No additional call

    async def test_searcher_force_refresh_bypasses_cache(self) -> None:
        """force_refresh should bypass cache."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        searcher = HuggingFaceSearcher(cache_ttl_seconds=3600)

        with patch("gpumod.discovery.hf_searcher.HfApi") as api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = []
            api_cls.return_value = mock_api

            # First call
            await searcher.search(query="test", driver="llamacpp")
            assert mock_api.list_models.call_count == 1

            # Second call with force_refresh
            await searcher.search(query="test", driver="llamacpp", force_refresh=True)
            assert mock_api.list_models.call_count == 2

    async def test_searcher_different_queries_different_cache(self) -> None:
        """Different queries should have separate cache entries."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        searcher = HuggingFaceSearcher(cache_ttl_seconds=3600)

        with patch("gpumod.discovery.hf_searcher.HfApi") as api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = []
            api_cls.return_value = mock_api

            await searcher.search(query="query1", driver="llamacpp")
            await searcher.search(query="query2", driver="llamacpp")
            await searcher.search(query="query1", driver="vllm")

            assert mock_api.list_models.call_count == 3

    async def test_searcher_clear_cache(self) -> None:
        """clear_cache should invalidate all entries."""
        from gpumod.discovery.hf_searcher import HuggingFaceSearcher

        searcher = HuggingFaceSearcher(cache_ttl_seconds=3600)

        with patch("gpumod.discovery.hf_searcher.HfApi") as api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = []
            api_cls.return_value = mock_api

            await searcher.search(query="test", driver="llamacpp")
            searcher.clear_cache()
            await searcher.search(query="test", driver="llamacpp")

            assert mock_api.list_models.call_count == 2


# ---------------------------------------------------------------------------
# Security Tests - SEC-V1 Input Validation (per SECURITY.md)
# ---------------------------------------------------------------------------


class TestDiscoveryToolsSecurity:
    """Security tests per SECURITY.md SEC-V1, SEC-E3 requirements."""

    # -------------------------------------------------------------------------
    # T1: Shell Injection Prevention
    # -------------------------------------------------------------------------

    async def test_repo_id_rejects_shell_injection(self) -> None:
        """repo_id with shell metacharacters should be rejected (SEC-V1/T1)."""
        from gpumod.mcp_tools import list_gguf_files, list_model_files

        shell_payloads = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "$(whoami)",
            "`id`",
            "&& ls",
            "|| true",
            "> /tmp/pwned",
            "user/repo; rm -rf /",
        ]

        for payload in shell_payloads:
            result = await list_gguf_files(repo_id=payload, ctx=_make_mock_ctx())
            assert "error" in result, f"Shell injection not rejected: {payload}"
            assert result["code"] == "VALIDATION_ERROR"

            result = await list_model_files(repo_id=payload, ctx=_make_mock_ctx())
            assert "error" in result, f"Shell injection not rejected: {payload}"

    async def test_gguf_file_rejects_shell_injection(self) -> None:
        """gguf_file with shell metacharacters should be rejected (SEC-V1/T1)."""
        from gpumod.mcp_tools import generate_preset

        shell_payloads = [
            "model.gguf; rm -rf /",
            "model.gguf | cat /etc/passwd",
            "$(whoami).gguf",
            "`id`.gguf",
        ]

        for payload in shell_payloads:
            result = await generate_preset(
                repo_id="user/repo", gguf_file=payload, ctx=_make_mock_ctx()
            )
            # Should either reject or sanitize - no shell execution
            assert "error" in result or "preset" in result

    # -------------------------------------------------------------------------
    # T2: SQL Injection Prevention
    # -------------------------------------------------------------------------

    async def test_repo_id_rejects_sql_injection(self) -> None:
        """repo_id with SQL metacharacters should be rejected (SEC-V1/T2)."""
        from gpumod.mcp_tools import list_gguf_files, list_model_files

        sql_payloads = [
            "'; DROP TABLE services--",
            "' OR '1'='1",
            "user/repo'; DELETE FROM models--",
            "1; SELECT * FROM users--",
            "UNION SELECT password FROM users",
        ]

        for payload in sql_payloads:
            result = await list_gguf_files(repo_id=payload, ctx=_make_mock_ctx())
            assert "error" in result, f"SQL injection not rejected: {payload}"

            result = await list_model_files(repo_id=payload, ctx=_make_mock_ctx())
            assert "error" in result, f"SQL injection not rejected: {payload}"

    # -------------------------------------------------------------------------
    # T3: Path Traversal Prevention
    # -------------------------------------------------------------------------

    async def test_repo_id_rejects_path_traversal(self) -> None:
        """repo_id with path traversal should be rejected (SEC-V1/T3)."""
        from gpumod.mcp_tools import list_gguf_files, list_model_files

        traversal_payloads = [
            "../../etc/passwd",
            "../../../root/.ssh/id_rsa",
            "..%2f..%2fetc/passwd",
            "....//....//etc/passwd",
            "/etc/passwd",
            "user/../../../etc/passwd",
        ]

        for payload in traversal_payloads:
            result = await list_gguf_files(repo_id=payload, ctx=_make_mock_ctx())
            assert "error" in result, f"Path traversal not rejected: {payload}"

            result = await list_model_files(repo_id=payload, ctx=_make_mock_ctx())
            assert "error" in result, f"Path traversal not rejected: {payload}"

    async def test_gguf_file_rejects_path_traversal(self) -> None:
        """gguf_file with path traversal should be rejected (SEC-V1/T3)."""
        from gpumod.mcp_tools import generate_preset

        traversal_payloads = [
            "../../../etc/passwd.gguf",
            "/etc/passwd.gguf",
            "..%2fmodel.gguf",
        ]

        for payload in traversal_payloads:
            result = await generate_preset(
                repo_id="user/repo", gguf_file=payload, ctx=_make_mock_ctx()
            )
            # Should either reject or produce safe output
            if "preset" in result:
                # Verify no traversal in output
                assert "../" not in result["preset"]
                assert "/etc/" not in result["preset"]

    # -------------------------------------------------------------------------
    # T4: Template Injection Prevention
    # -------------------------------------------------------------------------

    async def test_repo_id_rejects_template_injection(self) -> None:
        """repo_id with Jinja2 template syntax should be rejected (SEC-V1/T4)."""
        from gpumod.mcp_tools import list_gguf_files, list_model_files

        template_payloads = [
            "{{7*7}}",
            "{{config}}",
            "{%import os%}{{os.popen('id').read()}}",
            "{{ self.__class__.__mro__ }}",
            "user/{{7*7}}",
        ]

        for payload in template_payloads:
            result = await list_gguf_files(repo_id=payload, ctx=_make_mock_ctx())
            assert "error" in result, f"Template injection not rejected: {payload}"

            result = await list_model_files(repo_id=payload, ctx=_make_mock_ctx())
            assert "error" in result, f"Template injection not rejected: {payload}"

    async def test_service_id_rejects_template_injection(self) -> None:
        """service_id with template syntax should be rejected (SEC-V1/T4)."""
        from gpumod.mcp_tools import generate_preset

        result = await generate_preset(
            repo_id="user/repo",
            gguf_file="model.gguf",
            service_id="{{7*7}}",
            ctx=_make_mock_ctx(),
        )

        # Should reject template injection in service_id
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    # -------------------------------------------------------------------------
    # T10: Terminal Escape Injection Prevention (SEC-E3)
    # -------------------------------------------------------------------------

    async def test_search_sanitizes_ansi_escapes(self) -> None:
        """Model names with ANSI escapes should be sanitized (SEC-E3/T10)."""
        from gpumod.discovery.protocols import SearchResult
        from gpumod.mcp_tools import search_hf_models

        # Malicious model with ANSI escape codes
        malicious_result = SearchResult(
            repo_id="attacker/evil-model",
            name="\x1b[31mEVIL\x1b[0m \x1b[5mBLINKING\x1b[0m",
            description="Model with \x1b[7mreverse video\x1b[0m",
            last_modified=datetime.now(tz=UTC),
            tags=("gguf",),
            model_format="gguf",
            driver_hint="llamacpp",
        )

        with patch("gpumod.mcp_tools.HuggingFaceSearcher") as searcher_cls:
            mock_searcher = AsyncMock()
            mock_searcher.search.return_value = [malicious_result]
            searcher_cls.return_value = mock_searcher

            result = await search_hf_models(search="evil", driver="llamacpp", ctx=_make_mock_ctx())

            # ANSI codes should not appear in output (sanitized or stripped)
            model = result["models"][0]
            assert "\x1b[" not in model["name"]
            if model.get("description"):
                assert "\x1b[" not in model["description"]

    async def test_search_sanitizes_control_characters(self) -> None:
        """Model names with control characters should be sanitized (SEC-E3)."""
        from gpumod.discovery.protocols import SearchResult
        from gpumod.mcp_tools import search_hf_models

        # Model with control characters
        malicious_result = SearchResult(
            repo_id="attacker/control-model",
            name="Model\x00with\x07bell\x08backspace",
            description="Has\ttab\nand\rnewline",
            last_modified=datetime.now(tz=UTC),
            tags=("gguf",),
            model_format="gguf",
            driver_hint="llamacpp",
        )

        with patch("gpumod.mcp_tools.HuggingFaceSearcher") as searcher_cls:
            mock_searcher = AsyncMock()
            mock_searcher.search.return_value = [malicious_result]
            searcher_cls.return_value = mock_searcher

            result = await search_hf_models(
                search="control", driver="llamacpp", ctx=_make_mock_ctx()
            )

            model = result["models"][0]
            # Null bytes should not be in output
            assert "\x00" not in model["name"]
            assert "\x07" not in model["name"]
            assert "\x08" not in model["name"]

    # -------------------------------------------------------------------------
    # T5: Information Disclosure Prevention (SEC-E1)
    # -------------------------------------------------------------------------

    async def test_error_does_not_leak_internal_paths(self) -> None:
        """Error responses should not contain internal file paths (SEC-E1/T5)."""
        from gpumod.discovery.gguf_metadata import RepoNotFoundError
        from gpumod.mcp_tools import list_gguf_files

        with patch("gpumod.mcp_tools.GGUFMetadataFetcher") as fetcher_cls:
            mock_fetcher = AsyncMock()
            # Simulate error with internal path
            mock_fetcher.list_gguf_files.side_effect = RepoNotFoundError(
                "Error at /home/user/.config/gpumod/cache/repo: not found"
            )
            fetcher_cls.return_value = mock_fetcher

            result = await list_gguf_files(
                repo_id="fake/repo",
                ctx=_make_mock_ctx(),
            )

            assert "error" in result
            # Error message should not contain sensitive paths
            assert "/home/" not in result["error"]
            assert ".config" not in result["error"]

    # -------------------------------------------------------------------------
    # DoS Prevention
    # -------------------------------------------------------------------------

    async def test_extremely_long_repo_id_handled(self) -> None:
        """Extremely long repo_id should be handled gracefully."""
        from gpumod.mcp_tools import list_model_files

        # 10KB string
        long_repo_id = "a" * 10000 + "/" + "b" * 10000

        result = await list_model_files(
            repo_id=long_repo_id,
            ctx=_make_mock_ctx(),
        )

        # Should either reject or handle gracefully, not crash
        assert "error" in result or "files" in result

    async def test_null_bytes_in_input(self) -> None:
        """Null bytes in input should be handled safely."""
        from gpumod.mcp_tools import list_model_files

        result = await list_model_files(
            repo_id="user\x00/repo\x00",
            ctx=_make_mock_ctx(),
        )

        # Should reject or sanitize
        assert "error" in result or "files" in result
