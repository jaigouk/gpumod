"""Tests for gpumod.cli_model — Model CLI commands (list, info, register, remove)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import typer.testing

from gpumod.cli import app
from gpumod.models import ModelInfo, ModelSource

runner = typer.testing.CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(
    *,
    id: str = "meta-llama/Llama-3-8B",
    source: ModelSource = ModelSource.HUGGINGFACE,
    parameters_b: float | None = 8.0,
    architecture: str | None = "LlamaForCausalLM",
    base_vram_mb: int | None = 16000,
    kv_cache_per_1k_tokens_mb: int | None = 64,
    quantizations: list[str] | None = None,
    capabilities: list[str] | None = None,
    fetched_at: str | None = "2024-01-01T00:00:00+00:00",
    notes: str | None = None,
) -> ModelInfo:
    return ModelInfo(
        id=id,
        source=source,
        parameters_b=parameters_b,
        architecture=architecture,
        base_vram_mb=base_vram_mb,
        kv_cache_per_1k_tokens_mb=kv_cache_per_1k_tokens_mb,
        quantizations=quantizations if quantizations is not None else [],
        capabilities=capabilities if capabilities is not None else [],
        fetched_at=fetched_at,
        notes=notes,
    )


def _make_mock_context(
    **overrides: object,
) -> MagicMock:
    ctx = MagicMock()
    ctx.model_registry = MagicMock()
    ctx.db = MagicMock()

    # Set up async mocks with defaults
    ctx.model_registry.list_models = AsyncMock(return_value=[])
    ctx.model_registry.get = AsyncMock(return_value=None)
    ctx.model_registry.register = AsyncMock()
    ctx.model_registry.estimate_vram = AsyncMock(return_value=0)
    ctx.model_registry.remove = AsyncMock()
    ctx.db.close = AsyncMock()

    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


# ---------------------------------------------------------------------------
# model list tests
# ---------------------------------------------------------------------------


class TestModelList:
    """Tests for `gpumod model list` command."""

    def test_model_list_shows_all_models(self) -> None:
        m1 = _make_model(id="llama-8b", parameters_b=8.0)
        m2 = _make_model(
            id="mistral-7b",
            source=ModelSource.GGUF,
            parameters_b=7.0,
            architecture="MistralForCausalLM",
        )
        mock_ctx = _make_mock_context()
        mock_ctx.model_registry.list_models = AsyncMock(return_value=[m1, m2])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "llama-8b" in result.output
        assert "mistral-7b" in result.output

    def test_model_list_empty_shows_message(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.model_registry.list_models = AsyncMock(return_value=[])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "no models" in result.output.lower()

    def test_model_list_json_flag(self) -> None:
        m1 = _make_model(id="meta-llama/Llama-3-8B", parameters_b=8.0)
        m2 = _make_model(
            id="mistral-7b-q4",
            source=ModelSource.GGUF,
            parameters_b=7.0,
        )
        mock_ctx = _make_mock_context()
        mock_ctx.model_registry.list_models = AsyncMock(return_value=[m1, m2])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["model", "list", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["id"] == "meta-llama/Llama-3-8B"
        assert parsed[1]["id"] == "mistral-7b-q4"

    def test_model_list_table_columns(self) -> None:
        m1 = _make_model(id="llama-8b", parameters_b=8.0)
        mock_ctx = _make_mock_context()
        mock_ctx.model_registry.list_models = AsyncMock(return_value=[m1])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        output = result.output
        assert "ID" in output
        assert "Source" in output
        assert "Parameters" in output
        # Rich may truncate long column names; check for prefix
        assert "Architec" in output
        assert "Base VRAM" in output or "VRAM" in output
        assert "KV/1K" in output


# ---------------------------------------------------------------------------
# model info tests
# ---------------------------------------------------------------------------


class TestModelInfo:
    """Tests for `gpumod model info <model_id>` command."""

    def test_model_info_shows_model_details(self) -> None:
        m = _make_model(
            id="meta-llama/Llama-3-8B",
            parameters_b=8.0,
            architecture="LlamaForCausalLM",
            base_vram_mb=16000,
        )
        mock_ctx = _make_mock_context()
        mock_ctx.model_registry.get = AsyncMock(return_value=m)
        mock_ctx.model_registry.estimate_vram = AsyncMock(return_value=16262)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["model", "info", "meta-llama/Llama-3-8B"])

        assert result.exit_code == 0
        assert "meta-llama/Llama-3-8B" in result.output
        assert "LlamaForCausalLM" in result.output
        assert "8.0" in result.output

    def test_model_info_not_found_shows_error(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.model_registry.get = AsyncMock(return_value=None)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["model", "info", "nonexistent"])

        assert result.exit_code == 0  # error_handler catches it
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_model_info_shows_vram_estimate(self) -> None:
        m = _make_model(
            id="meta-llama/Llama-3-8B",
            base_vram_mb=16000,
            kv_cache_per_1k_tokens_mb=64,
        )
        mock_ctx = _make_mock_context()
        mock_ctx.model_registry.get = AsyncMock(return_value=m)
        mock_ctx.model_registry.estimate_vram = AsyncMock(return_value=16262)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["model", "info", "meta-llama/Llama-3-8B"])

        assert result.exit_code == 0
        assert "16262" in result.output or "16,262" in result.output
        assert "vram" in result.output.lower()

    def test_model_info_json_flag(self) -> None:
        m = _make_model(
            id="meta-llama/Llama-3-8B",
            parameters_b=8.0,
            architecture="LlamaForCausalLM",
            base_vram_mb=16000,
        )
        mock_ctx = _make_mock_context()
        mock_ctx.model_registry.get = AsyncMock(return_value=m)
        mock_ctx.model_registry.estimate_vram = AsyncMock(return_value=16262)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["model", "info", "meta-llama/Llama-3-8B", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["id"] == "meta-llama/Llama-3-8B"
        assert parsed["parameters_b"] == 8.0
        assert parsed["architecture"] == "LlamaForCausalLM"

    def test_model_info_context_size_option(self) -> None:
        m = _make_model(
            id="meta-llama/Llama-3-8B",
            base_vram_mb=16000,
            kv_cache_per_1k_tokens_mb=64,
        )
        mock_ctx = _make_mock_context()
        mock_ctx.model_registry.get = AsyncMock(return_value=m)
        mock_ctx.model_registry.estimate_vram = AsyncMock(return_value=16512)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(
                app, ["model", "info", "meta-llama/Llama-3-8B", "--context-size", "8000"]
            )

        assert result.exit_code == 0
        # Verify estimate_vram was called with context_size=8000
        mock_ctx.model_registry.estimate_vram.assert_awaited_once_with(
            "meta-llama/Llama-3-8B", context_size=8000
        )


# ---------------------------------------------------------------------------
# model register tests
# ---------------------------------------------------------------------------


class TestModelRegister:
    """Tests for `gpumod model register <model_id>` command."""

    def test_model_register_huggingface(self) -> None:
        registered = _make_model(id="meta-llama/Llama-3-8B", source=ModelSource.HUGGINGFACE)
        mock_ctx = _make_mock_context()
        mock_ctx.model_registry.register = AsyncMock(return_value=registered)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["model", "register", "meta-llama/Llama-3-8B"])

        assert result.exit_code == 0
        mock_ctx.model_registry.register.assert_awaited_once()
        call_args = mock_ctx.model_registry.register.call_args
        assert call_args[0][0] == "meta-llama/Llama-3-8B"
        assert call_args[0][1] == ModelSource.HUGGINGFACE

    def test_model_register_gguf(self) -> None:
        registered = _make_model(id="mistral-7b-q4", source=ModelSource.GGUF)
        mock_ctx = _make_mock_context()
        mock_ctx.model_registry.register = AsyncMock(return_value=registered)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(
                app,
                [
                    "model",
                    "register",
                    "mistral-7b-q4",
                    "--source",
                    "gguf",
                    "--file-path",
                    "/models/mistral-7b-q4.gguf",
                ],
            )

        assert result.exit_code == 0
        mock_ctx.model_registry.register.assert_awaited_once()
        call_args = mock_ctx.model_registry.register.call_args
        assert call_args[0][0] == "mistral-7b-q4"
        assert call_args[0][1] == ModelSource.GGUF
        assert call_args[1]["file_path"] == "/models/mistral-7b-q4.gguf"

    def test_model_register_shows_result(self) -> None:
        registered = _make_model(
            id="meta-llama/Llama-3-8B",
            source=ModelSource.HUGGINGFACE,
            parameters_b=8.0,
            architecture="LlamaForCausalLM",
        )
        mock_ctx = _make_mock_context()
        mock_ctx.model_registry.register = AsyncMock(return_value=registered)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["model", "register", "meta-llama/Llama-3-8B"])

        assert result.exit_code == 0
        assert "meta-llama/Llama-3-8B" in result.output
        # Should indicate successful registration
        assert "register" in result.output.lower() or "success" in result.output.lower()


# ---------------------------------------------------------------------------
# model remove tests
# ---------------------------------------------------------------------------


class TestModelRemove:
    """Tests for `gpumod model remove <model_id>` command."""

    def test_model_remove_calls_registry_remove(self) -> None:
        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["model", "remove", "meta-llama/Llama-3-8B"])

        assert result.exit_code == 0
        mock_ctx.model_registry.remove.assert_awaited_once_with("meta-llama/Llama-3-8B")
        assert "meta-llama/Llama-3-8B" in result.output
        # Should show removal message
        assert "remov" in result.output.lower()

    def test_model_remove_not_found_shows_error(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.model_registry.remove = AsyncMock(
            side_effect=ValueError("Model not found: 'nonexistent'")
        )

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["model", "remove", "nonexistent"])

        assert result.exit_code == 0  # error_handler catches it
        assert "error" in result.output.lower() or "not found" in result.output.lower()
