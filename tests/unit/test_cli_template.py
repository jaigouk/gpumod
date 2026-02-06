"""Tests for gpumod.cli_template -- Template CLI commands."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import typer.testing

from gpumod.cli import app
from gpumod.models import DriverType, Service

runner = typer.testing.CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(
    *,
    id: str = "svc-1",
    name: str = "Test vLLM",
    driver: DriverType = DriverType.VLLM,
    port: int | None = 8000,
    vram_mb: int = 4096,
    unit_name: str | None = None,
) -> Service:
    return Service(
        id=id, name=name, driver=driver, port=port, vram_mb=vram_mb, unit_name=unit_name
    )


def _make_mock_context() -> MagicMock:
    ctx = MagicMock()
    ctx.template_engine = MagicMock()
    ctx.template_engine.available_templates = MagicMock(return_value=[])
    ctx.template_engine.render = MagicMock(return_value="")
    ctx.template_engine.render_service_unit = MagicMock(return_value="")
    ctx.db = MagicMock()
    ctx.db.get_service = AsyncMock(return_value=None)
    ctx.db.get_setting = AsyncMock(return_value=None)
    ctx.db.close = AsyncMock()
    ctx.preset_loader = MagicMock()
    ctx.preset_loader.discover_presets = MagicMock(return_value=[])
    return ctx


# ---------------------------------------------------------------------------
# template list tests
# ---------------------------------------------------------------------------


class TestTemplateList:
    """Tests for `gpumod template list` command."""

    def test_template_list_shows_available_templates(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.template_engine.available_templates = MagicMock(
            return_value=["vllm.service.j2", "llamacpp.service.j2"]
        )

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["template", "list"])

        assert result.exit_code == 0
        assert "vllm.service.j2" in result.output
        assert "llamacpp.service.j2" in result.output

    def test_template_list_json_flag(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.template_engine.available_templates = MagicMock(
            return_value=["vllm.service.j2", "llamacpp.service.j2"]
        )

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["template", "list", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert "vllm.service.j2" in parsed
        assert "llamacpp.service.j2" in parsed

    def test_template_list_empty_shows_message(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.template_engine.available_templates = MagicMock(return_value=[])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["template", "list"])

        assert result.exit_code == 0
        assert "no templates" in result.output.lower()


# ---------------------------------------------------------------------------
# template show tests
# ---------------------------------------------------------------------------


class TestTemplateShow:
    """Tests for `gpumod template show <template_name>` command."""

    def test_template_show_renders_named_template(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.template_engine.render = MagicMock(return_value="[Unit]\nDescription=Test Unit\n")

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["template", "show", "vllm.service.j2"])

        assert result.exit_code == 0
        mock_ctx.template_engine.render.assert_called_once()
        assert "vllm.service.j2" in str(mock_ctx.template_engine.render.call_args)

    def test_template_show_not_found_shows_error(self) -> None:
        from jinja2.exceptions import TemplateNotFound

        mock_ctx = _make_mock_context()
        mock_ctx.template_engine.render = MagicMock(side_effect=TemplateNotFound("nonexistent.j2"))

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["template", "show", "nonexistent.j2"])

        assert result.exit_code == 0  # error_handler catches it
        assert "error" in result.output.lower() or "not found" in result.output.lower()

    def test_template_show_json_flag(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.template_engine.render = MagicMock(return_value="[Unit]\nDescription=Test\n")

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["template", "show", "vllm.service.j2", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "template_name" in parsed
        assert "rendered" in parsed
        assert parsed["template_name"] == "vllm.service.j2"


# ---------------------------------------------------------------------------
# template generate tests
# ---------------------------------------------------------------------------


class TestTemplateGenerate:
    """Tests for `gpumod template generate <service_id>` command."""

    def test_template_generate_renders_service_unit(self) -> None:
        svc = _make_service(id="svc-1", name="Test vLLM", port=8000)
        mock_ctx = _make_mock_context()
        mock_ctx.db.get_service = AsyncMock(return_value=svc)
        mock_ctx.template_engine.render_service_unit = MagicMock(
            return_value="[Unit]\nDescription=svc-1\n[Service]\nExecStart=/usr/bin/vllm\n"
        )

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["template", "generate", "svc-1"])

        assert result.exit_code == 0
        mock_ctx.template_engine.render_service_unit.assert_called_once()
        assert "[Unit]" in result.output or "svc-1" in result.output

    def test_template_generate_for_service_id(self) -> None:
        svc = _make_service(id="my-vllm", name="My vLLM Service")
        mock_ctx = _make_mock_context()
        mock_ctx.db.get_service = AsyncMock(return_value=svc)
        mock_ctx.template_engine.render_service_unit = MagicMock(
            return_value="[Unit]\nDescription=my-vllm\n"
        )

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["template", "generate", "my-vllm"])

        assert result.exit_code == 0
        # Verify the service was looked up from DB
        mock_ctx.db.get_service.assert_awaited_once_with("my-vllm")
        # Verify the engine was called with the looked-up service
        call_args = mock_ctx.template_engine.render_service_unit.call_args
        assert call_args[0][0] == svc or call_args[1].get("service") == svc

    def test_template_generate_json_flag(self) -> None:
        svc = _make_service(id="svc-1")
        mock_ctx = _make_mock_context()
        mock_ctx.db.get_service = AsyncMock(return_value=svc)
        mock_ctx.template_engine.render_service_unit = MagicMock(
            return_value="[Unit]\nDescription=svc-1\n"
        )

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["template", "generate", "svc-1", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "service_id" in parsed
        assert "rendered" in parsed
        assert parsed["service_id"] == "svc-1"

    def test_template_generate_service_not_found(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.db.get_service = AsyncMock(return_value=None)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["template", "generate", "missing"])

        assert result.exit_code == 0  # error_handler catches it
        assert "error" in result.output.lower() or "not found" in result.output.lower()

    def test_template_generate_with_output_file(self, tmp_path: object) -> None:
        """--output writes rendered unit to a file."""
        import tempfile
        from pathlib import Path

        svc = _make_service(id="svc-1")
        mock_ctx = _make_mock_context()
        mock_ctx.db.get_service = AsyncMock(return_value=svc)
        rendered = "[Unit]\nDescription=svc-1\n[Service]\nExecStart=/usr/bin/vllm\n"
        mock_ctx.template_engine.render_service_unit = MagicMock(return_value=rendered)

        with tempfile.TemporaryDirectory() as td:
            outfile = Path(td) / "test.service"
            with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
                result = runner.invoke(
                    app, ["template", "generate", "svc-1", "--output", str(outfile)]
                )

            assert result.exit_code == 0
            assert outfile.exists()
            assert outfile.read_text() == rendered


# ---------------------------------------------------------------------------
# template install tests
# ---------------------------------------------------------------------------


class TestTemplateInstall:
    """Tests for `gpumod template install <service_id>` command."""

    def test_template_install_writes_unit_file(self) -> None:
        import tempfile
        from pathlib import Path

        svc = _make_service(id="svc-1", unit_name="gpumod-svc-1")
        mock_ctx = _make_mock_context()
        mock_ctx.db.get_service = AsyncMock(return_value=svc)
        rendered = "[Unit]\nDescription=svc-1\n[Service]\nExecStart=/usr/bin/vllm\n"
        mock_ctx.template_engine.render_service_unit = MagicMock(return_value=rendered)

        with tempfile.TemporaryDirectory() as td:
            install_dir = Path(td)
            with (
                patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
                patch(
                    "gpumod.cli_template._SYSTEMD_UNIT_DIR",
                    install_dir,
                ),
            ):
                result = runner.invoke(app, ["template", "install", "svc-1", "--yes"])

            assert result.exit_code == 0
            unit_file = install_dir / "gpumod-svc-1.service"
            assert unit_file.exists()
            assert unit_file.read_text() == rendered

    def test_template_install_requires_confirmation(self) -> None:
        svc = _make_service(id="svc-1", unit_name="gpumod-svc-1")
        mock_ctx = _make_mock_context()
        mock_ctx.db.get_service = AsyncMock(return_value=svc)
        rendered = "[Unit]\nDescription=svc-1\n"
        mock_ctx.template_engine.render_service_unit = MagicMock(return_value=rendered)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            # Without --yes flag: shows rendered but does NOT write
            result = runner.invoke(app, ["template", "install", "svc-1"])

        assert result.exit_code == 0
        # Should show the rendered unit in preview
        assert "[Unit]" in result.output or "rendered" in result.output.lower()
        # Should indicate confirmation is needed
        assert "yes" in result.output.lower() or "confirm" in result.output.lower()

    def test_template_install_rejects_unsafe_paths(self) -> None:
        svc = _make_service(id="svc-1", unit_name="../../../tmp/evil")
        mock_ctx = _make_mock_context()
        mock_ctx.db.get_service = AsyncMock(return_value=svc)
        rendered = "[Unit]\nDescription=svc-1\n"
        mock_ctx.template_engine.render_service_unit = MagicMock(return_value=rendered)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["template", "install", "svc-1", "--yes"])

        assert result.exit_code == 0  # error_handler catches it
        # Should show a security error
        assert "error" in result.output.lower() or "unsafe" in result.output.lower()

    def test_template_install_service_not_found(self) -> None:
        mock_ctx = _make_mock_context()
        mock_ctx.db.get_service = AsyncMock(return_value=None)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["template", "install", "missing", "--yes"])

        assert result.exit_code == 0  # error_handler catches it
        assert "error" in result.output.lower() or "not found" in result.output.lower()

    def test_template_install_default_unit_name(self) -> None:
        """When service has no unit_name, use gpumod-{service_id} as default."""
        import tempfile
        from pathlib import Path

        svc = _make_service(id="svc-1", unit_name=None)
        mock_ctx = _make_mock_context()
        mock_ctx.db.get_service = AsyncMock(return_value=svc)
        rendered = "[Unit]\nDescription=svc-1\n"
        mock_ctx.template_engine.render_service_unit = MagicMock(return_value=rendered)

        with tempfile.TemporaryDirectory() as td:
            install_dir = Path(td)
            with (
                patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
                patch(
                    "gpumod.cli_template._SYSTEMD_UNIT_DIR",
                    install_dir,
                ),
            ):
                result = runner.invoke(app, ["template", "install", "svc-1", "--yes"])

            assert result.exit_code == 0
            unit_file = install_dir / "gpumod-svc-1.service"
            assert unit_file.exists()
