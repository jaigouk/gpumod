"""Tests for gpumod.services.mcp_installer — MCP server systemd unit self-install (gpumod-xbx)."""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path detection ─────────────────────────────────────────────────────


class TestDetectServerPaths:
    """detect_server_paths() should auto-detect python, venv bin, and working dir."""

    def test_returns_python_bin(self) -> None:
        from gpumod.services.mcp_installer import detect_server_paths

        paths = detect_server_paths()
        assert "python_bin" in paths
        assert Path(paths["python_bin"]).exists()

    def test_python_bin_matches_current_interpreter(self) -> None:
        from gpumod.services.mcp_installer import detect_server_paths

        paths = detect_server_paths()
        assert paths["python_bin"] == sys.executable

    def test_returns_venv_bin_dir(self) -> None:
        from gpumod.services.mcp_installer import detect_server_paths

        paths = detect_server_paths()
        assert "venv_bin" in paths
        assert Path(paths["venv_bin"]).is_dir()

    def test_venv_bin_is_parent_of_python(self) -> None:
        from gpumod.services.mcp_installer import detect_server_paths

        paths = detect_server_paths()
        assert paths["venv_bin"] == str(Path(sys.executable).parent)

    def test_returns_working_dir(self) -> None:
        from gpumod.services.mcp_installer import detect_server_paths

        paths = detect_server_paths()
        assert "working_dir" in paths
        assert Path(paths["working_dir"]).is_dir()

    def test_working_dir_contains_pyproject(self) -> None:
        """Working dir should be the project root (contains pyproject.toml)."""
        from gpumod.services.mcp_installer import detect_server_paths

        paths = detect_server_paths()
        assert (Path(paths["working_dir"]) / "pyproject.toml").exists()


# ── Template rendering ─────────────────────────────────────────────────


class TestRenderMcpUnit:
    """render_mcp_unit() should produce a valid systemd unit file."""

    def test_renders_unit_file(self) -> None:
        from gpumod.services.mcp_installer import render_mcp_unit

        result = render_mcp_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
        )
        assert "[Unit]" in result
        assert "[Service]" in result
        assert "[Install]" in result

    def test_contains_correct_exec_start(self) -> None:
        from gpumod.services.mcp_installer import render_mcp_unit

        result = render_mcp_unit(
            python_bin="/opt/gpumod/.venv/bin/python",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
        )
        assert "ExecStart=/opt/gpumod/.venv/bin/python -m gpumod.mcp_main" in result

    def test_contains_working_directory(self) -> None:
        from gpumod.services.mcp_installer import render_mcp_unit

        result = render_mcp_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/srv/gpumod",
        )
        assert "WorkingDirectory=/srv/gpumod" in result

    def test_contains_path_environment(self) -> None:
        from gpumod.services.mcp_installer import render_mcp_unit

        result = render_mcp_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
        )
        assert "PATH=/opt/gpumod/.venv/bin" in result

    def test_contains_start_limit_safeguard(self) -> None:
        from gpumod.services.mcp_installer import render_mcp_unit

        result = render_mcp_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
        )
        assert "StartLimitBurst=" in result
        assert "StartLimitIntervalSec=" in result

    def test_start_limit_in_unit_section(self) -> None:
        from gpumod.services.mcp_installer import render_mcp_unit

        result = render_mcp_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
        )
        unit_section = result.split("[Service]")[0]
        assert "StartLimitBurst=" in unit_section

    def test_contains_restart_on_failure(self) -> None:
        from gpumod.services.mcp_installer import render_mcp_unit

        result = render_mcp_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
        )
        assert "Restart=on-failure" in result

    def test_custom_host_and_port(self) -> None:
        from gpumod.services.mcp_installer import render_mcp_unit

        result = render_mcp_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
            host="0.0.0.0",  # noqa: S104
            port=8808,
        )
        assert "GPUMOD_MCP_HOST=0.0.0.0" in result
        assert "GPUMOD_MCP_PORT=8808" in result
        assert "GPUMOD_MCP_TRANSPORT=streamable-http" in result

    def test_default_renders_transport(self) -> None:
        """Default render (no explicit transport) sets streamable-http."""
        from gpumod.services.mcp_installer import render_mcp_unit

        result = render_mcp_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
        )
        assert "GPUMOD_MCP_TRANSPORT=streamable-http" in result
        assert "GPUMOD_MCP_HOST=127.0.0.1" in result
        assert "GPUMOD_MCP_PORT=8808" in result

    def test_custom_transport_sse(self) -> None:
        """Explicit transport=sse renders correctly."""
        from gpumod.services.mcp_installer import render_mcp_unit

        result = render_mcp_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
            transport="sse",
        )
        assert "GPUMOD_MCP_TRANSPORT=sse" in result

    def test_custom_host_port_override_defaults(self) -> None:
        """Custom host/port override the template defaults."""
        from gpumod.services.mcp_installer import render_mcp_unit

        result = render_mcp_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
            host="0.0.0.0",  # noqa: S104
            port=9999,
        )
        assert "GPUMOD_MCP_HOST=0.0.0.0" in result
        assert "GPUMOD_MCP_PORT=9999" in result

    def test_user_target(self) -> None:
        """User-level units should use default.target, not multi-user.target."""
        from gpumod.services.mcp_installer import render_mcp_unit

        result = render_mcp_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
        )
        assert "WantedBy=default.target" in result


# ── Unit file installation ─────────────────────────────────────────────


class TestInstallServerUnit:
    """install_server_unit() writes the unit file and flags daemon-reload."""

    def test_writes_unit_file(self, tmp_path: Path) -> None:
        from gpumod.services.mcp_installer import install_server_unit

        result = install_server_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
            unit_dir=tmp_path,
        )
        unit_path = tmp_path / "gpumod-mcp.service"
        assert unit_path.exists()
        content = unit_path.read_text()
        assert "[Unit]" in content
        assert result == unit_path

    def test_creates_unit_dir_if_missing(self, tmp_path: Path) -> None:
        from gpumod.services.mcp_installer import install_server_unit

        target = tmp_path / "subdir" / "systemd" / "user"
        install_server_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
            unit_dir=target,
        )
        assert (target / "gpumod-mcp.service").exists()

    def test_overwrites_existing_unit_file(self, tmp_path: Path) -> None:
        from gpumod.services.mcp_installer import install_server_unit

        unit_path = tmp_path / "gpumod-mcp.service"
        unit_path.write_text("old content")

        install_server_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
            unit_dir=tmp_path,
        )
        assert "old content" not in unit_path.read_text()
        assert "[Unit]" in unit_path.read_text()

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        from gpumod.services.mcp_installer import install_server_unit

        result = install_server_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
            unit_dir=tmp_path,
            dry_run=True,
        )
        assert not (tmp_path / "gpumod-mcp.service").exists()
        assert result is None

    def test_idempotent_no_change(self, tmp_path: Path) -> None:
        """Running twice with same params should produce identical file."""
        from gpumod.services.mcp_installer import install_server_unit

        install_server_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
            unit_dir=tmp_path,
        )
        first = (tmp_path / "gpumod-mcp.service").read_text()

        install_server_unit(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
            unit_dir=tmp_path,
        )
        second = (tmp_path / "gpumod-mcp.service").read_text()
        assert first == second


# ── CLI command ────────────────────────────────────────────────────────


class TestInstallServerCLI:
    """gpumod install-server CLI command integration."""

    def test_dry_run_prints_unit(self) -> None:
        from typer.testing import CliRunner

        from gpumod.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["install-server", "--dry-run"])
        assert result.exit_code == 0
        assert "[Unit]" in result.stdout
        assert "ExecStart=" in result.stdout
        assert "gpumod.mcp_main" in result.stdout

    def test_install_writes_file(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from gpumod.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["install-server", "--unit-dir", str(tmp_path), "--no-reload"])
        assert result.exit_code == 0
        assert (tmp_path / "gpumod-mcp.service").exists()

    def test_install_with_host_and_port(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from gpumod.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "install-server",
                "--unit-dir",
                str(tmp_path),
                "--no-reload",
                "--host",
                "0.0.0.0",  # noqa: S104
                "--port",
                "8808",
            ],
        )
        assert result.exit_code == 0
        content = (tmp_path / "gpumod-mcp.service").read_text()
        assert "GPUMOD_MCP_HOST=0.0.0.0" in content
        assert "GPUMOD_MCP_PORT=8808" in content

    def test_install_default_transport(self, tmp_path: Path) -> None:
        """Default install sets streamable-http transport."""
        from typer.testing import CliRunner

        from gpumod.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["install-server", "--unit-dir", str(tmp_path), "--no-reload"],
        )
        assert result.exit_code == 0
        content = (tmp_path / "gpumod-mcp.service").read_text()
        assert "GPUMOD_MCP_TRANSPORT=streamable-http" in content

    def test_install_custom_transport_sse(self, tmp_path: Path) -> None:
        """--transport=sse sets SSE transport in unit file."""
        from typer.testing import CliRunner

        from gpumod.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "install-server",
                "--unit-dir",
                str(tmp_path),
                "--no-reload",
                "--transport",
                "sse",
            ],
        )
        assert result.exit_code == 0
        content = (tmp_path / "gpumod-mcp.service").read_text()
        assert "GPUMOD_MCP_TRANSPORT=sse" in content

    def test_dry_run_shows_transport(self) -> None:
        """--dry-run output includes GPUMOD_MCP_TRANSPORT."""
        from typer.testing import CliRunner

        from gpumod.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["install-server", "--dry-run"])
        assert result.exit_code == 0
        assert "GPUMOD_MCP_TRANSPORT=streamable-http" in result.stdout
