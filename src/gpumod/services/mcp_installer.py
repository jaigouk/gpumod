"""MCP server systemd unit self-installer (gpumod-xbx).

Provides functions to auto-detect the gpumod installation paths,
render a systemd user unit file for the MCP server, and install it.

Usage::

    gpumod install-server          # install to ~/.config/systemd/user/
    gpumod install-server --dry-run  # print rendered unit without writing
"""

from __future__ import annotations

import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates" / "systemd"
_UNIT_NAME = "gpumod-mcp.service"
_SYSTEMD_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"


def detect_server_paths() -> dict[str, str]:
    """Auto-detect python binary, venv bin dir, and project working directory.

    Returns
    -------
    dict
        Keys: ``python_bin``, ``venv_bin``, ``working_dir``.
    """
    python_bin = sys.executable
    venv_bin = str(Path(python_bin).parent)

    # Walk up from this file to find the project root (contains pyproject.toml).
    current = Path(__file__).resolve().parent
    for ancestor in [current, *current.parents]:
        if (ancestor / "pyproject.toml").exists():
            working_dir = str(ancestor)
            break
    else:
        # Fallback: use the venv's parent (common layout: project/.venv/bin/python)
        working_dir = str(Path(venv_bin).parent.parent)

    return {
        "python_bin": python_bin,
        "venv_bin": venv_bin,
        "working_dir": working_dir,
    }


def render_mcp_unit(
    *,
    python_bin: str,
    venv_bin: str,
    working_dir: str,
    host: str | None = None,
    port: int | None = None,
) -> str:
    """Render the MCP server systemd unit file from the Jinja2 template.

    Parameters
    ----------
    python_bin:
        Absolute path to the Python interpreter.
    venv_bin:
        Absolute path to the venv bin directory.
    working_dir:
        Absolute path to the project working directory.
    host:
        Optional host to bind to (sets GPUMOD_MCP_HOST env var).
    port:
        Optional port to bind to (sets GPUMOD_MCP_PORT env var).

    Returns
    -------
    str
        The rendered systemd unit file content.
    """
    env = Environment(  # noqa: S701 — not user-facing, trusted templates only
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template("mcp-server.service.j2")

    context: dict[str, str | int] = {
        "python_bin": python_bin,
        "venv_bin": venv_bin,
        "working_dir": working_dir,
    }
    if host is not None:
        context["host"] = host
    if port is not None:
        context["port"] = port

    return template.render(**context)


def install_server_unit(
    *,
    python_bin: str,
    venv_bin: str,
    working_dir: str,
    host: str | None = None,
    port: int | None = None,
    unit_dir: Path | None = None,
    dry_run: bool = False,
) -> Path | None:
    """Render and install the MCP server systemd unit file.

    Parameters
    ----------
    python_bin, venv_bin, working_dir, host, port:
        Forwarded to :func:`render_mcp_unit`.
    unit_dir:
        Target directory for the unit file. Defaults to
        ``~/.config/systemd/user/``.
    dry_run:
        If True, render but do not write the file.

    Returns
    -------
    Path or None
        The path to the written unit file, or None if dry_run.
    """
    rendered = render_mcp_unit(
        python_bin=python_bin,
        venv_bin=venv_bin,
        working_dir=working_dir,
        host=host,
        port=port,
    )

    if dry_run:
        return None

    target_dir = unit_dir or _SYSTEMD_UNIT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    unit_path = target_dir / _UNIT_NAME
    unit_path.write_text(rendered)
    return unit_path
