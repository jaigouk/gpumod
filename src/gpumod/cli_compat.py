"""Compat profile CLI — `gpumod compat list|show` (gpumod-ng7).

Discovery + content surface for the curated driver venv compat profiles
shipped with gpumod under ``src/gpumod/compat/profiles/``.

These profiles are operator-facing references: copy-paste-friendly
``compat:`` blocks for service presets, plus a known-good combination
the project maintainers have verified.
"""

from __future__ import annotations

import sys
from importlib import resources

import typer
import yaml

compat_app = typer.Typer(
    name="compat",
    help="Driver venv compatibility profiles (gpumod doctor venv contracts).",
)

_PROFILES_PACKAGE = "gpumod.compat.profiles"
_YAML_SUFFIX = ".yaml"


def _profile_names() -> list[str]:
    """Discover shipped profile names (without the .yaml suffix)."""
    profile_files = resources.files(_PROFILES_PACKAGE)
    return sorted(
        p.name.removesuffix(_YAML_SUFFIX)
        for p in profile_files.iterdir()
        if p.is_file() and p.name.endswith(_YAML_SUFFIX)
    )


def _profile_text(name: str) -> str:
    """Return the raw YAML text of a shipped profile.

    Raises
    ------
    FileNotFoundError
        If the named profile doesn't ship with this gpumod build.
    """
    target = resources.files(_PROFILES_PACKAGE) / f"{name}{_YAML_SUFFIX}"
    if not target.is_file():
        available = ", ".join(_profile_names()) or "(none)"
        msg = f"compat profile not found: {name!r}. Available: {available}"
        raise FileNotFoundError(msg)
    return target.read_text(encoding="utf-8")


def load_profile(name: str) -> dict[str, str]:
    """Load a shipped profile's ``compat:`` block as a plain dict.

    Used by ``gpumod doctor venv --profile <name>`` to substitute for
    an absent inline service.compat.

    Raises
    ------
    FileNotFoundError
        If the named profile doesn't exist.
    """
    text = _profile_text(name)
    data = yaml.safe_load(text) or {}
    compat = data.get("compat") or {}
    if not isinstance(compat, dict):
        msg = f"compat profile {name!r}: 'compat:' block must be a dict"
        raise ValueError(msg)
    return {str(k): str(v) for k, v in compat.items()}


@compat_app.command("list")
def list_command() -> None:
    """List all shipped compat profiles."""
    names = _profile_names()
    if not names:
        typer.echo("(no compat profiles shipped)")
        return
    for n in names:
        typer.echo(n)


@compat_app.command("show")
def show_command(
    name: str = typer.Argument(..., help="Profile name (see `compat list`)."),
) -> None:
    """Print a shipped compat profile's YAML content."""
    try:
        typer.echo(_profile_text(name), nl=False)
    except FileNotFoundError as exc:
        sys.stderr.write(f"{exc!s}\n")
        raise typer.Exit(code=1) from None
