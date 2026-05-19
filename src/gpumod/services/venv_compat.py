"""Driver venv compatibility checking (gpumod-ng7).

Validates the installed Python packages in a service's driver venv
against PEP 440 specifiers declared in the service's ``compat`` field.

Failure modes the library catches:

  - **DRIFT**  — a declared package is installed, but its version is
    outside the allowed specifier range
  - **MISSING** — a declared package is not installed at all

The library is pure: no subprocess calls, no imports of the target
venv (which may be broken — that's exactly the case we want to
catch). Reads dist-info METADATA files directly.

Design notes
------------

We deliberately do NOT use ``packaging.metadata.distributions`` from
the current interpreter — that would scan THIS process's site-packages,
not the target venv's. Instead we parse the METADATA files under
``<venv>/lib/python*/site-packages/*.dist-info/`` ourselves.

Names are normalised per PEP 503 (lowercase, hyphens) so
``huggingface-hub`` and ``huggingface_hub`` are treated as the same
package regardless of which form the operator declares.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from gpumod.models import Service

logger = logging.getLogger(__name__)


class CompatStatus(StrEnum):
    """One per declared package."""

    OK = "ok"
    DRIFT = "drift"
    MISSING = "missing"


@dataclass(frozen=True)
class CompatViolation:
    """A declared package whose installed state fails the contract."""

    package: str  # normalised name (lowercase, hyphens)
    constraint: str  # the PEP 440 specifier as declared
    installed: str | None  # None when MISSING
    status: CompatStatus


@dataclass(frozen=True)
class CompatResult:
    """Aggregate result of check_compat()."""

    ok: bool
    violations: list[CompatViolation] = field(default_factory=list)


def _normalise(name: str) -> str:
    """PEP 503 normalisation: lowercase + hyphens."""
    return re.sub(r"[-_.]+", "-", name).lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_installed_versions(venv_root: Path) -> dict[str, str]:
    """Return ``{normalised_name: version}`` parsed from ``<venv>/lib/python*/site-packages``.

    Empty dict when no site-packages directory exists. When multiple
    Python versions are present (rare; legacy multi-version venvs), the
    most recent one wins (sorts lexicographically — fine for python3.x).
    """
    lib = venv_root / "lib"
    if not lib.is_dir():
        return {}

    site_dirs = sorted(
        (p for p in lib.iterdir() if p.is_dir() and p.name.startswith("python")),
        reverse=True,
    )
    if not site_dirs:
        return {}

    site_packages = site_dirs[0] / "site-packages"
    if not site_packages.is_dir():
        return {}

    versions: dict[str, str] = {}
    for dist_info in site_packages.glob("*.dist-info"):
        metadata = dist_info / "METADATA"
        if not metadata.is_file():
            continue
        name, version = _parse_metadata(metadata)
        if name and version:
            versions[_normalise(name)] = version
    return versions


def _parse_metadata(path: Path) -> tuple[str | None, str | None]:
    """Extract Name and Version from a dist-info METADATA file."""
    name: str | None = None
    version: str | None = None
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    break  # METADATA headers end at first blank line
                if line.startswith("Name:"):
                    name = line[len("Name:") :].strip()
                elif line.startswith("Version:"):
                    version = line[len("Version:") :].strip()
                if name and version:
                    break
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
    return name, version


def check_compat(
    declared: dict[str, str],
    installed: dict[str, str],
) -> CompatResult:
    """Validate ``installed`` against the PEP 440 specifiers in ``declared``.

    Extra installed packages are ignored — the contract is one-way.
    Name normalisation runs on both sides, so case + separator
    differences don't cause false MISSING.
    """
    # Normalise installed side once
    installed_normalised: dict[str, str] = {
        _normalise(name): version for name, version in installed.items()
    }

    violations: list[CompatViolation] = []
    for pkg, constraint in declared.items():
        norm_pkg = _normalise(pkg)
        installed_version = installed_normalised.get(norm_pkg)
        if installed_version is None:
            violations.append(
                CompatViolation(
                    package=norm_pkg,
                    constraint=constraint,
                    installed=None,
                    status=CompatStatus.MISSING,
                )
            )
            continue

        try:
            ver = Version(installed_version)
            spec = SpecifierSet(constraint)
        except (InvalidVersion, Exception) as exc:
            logger.warning(
                "Could not parse %s constraint=%r installed=%r: %s",
                pkg,
                constraint,
                installed_version,
                exc,
            )
            # Treat unparseable as DRIFT — better to fail loud than silently allow
            violations.append(
                CompatViolation(
                    package=norm_pkg,
                    constraint=constraint,
                    installed=installed_version,
                    status=CompatStatus.DRIFT,
                )
            )
            continue

        if ver not in spec:
            violations.append(
                CompatViolation(
                    package=norm_pkg,
                    constraint=constraint,
                    installed=installed_version,
                    status=CompatStatus.DRIFT,
                )
            )

    return CompatResult(ok=not violations, violations=violations)


def find_venv_root(
    service: Service,
    settings: dict[str, str],
) -> Path | None:
    """Resolve the driver venv root from service config or global settings.

    Resolution order (matches the vllm.service.j2 template):

      1. ``service.extra_config['unit_vars']['vllm_bin']``
      2. ``settings['vllm_bin']``
      3. ``shutil.which('vllm')`` on PATH

    Returns the venv root (the parent of ``bin/``), e.g. ``~/.venvs/vllm``.
    Returns ``None`` if no binary path is resolvable.
    """
    bin_path_str: str | None = None
    extra: dict[str, Any] = service.extra_config or {}
    unit_vars: dict[str, Any] = extra.get("unit_vars") or {}
    bin_path_str = unit_vars.get("vllm_bin")
    if not bin_path_str:
        bin_path_str = settings.get("vllm_bin")
    if not bin_path_str:
        bin_path_str = shutil.which("vllm")
    if not bin_path_str:
        return None
    # <venv>/bin/vllm → <venv>
    return Path(bin_path_str).parent.parent
