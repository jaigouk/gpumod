"""Doctor CLI — `gpumod doctor venv` (gpumod-ng7).

Validates a service's driver venv against the PEP 440 specifiers declared
in either the service's inline ``compat`` field, or a shipped profile
loaded via ``--profile <name>``.

Designed to be invoked both interactively (operator runs it after editing
the venv) and from systemd ``ExecStartPre=`` hooks (template integration
lands in gpumod-l20).

Exit codes:
  0 — venv is OK (or no contract declared)
  1 — DRIFT / MISSING with diagnostic on stderr
  2 — unknown service id
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import typer

from gpumod.services.venv_compat import (
    CompatStatus,
    check_compat,
    find_venv_root,
    read_installed_versions,
)

if TYPE_CHECKING:
    from gpumod.models import Service

doctor_app = typer.Typer(
    name="doctor",
    help="System health checks (venv compat, etc.).",
)


@doctor_app.command("venv")
def venv_command(
    service_id: str = typer.Option(
        ...,
        "--service-id",
        help="ID of the service whose driver venv to validate.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help=(
            "Override the service's inline compat with a shipped profile "
            "(e.g. 'vllm-0.11'). See `gpumod compat list` for available "
            "profiles."
        ),
    ),
) -> None:
    """Validate the installed packages in a service's driver venv."""
    from gpumod.cli import cli_context, run_async
    from gpumod.services.unit_installer import _build_settings

    async def _cmd() -> None:
        async with cli_context(no_sync=True) as ctx:
            try:
                service = await ctx.registry.get(service_id)
            except KeyError as exc:
                sys.stderr.write(f"{exc!s}\n")
                raise typer.Exit(code=2) from None

            declared = _resolve_declared(service, profile)
            if not declared:
                # No contract — no-op success
                raise typer.Exit(code=0)

            # Load settings from DB so settings.vllm_bin (operator-set, points
            # at the dedicated venv) participates in the resolution order.
            # Without this, find_venv_root falls through to shutil.which() and
            # checks the wrong venv — turning the safeguard into a false alarm.
            settings = await _build_settings(ctx.db)
            venv = find_venv_root(service, settings=settings)
            if venv is None:
                sys.stderr.write(
                    f"doctor venv: could not resolve venv root for "
                    f"service {service_id!r}. Set unit_vars.vllm_bin in "
                    "the preset or settings.vllm_bin globally.\n"
                )
                raise typer.Exit(code=1)

            installed = read_installed_versions(venv)
            result = check_compat(declared, installed)
            if result.ok:
                raise typer.Exit(code=0)

            for v in result.violations:
                label = "MISSING" if v.status == CompatStatus.MISSING else "DRIFT"
                installed_str = "(not installed)" if v.installed is None else v.installed
                sys.stderr.write(
                    f"[{label}] {v.package}: installed={installed_str} constraint={v.constraint}\n"
                )
            raise typer.Exit(code=1)

    run_async(_cmd())


def _resolve_declared(service: Service, profile_name: str | None) -> dict[str, str]:
    """Pick the active compat contract: --profile wins; else service.compat."""
    if profile_name is not None:
        from gpumod.cli_compat import load_profile

        try:
            return load_profile(profile_name)
        except FileNotFoundError as exc:
            sys.stderr.write(f"{exc!s}\n")
            raise typer.Exit(code=1) from None
    return service.compat or {}
