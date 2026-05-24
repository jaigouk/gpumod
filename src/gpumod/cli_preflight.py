"""Preflight CLI commands — gpumod preflight all / ram (gpumod-aeb, gpumod-ecr).

These commands are invoked from systemd unit `ExecStartPre=` hooks to
short-circuit a service start when the system isn't safe for a large
CUDA-pinned allocation. They're independent of gpumod's Python API
safeguard (which only fires on `gpumod start_service` / `mode switch`).

Exit codes:
  0 -- safe to start
  1 -- unsafe; diagnostic on stderr
  2 -- service id unknown (or other operator error)
"""

from __future__ import annotations

import sys

import typer

from gpumod.preflight import PreflightRunner
from gpumod.services.ram import RAMTracker, check_ram_safeguard
from gpumod.services.running_context import format_running_services

preflight_app = typer.Typer(
    name="preflight",
    help="Pre-start safety checks (called from systemd ExecStartPre).",
)


@preflight_app.command("all")
def all_command(
    service_id: str = typer.Option(
        ...,
        "--service-id",
        help="ID of the service about to start.",
    ),
) -> None:
    """Run the full preflight pipeline (model-file, VRAM, RAM) for a service.

    Exit codes:
      0 — all checks pass (safe to start)
      1 — at least one check failed; diagnostic on stderr
      2 — service id unknown or other operator error
    """
    from gpumod.cli import cli_context, run_async

    async def _cmd() -> None:
        async with cli_context(no_sync=True) as ctx:
            try:
                service = await ctx.registry.get(service_id)
            except KeyError as exc:
                sys.stderr.write(f"{exc!s}\n")
                raise typer.Exit(code=2) from None

            runner = PreflightRunner.startup_only()
            results = await runner.run_all(service)

            if runner.has_errors(results):
                error_msg = runner.format_errors(results)
                context_block = await format_running_services(ctx.registry, exclude_id=service.id)
                sys.stderr.write(f"{error_msg}\n")
                if context_block:
                    sys.stderr.write(f"\n{context_block}\n")
                raise typer.Exit(code=1)

            raise typer.Exit(code=0)

    run_async(_cmd())


@preflight_app.command("ram")
def ram_command(
    service_id: str = typer.Option(
        ...,
        "--service-id",
        help="ID of the service about to start.",
    ),
    hard_floor: int | None = typer.Option(
        None,
        "--hard-floor",
        help=(
            "Override the default 6000 MB MemAvailable hard floor. "
            "Useful for testing or for tighter operational policies."
        ),
        min=0,
    ),
) -> None:
    """Verify it's safe to start a service (deprecated; prefer 'preflight all').

    Reads the service's declared ``vram_mb`` from the gpumod registry,
    computes the required MemAvailable as
    ``vram_mb * 0.3 + 5000 MB`` (or the hard floor, whichever is larger),
    and refuses to exit 0 if the current MemAvailable is below that.

    Designed to be invoked from a systemd unit::

        ExecStartPre={{ settings.gpumod_bin | default('gpumod') }} \\
            preflight ram --service-id {{ service.id }}
    """
    from gpumod.cli import cli_context, run_async

    async def _cmd() -> None:
        async with cli_context(no_sync=True) as ctx:
            try:
                service = await ctx.registry.get(service_id)
            except KeyError as exc:
                # Unknown service id — operator error, not a runtime failure
                sys.stderr.write(f"{exc!s}\n")
                raise typer.Exit(code=2) from None

            ram = RAMTracker()
            error = await check_ram_safeguard(
                incoming_vram_mb=service.vram_mb,
                ram=ram,
                hard_floor_mb=hard_floor,
            )
            if error is None:
                # Safe; quiet success — systemd only cares about exit code.
                raise typer.Exit(code=0)
            sys.stderr.write(f"{error}\n")
            raise typer.Exit(code=1)

    run_async(_cmd())
