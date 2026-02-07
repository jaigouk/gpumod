"""Plan CLI commands -- gpumod plan suggest.

Provides AI-assisted VRAM planning using LLM backends.
Plans are purely advisory (SEC-L4) -- they show suggested commands
but never auto-execute service changes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.table import Table

from gpumod.config import get_settings
from gpumod.llm import get_backend
from gpumod.llm.prompts import PLANNING_SYSTEM_PROMPT, build_planning_prompt
from gpumod.llm.response_validator import PlanSuggestion, validate_plan_response

if TYPE_CHECKING:
    from gpumod.models import Service, SimulationResult

plan_app = typer.Typer(name="plan", help="AI-assisted VRAM planning.")

_console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_api_key(settings: Any) -> None:  # noqa: ANN401
    """Check that an API key is configured for non-Ollama backends.

    Raises
    ------
    ValueError
        If no API key is configured and the backend requires one.
    """
    if settings.llm_backend == "ollama":
        return
    if settings.llm_api_key is None:
        msg = (
            "No LLM API key configured. "
            "Set GPUMOD_LLM_API_KEY or use --dry-run to preview the prompt."
        )
        raise ValueError(msg)


def _build_service_dicts(services: list[Service]) -> list[dict[str, Any]]:
    """Convert Service models to minimal dicts for prompt building."""
    return [{"id": svc.id, "name": svc.name, "vram_mb": svc.vram_mb} for svc in services]


def _render_plan_table(
    plan: PlanSuggestion,
    sim_result: SimulationResult,
    *,
    console: Console,
) -> None:
    """Render the plan as a Rich table with simulation results.

    Parameters
    ----------
    plan:
        The validated LLM plan suggestion.
    sim_result:
        The simulation result for the proposed plan.
    console:
        The Rich console for output.
    """
    # Simulation summary
    if sim_result.fits:
        console.print(
            f"[bold green]Fits[/bold green]: "
            f"{sim_result.proposed_usage_mb} / {sim_result.gpu_total_mb} MB "
            f"(headroom: {sim_result.headroom_mb} MB)"
        )
    else:
        console.print(
            f"[bold red]Does not fit[/bold red]: "
            f"{sim_result.proposed_usage_mb} / {sim_result.gpu_total_mb} MB "
            f"(over by {-sim_result.headroom_mb} MB)"
        )

    # Plan table
    table = Table(title="AI-Suggested VRAM Plan")
    table.add_column("Service ID", style="cyan")
    table.add_column("VRAM (MB)", justify="right")

    total_vram = 0
    for alloc in plan.services:
        table.add_row(alloc.service_id, str(alloc.vram_mb))
        total_vram += alloc.vram_mb

    table.add_row("[bold]Total[/bold]", f"[bold]{total_vram}[/bold]")
    console.print(table)

    # Reasoning
    console.print(f"\n[bold]Reasoning:[/bold] {plan.reasoning}")

    # Advisory commands (SEC-L4)
    console.print("\n[bold yellow]Suggested commands (advisory only):[/bold yellow]")
    service_ids = ",".join(a.service_id for a in plan.services)
    console.print(f"  gpumod simulate services {service_ids}")
    for alloc in plan.services:
        console.print(f"  gpumod service start {alloc.service_id}")


def _build_json_output(
    plan: PlanSuggestion,
    sim_result: SimulationResult,
) -> dict[str, Any]:
    """Build the JSON output dict for --json mode."""
    return {
        "plan": {
            "services": [
                {"service_id": a.service_id, "vram_mb": a.vram_mb} for a in plan.services
            ],
            "reasoning": plan.reasoning,
        },
        "simulation": sim_result.model_dump(mode="json"),
        "advisory_commands": [
            f"gpumod simulate services {','.join(a.service_id for a in plan.services)}",
            *[f"gpumod service start {a.service_id}" for a in plan.services],
        ],
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@plan_app.command("suggest")
def suggest(
    mode: str | None = typer.Option(None, "--mode", help="Plan for a specific mode."),
    budget: int | None = typer.Option(None, "--budget", help="VRAM budget in MB."),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show prompt, don't call LLM."),
) -> None:
    """Get AI-suggested VRAM allocation plan."""
    from gpumod.cli import cli_context, error_handler, run_async

    async def _cmd() -> None:
        async with cli_context() as ctx:
            with error_handler(console=_console):
                settings = get_settings()

                # 1. Resolve services: from mode or from all DB services
                if mode is not None:
                    svc_mode = await ctx.db.get_mode(mode)
                    if svc_mode is None:
                        msg = f"Mode not found: {mode!r}"
                        raise ValueError(msg)
                    services: list[Service] = await ctx.db.get_mode_services(mode)
                else:
                    services = await ctx.db.list_services()

                if not services:
                    _console.print("[dim]No services registered.[/dim]")
                    return

                # 2. Get GPU info
                gpu_info = await ctx.vram.get_gpu_info()
                gpu_total_mb = gpu_info.vram_total_mb

                # 3. Get current mode
                current_mode = await ctx.db.get_setting("current_mode")

                # 4. Build prompt
                service_dicts = _build_service_dicts(services)
                prompt = build_planning_prompt(
                    services=service_dicts,
                    gpu_total_mb=gpu_total_mb,
                    current_mode=current_mode,
                    budget_mb=budget,
                )

                # 5. Dry run -- show prompt and return
                if dry_run:
                    _console.print("[bold]System prompt:[/bold]")
                    _console.print(PLANNING_SYSTEM_PROMPT)
                    _console.print("\n[bold]User prompt:[/bold]")
                    _console.print(prompt)
                    return

                # 6. Check API key (unless ollama)
                _check_api_key(settings)

                # 7. Call LLM backend
                backend = get_backend(settings)
                raw_response = await backend.generate(
                    prompt=prompt,
                    system=PLANNING_SYSTEM_PROMPT,
                    response_schema=PlanSuggestion,
                )

                # 8. Validate response (SEC-L1)
                plan = validate_plan_response(raw_response)

                # 9. Run simulation to verify plan fits GPU
                sim_result: SimulationResult = await ctx.simulation.simulate_services(
                    [a.service_id for a in plan.services],
                )

                # 10. Display results
                if as_json:
                    output = _build_json_output(plan, sim_result)
                    print(json.dumps(output, indent=2, default=str))  # noqa: T201
                    return

                _render_plan_table(plan, sim_result, console=_console)

    run_async(_cmd())
