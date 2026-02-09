#!/usr/bin/env python3
"""Benchmark mode switch latency for gpumod.

Measures mode transition times and compares sleep-aware vs restart-based
switching. Outputs results in markdown format for documentation.

Usage:
    # Run all mode transitions (3 runs each)
    uv run python scripts/benchmark_mode_switch.py

    # Custom transitions with more runs
    uv run python scripts/benchmark_mode_switch.py \
        --transitions "code→rag,rag→code,speak→code" \
        --runs 5 --warmup 2

    # Output to file
    uv run python scripts/benchmark_mode_switch.py \
        --output docs/research/vllm-sleep-mode.md

Requirements:
    - GPU with nvidia-smi available
    - gpumod services configured
    - Systemd user services installed
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
from pathlib import Path

from gpumod.benchmark.mode_switch import (
    ModeSwitchBenchmark,
    ModeSwitchResult,
    TransitionStats,
    generate_comparison_table,
)
from gpumod.db import Database
from gpumod.services.lifecycle import LifecycleManager
from gpumod.services.manager import ServiceManager
from gpumod.services.registry import ServiceRegistry
from gpumod.services.sleep import SleepController
from gpumod.services.unit_installer import UnitFileInstaller
from gpumod.services.vram import VRAMTracker

# Default transitions to benchmark
DEFAULT_TRANSITIONS = [
    ("blank", "code"),
    ("code", "rag"),
    ("rag", "code"),
    ("code", "speak"),
    ("speak", "code"),
]


def parse_transitions(spec: str) -> list[tuple[str, str]]:
    """Parse transition spec like 'code→rag,rag→code'."""
    transitions = []
    for part in spec.split(","):
        part = part.strip()
        if "→" in part:
            from_mode, to_mode = part.split("→", 1)
        elif "->" in part:
            from_mode, to_mode = part.split("->", 1)
        else:
            raise ValueError(f"Invalid transition format: {part!r}")
        transitions.append((from_mode.strip(), to_mode.strip()))
    return transitions


async def create_manager() -> ServiceManager:
    """Create a real ServiceManager from default config."""
    db = await Database.create()
    registry = ServiceRegistry.load_from_db(db)
    unit_installer = UnitFileInstaller()
    lifecycle = LifecycleManager(registry, unit_installer=unit_installer)
    vram = VRAMTracker(registry)
    sleep = SleepController(registry)

    return ServiceManager(
        db=db,
        registry=registry,
        lifecycle=lifecycle,
        vram=vram,
        sleep=sleep,
    )


def group_results(
    results: list[ModeSwitchResult],
) -> dict[str, list[ModeSwitchResult]]:
    """Group results by transition."""
    groups: dict[str, list[ModeSwitchResult]] = {}
    for r in results:
        key = r.transition
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    return groups


def _print_header(
    transitions: list[tuple[str, str]],
    warmup_runs: int,
    measured_runs: int,
) -> None:
    """Print benchmark header."""
    print("Running mode switch benchmark")
    print(f"  Transitions: {len(transitions)}")
    print(f"  Warmup runs: {warmup_runs}")
    print(f"  Measured runs: {measured_runs}")
    print()


async def _run_single_transition(
    benchmark: ModeSwitchBenchmark,
    from_mode: str,
    to_mode: str,
    warmup_runs: int,
    measured_runs: int,
    verbose: bool,
) -> list[ModeSwitchResult]:
    """Run warmup and measured iterations for a single transition."""
    results: list[ModeSwitchResult] = []

    if verbose:
        print(f"Benchmarking {from_mode} → {to_mode}...")

    # Warmup
    for i in range(warmup_runs):
        if verbose:
            print(f"  Warmup {i + 1}/{warmup_runs}...", end=" ", flush=True)
        result = await benchmark.run_transition(from_mode, to_mode)
        if verbose:
            print(f"{result.total_ms:.0f}ms")

    # Measured runs
    for i in range(measured_runs):
        if verbose:
            print(f"  Run {i + 1}/{measured_runs}...", end=" ", flush=True)
        result = await benchmark.run_transition(from_mode, to_mode)
        results.append(result)
        if verbose:
            sleep_flag = " [sleep]" if result.is_sleep_capable else ""
            print(f"{result.total_ms:.0f}ms{sleep_flag}")

    if verbose:
        print()

    return results


async def run_benchmark(
    transitions: list[tuple[str, str]],
    warmup_runs: int,
    measured_runs: int,
    verbose: bool = True,
) -> list[TransitionStats]:
    """Run the mode switch benchmark."""
    manager = await create_manager()
    benchmark = ModeSwitchBenchmark(manager=manager)

    if verbose:
        _print_header(transitions, warmup_runs, measured_runs)

    all_results: list[ModeSwitchResult] = []

    for from_mode, to_mode in transitions:
        results = await _run_single_transition(
            benchmark, from_mode, to_mode, warmup_runs, measured_runs, verbose
        )
        all_results.extend(results)

    # Compute stats per transition
    groups = group_results(all_results)
    return [TransitionStats.from_results(results) for results in groups.values()]


def generate_report(
    stats: list[TransitionStats],
    baseline: list[TransitionStats] | None = None,
) -> str:
    """Generate a markdown report from benchmark results."""
    now = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Mode Switch Benchmark Results",
        "",
        f"**Date:** {now}",
        "",
        "## Results",
        "",
    ]

    if baseline:
        lines.append(generate_comparison_table(baseline, stats))
    else:
        # Simple results table
        lines.extend(
            [
                "| Transition | Runs | Min (ms) | Median (ms) | Max (ms) | Sleep-Capable |",
                "|------------|------|----------|-------------|----------|---------------|",
            ]
        )
        for s in stats:
            sleep = "Yes" if s.is_sleep_capable else "No"
            lines.append(
                f"| {s.transition} | {s.runs} | {s.min_ms:.0f} | "
                f"{s.median_ms:.0f} | {s.max_ms:.0f} | {sleep} |"
            )

    lines.extend(
        [
            "",
            "## Analysis",
            "",
        ]
    )

    # Calculate summary stats
    sleep_transitions = [s for s in stats if s.is_sleep_capable]
    restart_transitions = [s for s in stats if not s.is_sleep_capable]

    if sleep_transitions:
        avg_sleep = sum(s.median_ms for s in sleep_transitions) / len(sleep_transitions)
        lines.append(f"- **Sleep-capable transitions avg:** {avg_sleep:.0f}ms")

    if restart_transitions:
        avg_restart = sum(s.median_ms for s in restart_transitions) / len(restart_transitions)
        lines.append(f"- **Restart transitions avg:** {avg_restart:.0f}ms")

    if sleep_transitions and restart_transitions:
        speedup = avg_restart / avg_sleep if avg_sleep > 0 else 0
        lines.append(f"- **Speedup factor:** {speedup:.1f}x")

    # Check against target
    lines.extend(
        [
            "",
            "## Success Criteria",
            "",
        ]
    )

    target_ms = 5000
    passing = all(s.median_ms < target_ms for s in sleep_transitions)
    status = "✅ PASS" if passing else "❌ FAIL"

    lines.append(f"Target: Sleep-capable transitions < {target_ms}ms")
    lines.append(f"Status: {status}")

    if not passing:
        failing = [s for s in sleep_transitions if s.median_ms >= target_ms]
        lines.extend(f"  - {s.transition}: {s.median_ms:.0f}ms (over target)" for s in failing)

    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark mode switch latency for gpumod")
    parser.add_argument(
        "--transitions",
        type=str,
        help="Comma-separated transitions (e.g., 'code→rag,rag→code')",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of measured runs per transition (default: 3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs per transition (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for markdown report",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="JSON file with baseline stats for comparison",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        help="Save raw results as JSON (for use as future baseline)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output",
    )
    args = parser.parse_args()

    # Parse transitions
    transitions = parse_transitions(args.transitions) if args.transitions else DEFAULT_TRANSITIONS

    # Load baseline if provided
    baseline = None
    if args.baseline and args.baseline.exists():
        data = json.loads(args.baseline.read_text())
        baseline = [TransitionStats(**s) for s in data.get("stats", [])]

    # Run benchmark
    try:
        stats = await run_benchmark(
            transitions=transitions,
            warmup_runs=args.warmup,
            measured_runs=args.runs,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"Benchmark failed: {e}", file=sys.stderr)
        return 1

    # Save JSON if requested
    if args.save_json:
        data = {
            "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
            "stats": [
                {
                    "transition": s.transition,
                    "runs": s.runs,
                    "min_ms": s.min_ms,
                    "max_ms": s.max_ms,
                    "mean_ms": s.mean_ms,
                    "median_ms": s.median_ms,
                    "is_sleep_capable": s.is_sleep_capable,
                }
                for s in stats
            ],
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(data, indent=2))
        print(f"Saved JSON: {args.save_json}")

    # Generate report
    report = generate_report(stats, baseline)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Saved report: {args.output}")
    else:
        print()
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
