#!/usr/bin/env python3
"""Aggregate Phase 2 per-config JSON summaries into a single markdown table
plus a Phase 2 results blob suitable for committing.

Usage:
    phase2_aggregate.py phase2_results/*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: phase2_aggregate.py <json...>\n")
        return 1

    rows = []
    for path_str in sorted(sys.argv[1:]):
        path = Path(path_str)
        with path.open() as f:
            data = json.load(f)
        rows.append((path.stem, data))

    # ------------------------------------------------------------------
    # Per-config summary table
    # ------------------------------------------------------------------
    print("## Phase 2 — per-config rollup")
    print()
    print(
        "| Config | Slots | Agg TPS | Turns OK | Turns Err | Avg per-call TPS | "
        "p50 turn (s) | p95 turn (s) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for stem, data in rows:
        n = data.get("n_agents", "?")
        agg_tps = data.get("aggregate_tps_window", 0)
        ok = data.get("turns_ok", 0)
        err = data.get("turns_err", 0)
        all_per_call = []
        all_p50, all_p95 = [], []
        for agent_name, agent in data.get("per_agent", {}).items():
            if not isinstance(agent, dict):
                continue
            if "tps_per_call_mean" in agent:
                all_per_call.append(agent["tps_per_call_mean"])
            if "turn_duration_p50" in agent:
                all_p50.append(agent["turn_duration_p50"])
            if "turn_duration_p95" in agent:
                all_p95.append(agent["turn_duration_p95"])
        avg_per_call = sum(all_per_call) / len(all_per_call) if all_per_call else 0
        p50 = sum(all_p50) / len(all_p50) if all_p50 else 0
        p95 = sum(all_p95) / len(all_p95) if all_p95 else 0
        print(
            f"| {stem} | {n} | {agg_tps:.1f} | {ok} | {err} | "
            f"{avg_per_call:.1f} | {p50:.1f} | {p95:.1f} |"
        )

    # ------------------------------------------------------------------
    # Per-agent detail per config
    # ------------------------------------------------------------------
    print()
    print("## Phase 2 — per-agent detail")
    print()
    for stem, data in rows:
        print(f"### {stem}")
        print()
        print(
            "| Agent | Turns OK | Tokens total | Tokens/turn | "
            "TPS (per-call) | TPS (window) | p50 turn (s) | p95 turn (s) |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for agent_name, agent in data.get("per_agent", {}).items():
            if not isinstance(agent, dict) or "turns_ok" not in agent:
                continue
            if agent["turns_ok"] == 0:
                print(
                    f"| {agent_name} | 0 | 0 | — | — | — | — | — |"
                )
                continue
            print(
                f"| {agent_name} | {agent['turns_ok']} | {agent['tokens_total']} | "
                f"{agent['tokens_per_turn_mean']:.0f} | "
                f"{agent['tps_per_call_mean']:.1f} | "
                f"{agent.get('throughput_tps_window', 0):.1f} | "
                f"{agent['turn_duration_p50']:.1f} | "
                f"{agent['turn_duration_p95']:.1f} |"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
