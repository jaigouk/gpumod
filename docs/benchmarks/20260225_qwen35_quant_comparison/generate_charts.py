#!/usr/bin/env python3
"""Generate comparison charts for the Qwen3.5 quantization comparison benchmark.

Usage:
    uv run python docs/benchmarks/20260225_qwen35_quant_comparison/generate_charts.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
COLORS = {
    "Qwen3.5-35B Q4_K_XL": "#2196F3",
    "Qwen3.5-35B Q3_K_XL": "#FF5722",
}
LABELS = {
    "Qwen3.5-35B Q4_K_XL": "Q4_K_XL (22GB)",
    "Qwen3.5-35B Q3_K_XL": "Q3_K_XL (18GB)",
}


def load_data() -> dict[str, dict]:
    """Load all benchmark JSON files from the current directory."""
    results = {}
    for f in sorted(HERE.glob("20260225_qwen3.5*.json")):
        data = json.loads(f.read_text())
        model = data["model"]
        results[model] = data
    return results


# ---------------------------------------------------------------------------
# Radar chart — 6 axes comparing models
# ---------------------------------------------------------------------------

RADAR_AXES = [
    "Solo Speed\n(tok/s)",
    "TTFT\nStability",
    "Latency\nUnder Load",
    "Side Task\nAccuracy",
    "Throughput\nScaling",
    "VRAM\nEfficiency",
]


def _extract_radar_values(data: dict, model: str) -> list[float]:
    """Extract and normalize metrics to 0-5 scale for radar chart."""
    levels = {lv["concurrency"]: lv for lv in data["concurrency_levels"]}

    # 1. Solo Speed: tok/s at concurrency 1, primary turn 3 (steady state)
    solo_toks = levels[1]["primary"][2]["tokens_per_sec"]
    # Normalize: 100 tok/s = 5, 0 = 0
    solo_score = min(solo_toks / 20, 5.0)

    # 2. TTFT Stability: how little TTFT grows from concurrency 1 to 3
    ttft_1 = levels[1]["primary"][0]["ttft_ms"] or 100
    ttft_3 = levels[3]["primary"][0]["ttft_ms"] or 300
    ttft_ratio = ttft_3 / ttft_1  # lower is better
    # 1.0x = 5, 4.0x = 1
    ttft_score = max(1.0, min(5.0, 6.0 - ttft_ratio))

    # 3. Latency Under Load: total time at concurrency 3 (lower is better)
    total_3 = levels[3]["total_time_ms"]
    # 20000ms = 5, 30000ms = 1
    latency_score = max(1.0, min(5.0, 5.0 - (total_3 - 20000) / 2500))

    # 4. Side Task Accuracy: pass rate at concurrency 3
    sides_3 = levels[3].get("side_tasks", [])
    if sides_3:
        passed = sum(1 for s in sides_3 if s.get("verified", False))
        pass_rate = passed / len(sides_3)
    else:
        pass_rate = 0
    accuracy_score = pass_rate * 5

    # 5. Throughput Scaling: aggregate tok/s at concurrency 3
    all_toks_3 = sum(
        p["tokens_generated"]
        for p in levels[3]["primary"] + levels[3].get("side_tasks", [])
    )
    agg_toks = all_toks_3 / (levels[3]["total_time_ms"] / 1000)
    # 100 tok/s = 5, 0 = 0
    throughput_score = min(agg_toks / 20, 5.0)

    # 6. VRAM Efficiency: based on known VRAM usage
    vram = {"Qwen3.5-35B Q4_K_XL": 22000, "Qwen3.5-35B Q3_K_XL": 18000}.get(model, 20000)
    # 14GB = 5, 24GB = 1
    vram_score = max(1.0, min(5.0, 5.0 - (vram - 14000) / 2500))

    return [solo_score, ttft_score, latency_score, accuracy_score, throughput_score, vram_score]


def generate_radar(datasets: dict[str, dict], output: Path) -> None:
    """Radar chart comparing models across 6 dimensions."""
    n = len(RADAR_AXES)
    angles = [i / n * 2 * math.pi for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for model, data in datasets.items():
        values = _extract_radar_values(data, model)
        values += values[:1]
        color = COLORS.get(model, "#666")
        label = LABELS.get(model, model)
        ax.plot(angles, values, "o-", linewidth=2.5, label=label, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_AXES, fontsize=11)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=9)
    ax.set_title(
        "Qwen3.5-35B: Q4 vs Q3 Quantization",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=12)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Radar chart: {output}")


# ---------------------------------------------------------------------------
# Degradation line chart — TTFT and total time vs concurrency
# ---------------------------------------------------------------------------


def generate_degradation(datasets: dict[str, dict], output: Path) -> None:
    """Line charts showing TTFT and total time vs concurrency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    concurrency_levels = [1, 2, 3]

    for model, data in datasets.items():
        color = COLORS.get(model, "#666")
        label = LABELS.get(model, model)
        levels = {lv["concurrency"]: lv for lv in data["concurrency_levels"]}

        # TTFT (primary turn 1)
        ttfts = [levels[c]["primary"][0].get("ttft_ms") or 0 for c in concurrency_levels]
        ax1.plot(
            concurrency_levels, ttfts, "o-", linewidth=2.5, color=color,
            label=label, markersize=8,
        )
        for x, y in zip(concurrency_levels, ttfts):
            ax1.annotate(f"{y:.0f}ms", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9, color=color)

        # Total time
        totals = [levels[c]["total_time_ms"] for c in concurrency_levels]
        ax2.plot(
            concurrency_levels, totals, "s-", linewidth=2.5, color=color,
            label=label, markersize=8,
        )
        for x, y in zip(concurrency_levels, totals):
            ax2.annotate(f"{y / 1000:.1f}s", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9, color=color)

    for ax in (ax1, ax2):
        ax.set_xlabel("Concurrent Slots", fontsize=12)
        ax.set_xticks(concurrency_levels)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("TTFT (ms)", fontsize=12)
    ax1.set_title("TTFT Degradation (Primary Turn 1)", fontsize=13, fontweight="bold")

    ax2.set_ylabel("Total Time (ms)", fontsize=12)
    ax2.set_title("Total Workload Time", fontsize=13, fontweight="bold")

    fig.suptitle(
        "Performance vs Concurrency — Q4 vs Q3 Quantization",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Degradation chart: {output}")


# ---------------------------------------------------------------------------
# Throughput comparison bar chart
# ---------------------------------------------------------------------------


def generate_throughput(datasets: dict[str, dict], output: Path) -> None:
    """Bar chart comparing throughput across concurrency levels."""
    fig, ax = plt.subplots(figsize=(10, 6))

    concurrency_levels = [1, 2, 3]
    x = np.arange(len(concurrency_levels))
    width = 0.35
    models = list(datasets.keys())

    for i, model in enumerate(models):
        data = datasets[model]
        levels = {lv["concurrency"]: lv for lv in data["concurrency_levels"]}

        # tok/s for primary turn 3 (final turn, most comparable)
        toks = [levels[c]["primary"][2]["tokens_per_sec"] for c in concurrency_levels]

        color = COLORS.get(model, "#666")
        label = LABELS.get(model, model)
        offset = x + (i - 0.5) * width
        bars = ax.bar(offset, toks, width, label=label, color=color, alpha=0.85)

        # Add value labels
        for bar, t in zip(bars, toks):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{t:.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"C={c}" for c in concurrency_levels], fontsize=11)
    ax.set_ylabel("Tokens/sec (Turn 3)", fontsize=12)
    ax.set_title(
        "Throughput by Concurrency Level",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Throughput chart: {output}")


# ---------------------------------------------------------------------------
# Side task bar chart
# ---------------------------------------------------------------------------


def generate_side_tasks(datasets: dict[str, dict], output: Path) -> None:
    """Grouped bar chart of side task response times at concurrency 3."""
    fig, ax = plt.subplots(figsize=(10, 6))

    task_ids = ["S1_docstring", "S2_unit_tests"]
    task_labels = ["S1\nDocstring", "S2\nUnit Tests"]

    x = np.arange(len(task_ids))
    width = 0.35
    models = list(datasets.keys())

    for i, model in enumerate(models):
        data = datasets[model]
        levels = {lv["concurrency"]: lv for lv in data["concurrency_levels"]}
        sides = {s["role"]: s for s in levels[3].get("side_tasks", [])}

        times = [sides.get(tid, {}).get("response_time_ms", 0) for tid in task_ids]
        verified = [sides.get(tid, {}).get("verified", None) for tid in task_ids]

        color = COLORS.get(model, "#666")
        label = LABELS.get(model, model)
        offset = x + (i - 0.5) * width
        bars = ax.bar(offset, times, width, label=label, color=color, alpha=0.85)

        # Mark pass/fail
        for bar, v, t in zip(bars, verified, times):
            if t == 0:
                continue
            marker = "PASS" if v else "FAIL*"
            marker_color = "#2E7D32" if v else "#C62828"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 80,
                marker,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=marker_color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=11)
    ax.set_ylabel("Response Time (ms)", fontsize=12)
    ax.set_title(
        "Side Task Performance at Concurrency 3\n*FAIL = thinking model keyword verification artifact",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Side tasks chart: {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    datasets = load_data()
    if not datasets:
        print("No benchmark JSON files found.")
        return

    print(f"Loaded {len(datasets)} model(s): {', '.join(datasets.keys())}")

    charts_dir = HERE / "charts"
    charts_dir.mkdir(exist_ok=True)

    generate_radar(datasets, charts_dir / "radar.png")
    generate_degradation(datasets, charts_dir / "degradation.png")
    generate_throughput(datasets, charts_dir / "throughput.png")
    generate_side_tasks(datasets, charts_dir / "side_tasks.png")

    print(f"\nAll charts saved to {charts_dir}/")


if __name__ == "__main__":
    main()
