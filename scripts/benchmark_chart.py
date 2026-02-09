#!/usr/bin/env python3
"""Generate comparison charts from benchmark JSON results.

Usage:
    uv run python scripts/benchmark_chart.py \
        docs/benchmarks/20260207_nemotron_devstral/20260207_*.json \
        --output docs/benchmarks/20260207_nemotron_devstral/charts/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Radar chart — quality scores across 5 categories
# ---------------------------------------------------------------------------

CATEGORIES = [
    "factual",
    "reasoning",
    "code",
    "tool_use",
    "writing",
]

CATEGORY_LABELS = [
    "Factual\nKnowledge",
    "Reasoning\n& Logic",
    "Code\nGeneration",
    "Tool Use /\nStructured",
    "Writing &\nSummarization",
]

COLORS = [
    "#2196F3",  # blue
    "#FF5722",  # red-orange
    "#4CAF50",  # green
    "#9C27B0",  # purple
    "#FF9800",  # orange
]


def _category_avg(data: dict, category: str) -> float:
    """Average quality_score for a category, or 0 if unscored."""
    scores = [
        r["quality_score"]
        for r in data.get("results", [])
        if r["category"] == category and r.get("quality_score") is not None
    ]
    return sum(scores) / len(scores) if scores else 0.0


def generate_radar_chart(datasets: list[dict], output_path: Path) -> None:
    """Radar chart comparing quality scores across categories."""
    n = len(CATEGORIES)
    angles = [i / n * 2 * math.pi for i in range(n)]
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for idx, data in enumerate(datasets):
        name = data["metadata"]["short_name"]
        values = [_category_avg(data, cat) for cat in CATEGORIES]
        values += values[:1]
        color = COLORS[idx % len(COLORS)]
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATEGORY_LABELS, fontsize=11)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=9)
    ax.set_title("Quality Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Radar chart: {output_path}")


# ---------------------------------------------------------------------------
# Grouped bar chart — performance metrics
# ---------------------------------------------------------------------------

PERF_METRICS = [
    ("avg_ttft_ms", "Avg TTFT (ms)"),
    ("avg_gen_tokens_per_sec", "Gen tok/s"),
    ("load_ms", "Load time (ms)"),
]


def generate_performance_chart(datasets: list[dict], output_path: Path) -> None:
    """Grouped bar chart comparing performance metrics."""
    model_names = [d["metadata"]["short_name"] for d in datasets]
    n_models = len(model_names)
    n_metrics = len(PERF_METRICS)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for i, (key, label) in enumerate(PERF_METRICS):
        ax = axes[i]
        values = []
        for d in datasets:
            summary = d.get("summary", {})
            values.append(summary.get(key, 0))

        x = np.arange(n_models)
        bars = ax.bar(
            x,
            values,
            color=[COLORS[j % len(COLORS)] for j in range(n_models)],
            width=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel(label)

        # Add value labels
        for bar, val in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.suptitle("Performance Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Performance chart: {output_path}")


# ---------------------------------------------------------------------------
# Lifecycle (mode-switching) chart
# ---------------------------------------------------------------------------


def generate_lifecycle_chart(lifecycle_data: dict, output_path: Path) -> None:
    """Stacked bar chart of mode-switching times."""
    lifecycle = lifecycle_data.get("lifecycle", {})
    if not lifecycle:
        print("  No lifecycle data found, skipping chart.")
        return

    models = list(lifecycle.keys())
    transitions = [s["transition"] for s in lifecycle[models[0]]]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(transitions))
    width = 0.8 / len(models)

    for idx, model in enumerate(models):
        steps = lifecycle[model]
        unload = [s["unload_ms"] for s in steps]
        load = [s["load_ms"] for s in steps]
        verify = [s["first_request_total_ms"] for s in steps]

        offset = x + idx * width - 0.4 + width / 2
        color = COLORS[idx % len(COLORS)]
        sname = model.split("-")[0]

        ax.bar(offset, unload, width, label=f"{sname} unload", color=color, alpha=0.5)
        ax.bar(
            offset,
            load,
            width,
            bottom=unload,
            label=f"{sname} load",
            color=color,
            alpha=0.75,
        )
        ax.bar(
            offset,
            verify,
            width,
            bottom=[u + ld for u, ld in zip(unload, load, strict=True)],
            label=f"{sname} verify",
            color=color,
            alpha=1.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(transitions, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Mode-Switching Lifecycle", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=len(models), loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Lifecycle chart: {output_path}")


# ---------------------------------------------------------------------------
# Hallucination chart
# ---------------------------------------------------------------------------


def generate_hallucination_chart(datasets: list[dict], output_path: Path) -> None:
    """Bar chart showing hallucination results per model."""
    model_names = [d["metadata"]["short_name"] for d in datasets]

    factual_results: dict[str, dict] = {}
    for d in datasets:
        name = d["metadata"]["short_name"]
        checked = correct = hallucinated = missing = 0
        for r in d.get("results", []):
            h = r.get("hallucination")
            if h:
                checked += h["facts_checked"]
                correct += h["facts_correct"]
                hallucinated += h["facts_hallucinated"]
                missing += h["facts_missing"]
        factual_results[name] = {
            "correct": correct,
            "hallucinated": hallucinated,
            "missing": missing,
        }

    if not factual_results:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(model_names))
    width = 0.25

    corrects = [factual_results[n]["correct"] for n in model_names]
    misses = [factual_results[n]["missing"] for n in model_names]
    hallucinations = [factual_results[n]["hallucinated"] for n in model_names]

    ax.bar(x - width, corrects, width, label="Correct", color="#4CAF50")
    ax.bar(x, misses, width, label="Missing", color="#FFC107")
    ax.bar(x + width, hallucinations, width, label="Hallucinated", color="#F44336")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Number of facts")
    ax.set_title("Hallucination Detection Results", fontsize=14, fontweight="bold")
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Hallucination chart: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark comparison charts")
    parser.add_argument("files", nargs="+", help="Benchmark result JSON files")
    parser.add_argument(
        "--output",
        default="charts/",
        help="Output directory for charts",
    )
    parser.add_argument(
        "--lifecycle",
        help="Lifecycle JSON file (separate from model results)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Load model benchmark results (skip lifecycle files)
    datasets = []
    lifecycle_data = None
    for fpath in args.files:
        data = json.loads(Path(fpath).read_text())
        if data.get("metadata", {}).get("type") == "lifecycle":
            lifecycle_data = data
        elif "results" in data:
            datasets.append(data)

    # Load explicit lifecycle file if given
    if args.lifecycle:
        lifecycle_data = json.loads(Path(args.lifecycle).read_text())

    if not datasets:
        print("No model benchmark files found.")
        return

    print(f"Loaded {len(datasets)} model result(s)")
    for d in datasets:
        print(f"  - {d['metadata']['short_name']}")

    # Generate charts
    generate_radar_chart(datasets, output_dir / "radar.png")
    generate_performance_chart(datasets, output_dir / "performance.png")
    generate_hallucination_chart(datasets, output_dir / "hallucination.png")

    if lifecycle_data:
        generate_lifecycle_chart(lifecycle_data, output_dir / "lifecycle.png")
    else:
        print("  No lifecycle data — skipping lifecycle chart.")

    print(f"\nAll charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
