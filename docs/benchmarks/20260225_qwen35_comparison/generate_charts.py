#!/usr/bin/env python3
"""Generate comparison charts for Qwen3.5 benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Load results
results_dir = Path(__file__).parent
results = {}
for f in results_dir.glob("result_*.json"):
    data = json.loads(f.read_text())
    results[data["model_id"]] = data

# Model order and labels
models = ["qwen35-27b-multi", "qwen35-35b-q3-multi", "qwen35-35b-multi"]
labels = ["27B Q4", "35B Q3", "35B Q4"]
colors = ["#4CAF50", "#2196F3", "#FF9800"]

# Create charts directory
charts_dir = results_dir / "charts"
charts_dir.mkdir(exist_ok=True)

# Chart 1: Total Time Comparison (Bar)
fig, ax = plt.subplots(figsize=(10, 6))
times = [results[m]["total_time_ms"] / 1000 for m in models]
bars = ax.bar(labels, times, color=colors, edgecolor="black", linewidth=1.2)
ax.set_ylabel("Total Time (seconds)", fontsize=12)
ax.set_title("Qwen3.5 Multi-Agent Tetris: Total Time", fontsize=14, fontweight="bold")
ax.set_ylim(0, max(times) * 1.15)

# Add value labels
for bar, time in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{time:.1f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")

# Add speedup annotations
fastest = min(times)
for i, time in enumerate(times):
    if time != fastest:
        speedup = time / fastest
        ax.text(i, times[i] / 2, f"{speedup:.1f}x slower",
                ha="center", va="center", fontsize=10, color="white", fontweight="bold")

plt.tight_layout()
plt.savefig(charts_dir / "total_time.png", dpi=150)
plt.close()

# Chart 2: Phase Breakdown (Stacked Bar)
fig, ax = plt.subplots(figsize=(10, 6))

plan_times = [results[m]["phases"][0]["total_time_ms"] / 1000 for m in models]
dev_times = [results[m]["phases"][1]["total_time_ms"] / 1000 for m in models]
qa_times = [results[m]["phases"][2]["total_time_ms"] / 1000 for m in models]

x = np.arange(len(labels))
width = 0.6

p1 = ax.bar(x, plan_times, width, label="Planning", color="#66BB6A")
p2 = ax.bar(x, dev_times, width, bottom=plan_times, label="Development", color="#42A5F5")
p3 = ax.bar(x, qa_times, width, bottom=[p + d for p, d in zip(plan_times, dev_times)], label="QA Review", color="#FFA726")

ax.set_ylabel("Time (seconds)", fontsize=12)
ax.set_title("Qwen3.5 Multi-Agent Tetris: Phase Breakdown", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc="upper right")
ax.set_ylim(0, max([p + d + q for p, d, q in zip(plan_times, dev_times, qa_times)]) * 1.15)

plt.tight_layout()
plt.savefig(charts_dir / "phase_breakdown.png", dpi=150)
plt.close()

# Chart 3: VRAM vs Speed (Scatter)
fig, ax = plt.subplots(figsize=(10, 6))

vrams = [results[m]["vram_gb"] for m in models]
times = [results[m]["total_time_ms"] / 1000 for m in models]

for i, (vram, time, label) in enumerate(zip(vrams, times, labels)):
    ax.scatter(vram, time, s=200, c=colors[i], edgecolors="black", linewidth=1.5, zorder=3)
    ax.annotate(label, (vram, time), textcoords="offset points",
                xytext=(10, 10), fontsize=11, fontweight="bold")

ax.set_xlabel("VRAM Usage (GB)", fontsize=12)
ax.set_ylabel("Total Time (seconds)", fontsize=12)
ax.set_title("Qwen3.5: VRAM Efficiency", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xlim(14, 22)
ax.set_ylim(0, max(times) * 1.2)

# Highlight best (lower-left is better)
ax.annotate("Best: Low VRAM, Fast", xy=(16, 34.8), xytext=(17, 50),
            arrowprops=dict(arrowstyle="->", color="green", lw=2),
            fontsize=10, color="green", fontweight="bold")

plt.tight_layout()
plt.savefig(charts_dir / "vram_efficiency.png", dpi=150)
plt.close()

# Chart 4: Code Output Comparison (Bar)
fig, ax = plt.subplots(figsize=(10, 6))

lines = [results[m]["code_lines"] for m in models]
chars = [results[m]["phases"][1]["output_chars"] for m in models]

x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, lines, width, label="Code Lines", color="#81C784")
bars2 = ax.bar(x + width/2, [c/100 for c in chars], width, label="Output (×100 chars)", color="#64B5F6")

ax.set_ylabel("Count", fontsize=12)
ax.set_title("Qwen3.5: Code Output Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add value labels
for bar, val in zip(bars1, lines):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{val}", ha="center", va="bottom", fontsize=10)
for bar, val in zip(bars2, chars):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{val}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(charts_dir / "code_output.png", dpi=150)
plt.close()

print("Charts generated:")
for f in charts_dir.glob("*.png"):
    print(f"  {f.name}")
