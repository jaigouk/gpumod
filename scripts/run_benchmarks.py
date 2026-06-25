#!/usr/bin/env python
"""Unified benchmark report: coding (L1-L5) + AgentWorldBench, with charts (gpumod-kpmq.5).

Merges per-suite result JSONs (coding from run_qwen36_benchmark.py, world-modeling from
run_agentworld_benchmark.py) into one combined per-model view, then generates a rich
markdown report (src/gpumod/benchmarks/report.py) matching the house format (TL;DR with
full coding stats, Setup, Models-tested, per-domain/per-level/dimension tables) plus
matplotlib charts (PNGs in charts/).

Example (report from existing results):

    uv run python scripts/run_benchmarks.py \
        --coding docs/benchmarks/.../result_*.json \
        --agentworld docs/benchmarks/.../result_agentworld_*.json \
        --output-dir docs/benchmarks/20260625_unified/ --title "Coding + AgentWorld"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gpumod.benchmarks.agentworld.dataset import DOMAINS
from gpumod.benchmarks.agentworld.scorer import DIMENSIONS
from gpumod.benchmarks.report import build_report

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4", "#795548"]
CODING_LEVELS = ("L1", "L2", "L3", "L4", "L5")


# --------------------------------------------------------------------------- #
# Load per-suite result JSONs into the combined schema (keyed by model)
# --------------------------------------------------------------------------- #
def _gguf_gb(filename) -> float | None:
    if not filename:
        return None
    path = os.path.expanduser(f"~/bin/{filename}")
    try:
        return os.path.getsize(path) / 1073741824 if os.path.exists(path) else None
    except OSError:
        return None


def _sampler_name(sampler) -> str:
    if not isinstance(sampler, dict):
        return "—"
    tk, t = sampler.get("top_k"), sampler.get("temperature")
    if tk == 64:
        return "GEMMA_CODING"
    if tk == 20 and t == 0.6:
        return "THINKING_CODING"
    return f"temp={t}/top_k={tk}"


def load_coding(path: Path) -> tuple[str, dict, dict, float | None]:
    d = json.loads(path.read_text())
    model = d.get("model", {})
    key = model.get("id") or model.get("name") or path.stem
    iters = d.get("iterations", [])
    per_level: dict[str, int] = {}
    vrams = []
    for it in iters:
        for lv in it.get("levels", []):
            tag = f"L{lv['level']}"
            passed = lv.get("points_earned") == lv.get("points_possible")
            per_level[tag] = per_level.get(tag, 0) + (1 if passed else 0)
        v = (it.get("metrics") or {}).get("vram_idle_mb")
        if v:
            vrams.append(v)
    summary = d.get("summary", {})
    stats = summary.get("stats", {})
    ci = summary.get("confidence_interval", {})
    coding = {
        "mean": stats.get("mean"),
        "std": stats.get("std"),
        "min": stats.get("min"),
        "max": stats.get("max"),
        "ci": [ci.get("lower"), ci.get("upper")] if ci else None,
        "tps": summary.get("tps", {}).get("mean"),
        "vram_idle_mb": (sum(vrams) / len(vrams)) if vrams else None,
        "iterations": len(iters),
        "per_level": per_level,
    }
    meta = {
        "repo": model.get("repo"),
        "architecture": model.get("architecture"),
        "quant": model.get("quant"),
        "sampler": _sampler_name(model.get("sampler")),
        "port": model.get("port"),
    }
    return key, coding, meta, _gguf_gb(model.get("file"))


def load_agentworld(path: Path) -> tuple[str, dict, dict, float | None]:
    d = json.loads(path.read_text())
    key = d.get("model_name") or path.stem
    summary = d.get("summary", {})
    overall = summary.get("overall", {})
    per_domain = {dom: st.get("mean") for dom, st in summary.get("per_domain", {}).items()}
    aw = {
        "overall": overall.get("mean"),
        "n": d.get("n_scored", overall.get("n")),
        "n_loaded": d.get("n_loaded"),
        "judge_model": d.get("judge", {}).get("model", "?"),
        "sample_per_domain": d.get("sample_per_domain"),
        "per_domain": per_domain,
        "dimensions": overall.get("dimensions", {}),
    }
    return key, aw, {}, None


def merge(coding_paths: list[Path], agentworld_paths: list[Path]) -> list[dict]:
    combined: dict[str, dict] = {}
    for p in coding_paths:
        key, coding, meta, gguf = load_coding(p)
        r = combined.setdefault(key, {"model_name": key, "meta": {}})
        r["coding"] = coding
        r["meta"].update({k: v for k, v in meta.items() if v is not None})
        if gguf:
            r["gguf_gb"] = gguf
    for p in agentworld_paths:
        key, aw, meta, gguf = load_agentworld(p)
        r = combined.setdefault(key, {"model_name": key, "meta": {}})
        r["agentworld"] = aw
        r["meta"].update({k: v for k, v in meta.items() if v is not None})
        if gguf and "gguf_gb" not in r:
            r["gguf_gb"] = gguf
    return list(combined.values())


def detect_setup() -> dict[str, str]:
    setup: dict[str, str] = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.stdout.strip():
            setup["GPU"] = out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    setup["Host RAM"] = f"{int(line.split()[1]) / 1024 / 1024:.0f} GiB"
                    break
    except OSError:
        pass
    try:
        out = subprocess.run(  # noqa: S603
            [os.path.expanduser("~/bin/llama-server"), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        blob = (out.stderr + out.stdout).strip()
        for line in blob.splitlines():
            if line.startswith("version:"):
                setup["llama.cpp"] = line.replace("version:", "").strip()
                break
    except (OSError, subprocess.SubprocessError):
        pass
    setup["Stability defaults"] = (
        "GGML_CUDA_NO_PINNED=1 (template default); preflight RAM/VRAM checks"
    )
    return setup


# --------------------------------------------------------------------------- #
# Charts (house style: COLORS, dpi 150, value labels)
# --------------------------------------------------------------------------- #
def _grouped_bar(ax, categories, series_by_model, model_names, ylabel, ylim=None):
    x = np.arange(len(categories))
    n = max(len(model_names), 1)
    width = 0.8 / n
    for i, (name, vals) in enumerate(zip(model_names, series_by_model, strict=True)):
        offset = x + i * width - 0.4 + width / 2
        ax.bar(offset, vals, width, label=name, color=COLORS[i % len(COLORS)])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9)


def _save(fig, path: Path, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {msg}: {path}")


def chart_overall(results, path):
    names = [r["model_name"] for r in results]
    coding = [(r.get("coding") or {}).get("mean") or 0.0 for r in results]
    aw = [(r.get("agentworld") or {}).get("overall") or 0.0 for r in results]
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 5))
    _grouped_bar(
        ax, names, [coding, aw], ["Coding (L1-L5)", "AgentWorldBench"], "Score (0-100)", (0, 100)
    )
    ax.set_title("Overall: coding vs world-modeling", fontweight="bold")
    _save(fig, path, "overall")


def chart_agentworld_domains(results, path):
    names = [r["model_name"] for r in results]
    series = [
        [((r.get("agentworld") or {}).get("per_domain") or {}).get(d) or 0.0 for d in DOMAINS]
        for r in results
    ]
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(DOMAINS)), 5))
    _grouped_bar(ax, [d.upper() for d in DOMAINS], series, names, "Score (0-100)", (0, 100))
    ax.set_title("AgentWorldBench — per-domain (5-dim rubric mean)", fontweight="bold")
    _save(fig, path, "agentworld_domains")


def chart_coding_levels(results, path):
    names = [r["model_name"] for r in results]
    series = []
    for r in results:
        c = r.get("coding") or {}
        pl = c.get("per_level") or {}
        it = c.get("iterations") or 1
        series.append([100.0 * pl.get(lvl, 0) / it for lvl in CODING_LEVELS])
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bar(ax, list(CODING_LEVELS), series, names, "Pass rate (%)", (0, 100))
    ax.set_title("Coding suite — per-level pass rate", fontweight="bold")
    _save(fig, path, "coding_levels")


def chart_agentworld_dimensions(results, path):
    names = [r["model_name"] for r in results]
    series = [
        [((r.get("agentworld") or {}).get("dimensions") or {}).get(d) or 0.0 for d in DIMENSIONS]
        for r in results
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bar(ax, [d.capitalize() for d in DIMENSIONS], series, names, "Mean (1-5)", (0, 5))
    ax.set_title("AgentWorldBench — judge dimensions", fontweight="bold")
    _save(fig, path, "agentworld_dimensions")


def chart_tps(results, path):
    names = [r["model_name"] for r in results]
    tps = [(r.get("coding") or {}).get("tps") or 0.0 for r in results]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(names)), 5))
    bars = ax.bar(
        np.arange(len(names)),
        tps,
        color=[COLORS[i % len(COLORS)] for i in range(len(names))],
        width=0.6,
    )
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Gen tok/s")
    ax.set_title("Generation throughput", fontweight="bold")
    for b, v in zip(bars, tps, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    _save(fig, path, "tps")


def generate_charts(results, charts_dir: Path) -> list[tuple[str, str]]:
    has_coding = any(r.get("coding") for r in results)
    has_agentworld = any(r.get("agentworld") for r in results)
    specs = []
    if has_coding and has_agentworld:
        specs.append(("Overall: coding vs world-modeling", "overall.png", chart_overall))
    if has_agentworld:
        specs.append(
            (
                "AgentWorldBench per-domain (0-100)",
                "agentworld_domains.png",
                chart_agentworld_domains,
            )
        )
        specs.append(
            (
                "AgentWorldBench judge dimensions (1-5)",
                "agentworld_dimensions.png",
                chart_agentworld_dimensions,
            )
        )
    if has_coding:
        specs.append(
            ("Coding suite per-level pass rate", "coding_levels.png", chart_coding_levels)
        )
        specs.append(("Generation throughput (TPS)", "tps.png", chart_tps))
    refs = []
    for caption, fname, fn in specs:
        fn(results, charts_dir / fname)
        refs.append((caption, f"charts/{fname}"))
    return refs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified coding + AgentWorldBench report with charts")
    p.add_argument("--coding", nargs="*", default=[], type=Path, help="coding result JSON(s)")
    p.add_argument(
        "--agentworld", nargs="*", default=[], type=Path, help="agentworld result JSON(s)"
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--title", default="Unified benchmark — coding + AgentWorldBench")
    p.add_argument("--caveat", nargs="*", default=[], help="extra methodology caveats")
    p.add_argument("--no-setup", action="store_true", help="skip Setup auto-detection")
    return p.parse_args()


def _coverage_caveats(results) -> list[str]:
    out: list[str] = []
    for r in results:
        a = r.get("agentworld") or {}
        loaded, scored = a.get("n_loaded"), a.get("n")
        if loaded and scored is not None and scored < loaded:
            out.append(
                f"**{r['model_name']}** AgentWorldBench coverage {scored}/{loaded} — "
                f"{loaded - scored} sample(s) dropped (oversized prompt exceeded context, "
                f"or a gen/judge error). Comparisons are over scored samples only."
            )
    return out


def main() -> int:
    args = parse_args()
    results = merge(args.coding, args.agentworld)
    if not results:
        print("No result files given (--coding / --agentworld).")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models: {[r['model_name'] for r in results]}")
    chart_refs = generate_charts(results, args.output_dir / "charts")
    caveats = _coverage_caveats(results) + list(args.caveat or [])
    md = build_report(
        results,
        title=args.title,
        setup=None if args.no_setup else detect_setup(),
        chart_files=chart_refs,
        caveats=caveats or None,
    )
    (args.output_dir / "README.md").write_text(md)
    print(f"\nReport: {args.output_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
