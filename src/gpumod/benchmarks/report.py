"""Unified benchmark report builder (gpumod-kpmq.5).

Turns a list of combined per-model results into a rich markdown report covering BOTH
axes — the L1-L5 coding suite and AgentWorldBench world-modeling — matching the
hand-written house format (TL;DR with full coding stats, Setup, Models-tested,
per-domain / per-level / dimension tables) with embedded chart images (PNGs in
``charts/``).

Combined-result schema (one dict per model)::

    {
      "model_name": str,
      "meta": {"repo", "architecture", "quant", "gguf", "sampler", "port"},
      "gguf_gb": float | None,
      "verdict": str | None,
      "coding": {"mean", "std", "min", "max", "ci": [lo, hi], "tps",
                 "vram_idle_mb", "iterations", "per_level": {"L1".."L5": pass_count}},
      "agentworld": {"overall", "n", "n_loaded", "judge_model", "sample_per_domain",
                     "per_domain": {<domain>: score_0_100},
                     "dimensions": {<dim>: mean_1_5}, "vram_idle_mb"},
    }
"""

from __future__ import annotations

import math
from typing import Any

from gpumod.benchmarks.agentworld.dataset import DOMAINS
from gpumod.benchmarks.agentworld.scorer import DIMENSIONS

CODING_LEVELS = ("L1", "L2", "L3", "L4", "L5")


def _f(value: Any, nd: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float):
        return f"{value:.{nd}f}"
    return str(value)


def _minmax(coding: dict[str, Any]) -> str:
    lo, hi = coding.get("min"), coding.get("max")
    if lo is None and hi is None:
        return "—"
    return f"{_f(lo, 0)}/{_f(hi, 0)}"


def _ci(coding: dict[str, Any]) -> str:
    ci = coding.get("ci")
    if not ci or len(ci) != 2:
        return "—"
    return f"[{_f(ci[0])}, {_f(ci[1])}]"


def _gguf(r: dict[str, Any]) -> str:
    gb = r.get("gguf_gb")
    return f"{gb:.1f} GB" if isinstance(gb, int | float) else "—"


def _vram(r: dict[str, Any]) -> str:
    for axis in ("coding", "agentworld"):
        v = (r.get(axis) or {}).get("vram_idle_mb")
        if v:
            return f"{int(v)} MB"
    return "—"


def tldr_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Coding | σ | Min/Max | 95% CI | TPS | AgentWorld | VRAM idle | GGUF | Verdict |",  # noqa: RUF001, E501
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        c = r.get("coding", {}) or {}
        a = r.get("agentworld", {}) or {}
        lines.append(
            f"| {r['model_name']} | {_f(c.get('mean'))} | {_f(c.get('std'), 2)} | "
            f"{_minmax(c)} | {_ci(c)} | {_f(c.get('tps'))} | {_f(a.get('overall'))} | "
            f"{_vram(r)} | {_gguf(r)} | {r.get('verdict') or '—'} |"
        )
    return "\n".join(lines)


def models_tested_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Source | Architecture | Quant | GGUF | Sampler | Port |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for r in results:
        m = r.get("meta", {}) or {}
        lines.append(
            f"| `{r['model_name']}` | `{m.get('repo', '—')}` | {m.get('architecture', '—')} | "
            f"{m.get('quant', '—')} | {_gguf(r)} | {m.get('sampler', '—')} | "
            f"{_f(m.get('port'), 0)} |"
        )
    return "\n".join(lines)


def setup_table(setup: dict[str, str]) -> str:
    lines = ["| Component | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in setup.items()]
    return "\n".join(lines)


def agentworld_domain_table(results: list[dict[str, Any]]) -> str:
    header = "| Model | " + " | ".join(d.upper() for d in DOMAINS) + " | Overall | Coverage |"
    sep = "|---|" + "---:|" * (len(DOMAINS) + 2)
    lines = [header, sep]
    for r in results:
        a = r.get("agentworld", {}) or {}
        per_domain = a.get("per_domain", {}) or {}
        cells = " | ".join(_f(per_domain.get(d)) for d in DOMAINS)
        nscored, nloaded = a.get("n"), a.get("n_loaded")
        coverage = f"{nscored}/{nloaded}" if nscored is not None and nloaded else _f(nscored, 0)
        lines.append(f"| {r['model_name']} | {cells} | {_f(a.get('overall'))} | {coverage} |")
    return "\n".join(lines)


def agentworld_dimension_table(results: list[dict[str, Any]]) -> str:
    header = "| Model | " + " | ".join(d.capitalize() for d in DIMENSIONS) + " | Judge |"
    sep = "|---|" + "---:|" * len(DIMENSIONS) + "---|"
    lines = [header, sep]
    for r in results:
        a = r.get("agentworld", {}) or {}
        dims = a.get("dimensions", {}) or {}
        cells = " | ".join(_f(dims.get(d), 2) for d in DIMENSIONS)
        lines.append(f"| {r['model_name']} | {cells} | {a.get('judge_model', '—')} |")
    return "\n".join(lines)


def coding_level_table(results: list[dict[str, Any]]) -> str:
    header = "| Model | " + " | ".join(CODING_LEVELS) + " | Score |"
    sep = "|---|" + "---:|" * (len(CODING_LEVELS) + 1)
    lines = [header, sep]
    for r in results:
        c = r.get("coding", {}) or {}
        per_level = c.get("per_level", {}) or {}
        iters = c.get("iterations", "?")
        cells = " | ".join(f"{per_level.get(lvl, 0)}/{iters}" for lvl in CODING_LEVELS)
        lines.append(f"| {r['model_name']} | {cells} | {_f(c.get('mean'))} |")
    return "\n".join(lines)


def build_report(
    results: list[dict[str, Any]],
    *,
    title: str,
    setup: dict[str, str] | None = None,
    chart_files: list[tuple[str, str]] | None = None,
    caveats: list[str] | None = None,
    recommendation: str | None = None,
) -> str:
    """Assemble the full unified markdown report (rich, house format)."""
    has_coding = any(r.get("coding") for r in results)
    has_agentworld = any(r.get("agentworld") for r in results)

    out: list[str] = [f"# {title}", "", "## TL;DR", "", tldr_table(results), ""]

    if chart_files:
        out += ["## Charts", ""]
        for caption, path in chart_files:
            out += [f"![{caption}]({path})", "", f"*{caption}*", ""]

    if setup:
        out += ["## Setup", "", setup_table(setup), ""]

    out += ["## Models tested", "", models_tested_table(results), ""]

    if has_coding:
        out += [
            "## Coding suite (L1-L5)",
            "",
            "Per-level pass counts (of N iterations) and mean score.",
            "",
            coding_level_table(results),
            "",
        ]

    if has_agentworld:
        out += [
            "## AgentWorldBench (world-modeling)",
            "",
            "Five-dimensional reference-grounded judge, 0-100 per domain.",
            "",
            agentworld_domain_table(results),
            "",
            "### Judge dimensions (mean 1-5)",
            "",
            agentworld_dimension_table(results),
            "",
        ]

    if caveats:
        out += ["## Methodology caveats", ""]
        out += [f"- {c}" for c in caveats]
        out += [""]

    if recommendation:
        out += ["## Recommendation", "", recommendation, ""]

    return "\n".join(out)
