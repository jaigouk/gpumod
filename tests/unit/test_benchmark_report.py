"""Unit tests for the unified benchmark report builders (gpumod-kpmq.5).

Tests the pure markdown-table functions over the combined-result schema. Chart PNGs
and the live runner are operational, not unit-tested here.
"""

from __future__ import annotations

from gpumod.benchmarks.report import (
    agentworld_dimension_table,
    agentworld_domain_table,
    build_report,
    coding_level_table,
    models_tested_table,
    setup_table,
    tldr_table,
)

_R1 = {
    "model_name": "gemma4-26b-a4b-qat",
    "gguf_gb": 14.25,
    "verdict": "Quality winner",
    "meta": {
        "repo": "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
        "architecture": "MoE 26B/A4B",
        "quant": "QAT UD-Q4_K_XL",
        "sampler": "GEMMA_CODING",
        "port": 7110,
    },
    "coding": {
        "mean": 100.0,
        "std": 0.0,
        "min": 100,
        "max": 100,
        "ci": [100.0, 100.0],
        "tps": 169.8,
        "vram_idle_mb": 17567,
        "iterations": 15,
        "per_level": {"L1": 15, "L2": 15, "L3": 15, "L4": 15, "L5": 15},
    },
    "agentworld": {
        "overall": 40.0,
        "n": 14,
        "n_loaded": 14,
        "judge_model": "claude-opus",
        "sample_per_domain": 2,
        "per_domain": {
            "mcp": 50.0,
            "search": 40.0,
            "terminal": 45.0,
            "swe": 30.0,
            "android": 35.0,
            "web": 40.0,
            "os": 40.0,
        },
        "dimensions": {
            "format": 3,
            "factuality": 2.5,
            "consistency": 3,
            "realism": 2,
            "quality": 2.5,
        },
    },
}
_R2 = {
    "model_name": "gemma4-e2b-qat-q2",
    "gguf_gb": 2.04,
    "coding": {
        "mean": 20.0,
        "std": 10.0,
        "min": 5,
        "max": 40,
        "ci": [12.0, 28.0],
        "tps": 250.0,
        "iterations": 15,
        "per_level": {"L1": 2, "L2": 1, "L3": 15, "L4": 15, "L5": 0},
    },
    "agentworld": {
        "overall": 2.1,
        "n": 12,
        "n_loaded": 14,
        "judge_model": "claude-opus",
        "sample_per_domain": 2,
        # 'web' deliberately missing -> table must render a placeholder, not crash
        "per_domain": {
            "mcp": 10.0,
            "search": 0.0,
            "terminal": 2.5,
            "swe": 0.0,
            "android": 0.0,
            "os": 0.0,
        },
        "dimensions": {"format": 1, "factuality": 1, "consistency": 1, "realism": 1, "quality": 1},
    },
}


def test_tldr_table_rich_columns_and_values() -> None:
    md = tldr_table([_R1, _R2])
    header = md.splitlines()[0]
    for col in ("Min/Max", "95% CI", "TPS", "AgentWorld", "VRAM idle", "GGUF", "Verdict"):
        assert col in header
    assert "169.8" in md  # coding TPS
    assert "100.0" in md  # coding mean (R1)
    assert "2.1" in md  # agentworld overall (R2)
    assert "Quality winner" in md  # verdict (R1)
    assert "14.2 GB" in md  # gguf size (R1)
    assert "17567 MB" in md  # vram idle (R1)


def test_models_tested_table() -> None:
    md = models_tested_table([_R1])
    assert "unsloth/gemma-4-26B-A4B-it-qat-GGUF" in md
    assert "MoE 26B/A4B" in md
    assert "QAT UD-Q4_K_XL" in md
    assert "GEMMA_CODING" in md
    assert "7110" in md


def test_setup_table() -> None:
    md = setup_table({"GPU": "RTX 4090", "llama.cpp": "b9784 (8be759e6f)"})
    assert "| GPU | RTX 4090 |" in md
    assert "b9784" in md


def test_agentworld_domain_table_domains_and_coverage() -> None:
    md = agentworld_domain_table([_R1, _R2])
    header = md.splitlines()[0]
    for dom in ("MCP", "SEARCH", "TERMINAL", "SWE", "ANDROID", "WEB", "OS"):
        assert dom in header
    assert "Coverage" in header
    assert "50.0" in md  # R1 mcp
    assert "12/14" in md  # R2 coverage (2 dropped)
    assert "—" in md  # R2 missing 'web' -> placeholder


def test_agentworld_dimension_table_has_judge() -> None:
    md = agentworld_dimension_table([_R1])
    assert "claude-opus" in md  # judge moved here from TL;DR
    assert "Format" in md


def test_coding_level_table_shows_pass_counts() -> None:
    md = coding_level_table([_R1, _R2])
    assert "L1" in md
    assert "L5" in md
    assert "15/15" in md
    assert "2/15" in md
    assert "0/15" in md


def test_build_report_omits_coding_section_when_no_coding_data() -> None:
    aw_only = {"model_name": "e2b", "agentworld": _R2["agentworld"]}
    md = build_report([aw_only], title="AW only")
    assert "AgentWorldBench (world-modeling)" in md
    assert "Coding suite (L1-L5)" not in md


def test_build_report_includes_setup_and_models_and_both_sections() -> None:
    md = build_report([_R1], title="Both", setup={"GPU": "RTX 4090"})
    assert "## Setup" in md
    assert "## Models tested" in md
    assert "Coding suite (L1-L5)" in md
    assert "AgentWorldBench (world-modeling)" in md
