"""Load Qwen/AgentWorldBench (gpumod-kpmq.1).

The dataset ships 7 per-domain JSONL files (``<domain>_test.jsonl``), downloaded +
cached via ``huggingface_hub``. ``take_per_domain`` limits N samples per domain for
fast development / cost control on the judge.
"""

from __future__ import annotations

import json
from typing import Any

from gpumod.benchmarks.agentworld.models import AgentWorldSample

REPO_ID = "Qwen/AgentWorldBench"
DOMAINS: tuple[str, ...] = ("mcp", "search", "terminal", "swe", "android", "web", "os")


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def take_per_domain(rows: list[dict[str, Any]], n: int | None) -> list[dict[str, Any]]:
    """Keep at most ``n`` rows per domain (by the ``task`` field). ``None`` keeps all."""
    if n is None:
        return list(rows)
    counts: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        domain = str(row.get("task", ""))
        if counts.get(domain, 0) < n:
            out.append(row)
            counts[domain] = counts.get(domain, 0) + 1
    return out


def load_rows(domains: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    rows: list[dict[str, Any]] = []
    for domain in domains or DOMAINS:
        path = hf_hub_download(REPO_ID, f"{domain}_test.jsonl", repo_type="dataset")
        rows.extend(_read_jsonl(path))
    return rows


def load_samples(
    domains: tuple[str, ...] | None = None,
    sample_per_domain: int | None = None,
) -> list[AgentWorldSample]:
    rows = take_per_domain(load_rows(domains), sample_per_domain)
    return [AgentWorldSample.from_row(row) for row in rows]
