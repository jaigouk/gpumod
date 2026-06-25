"""Scoring + aggregation for AgentWorldBench (gpumod-kpmq.1).

Each predicted observation is judged on five dimensions (Format, Factuality,
Consistency, Realism, Quality) on a 1-5 scale (per arxiv 2606.24597). A prediction's
score is the mean of the five dimensions, normalized to 0-100 (min-max: 1 -> 0,
5 -> 100). Results aggregate per-domain and overall, matching how the paper reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DIMENSIONS = ("format", "factuality", "consistency", "realism", "quality")


@dataclass(frozen=True)
class DimensionScores:
    format: int
    factuality: int
    consistency: int
    realism: int
    quality: int

    def values(self) -> tuple[int, int, int, int, int]:
        return (self.format, self.factuality, self.consistency, self.realism, self.quality)

    def mean_raw(self) -> float:
        """Mean of the five 1-5 dimension scores."""
        return sum(self.values()) / len(DIMENSIONS)

    def score_0_100(self) -> float:
        """Normalize the 1-5 mean to 0-100 (1 -> 0, 5 -> 100)."""
        return (self.mean_raw() - 1.0) / 4.0 * 100.0


def _summary(
    scores_0_100: list[float], dim_means: dict[str, float] | None = None
) -> dict[str, Any]:
    n = len(scores_0_100)
    mean = sum(scores_0_100) / n if n else 0.0
    out: dict[str, Any] = {"mean": round(mean, 4), "n": n}
    if dim_means is not None:
        out["dimensions"] = {k: round(v, 4) for k, v in dim_means.items()}
    return out


def aggregate(scored: list[tuple[str, DimensionScores]]) -> dict[str, Any]:
    """Aggregate (domain, DimensionScores) pairs into per-domain + overall summaries.

    Each summary has ``mean`` (0-100), ``n``, and per-dimension raw (1-5) means.
    """
    by_domain: dict[str, list[DimensionScores]] = {}
    for domain, ds in scored:
        by_domain.setdefault(domain, []).append(ds)

    def dim_means(items: list[DimensionScores]) -> dict[str, float]:
        if not items:
            return dict.fromkeys(DIMENSIONS, 0.0)
        return {dim: sum(getattr(i, dim) for i in items) / len(items) for dim in DIMENSIONS}

    per_domain = {
        domain: _summary([i.score_0_100() for i in items], dim_means(items))
        for domain, items in by_domain.items()
    }
    all_items = [ds for _, ds in scored]
    overall = _summary([i.score_0_100() for i in all_items], dim_means(all_items))
    return {"overall": overall, "per_domain": per_domain}
