"""Scoring calculations for Qwen3.5 benchmark."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class LevelResult:
    """Result for a single benchmark level."""

    level: int
    name: str
    points: int
    passed: bool


@dataclass
class BenchmarkResult:
    """Result for a single benchmark iteration."""

    iteration: int
    levels: list[LevelResult]

    @property
    def total_score(self) -> int:
        """Total points across all levels."""
        return sum(level.points for level in self.levels)

    @property
    def levels_passed(self) -> int:
        """Number of levels passed."""
        return sum(1 for level in self.levels if level.passed)


def calculate_stats(scores: Sequence[int | float]) -> dict[str, float]:
    """Calculate descriptive statistics for scores.

    Args:
        scores: List of benchmark scores

    Returns:
        Dict with mean, std, min, max

    Raises:
        ValueError: If scores is empty
    """
    if not scores:
        msg = "No scores provided"
        raise ValueError(msg)

    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0

    return {
        "mean": mean,
        "std": std,
        "min": min(scores),
        "max": max(scores),
    }


def calculate_confidence_interval(
    scores: Sequence[int | float],
    confidence: float = 0.95,
) -> dict[str, float]:
    """Calculate confidence interval for scores.

    Uses t-distribution for small samples.

    Args:
        scores: List of benchmark scores
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Dict with lower and upper bounds
    """
    if len(scores) <= 1:
        val = float(scores[0]) if scores else 0.0
        return {"lower": val, "upper": val}

    mean = statistics.mean(scores)
    std = statistics.stdev(scores)

    if std == 0:
        return {"lower": mean, "upper": mean}

    n = len(scores)
    # t-value for 95% CI approximation
    # For n >= 30, use 1.96; for smaller n, use larger values
    t_values: dict[int, float] = {
        2: 12.71,
        3: 4.30,
        4: 3.18,
        5: 2.78,
        6: 2.57,
        7: 2.45,
        8: 2.36,
        9: 2.31,
        10: 2.26,
        15: 2.14,
        20: 2.09,
        30: 2.04,
    }

    # Find closest t-value
    t = 1.96  # default for large n
    for threshold, t_val in sorted(t_values.items()):
        if n <= threshold:
            t = t_val
            break

    margin = t * (std / (n**0.5))
    return {
        "lower": mean - margin,
        "upper": mean + margin,
    }
