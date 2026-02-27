"""Benchmark runner for Qwen3.5 coding tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .sampler_config import THINKING_CODING, SamplerConfig
from .scoring import (
    BenchmarkResult,
    LevelResult,
    calculate_confidence_interval,
    calculate_stats,
)

# Spike recommendations
DEFAULT_ITERATIONS = 15
DEFAULT_CONTEXT_SIZE = 32768


class LLMClient(Protocol):
    """Protocol for LLM client interface."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    model_id: str
    iterations: int = DEFAULT_ITERATIONS
    context_size: int = DEFAULT_CONTEXT_SIZE
    sampler: SamplerConfig = field(default_factory=lambda: THINKING_CODING)


@dataclass
class BenchmarkLevel:
    """Definition of a benchmark level."""

    level: int
    name: str
    points: int
    prompt: str
    validator: Any  # Callable to validate response


class BenchmarkRunner:
    """Runs coding benchmarks against Qwen3.5 models."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._client: LLMClient | None = None
        self._results: list[BenchmarkResult] = []
        self._levels = self._create_levels()

    @property
    def levels(self) -> list[BenchmarkLevel]:
        """Get benchmark levels."""
        return self._levels

    def _create_levels(self) -> list[BenchmarkLevel]:
        """Create benchmark level definitions."""
        # Basic levels for now - can be extended
        return [
            BenchmarkLevel(
                level=1,
                name="Basic Queue",
                points=25,
                prompt="Implement a basic job queue with add_job and get_result",
                validator=lambda x: "def add_job" in x,
            ),
            BenchmarkLevel(
                level=2,
                name="Retry with Backoff",
                points=25,
                prompt="Add retry logic with exponential backoff",
                validator=lambda x: "retry" in x.lower(),
            ),
            BenchmarkLevel(
                level=3,
                name="Priority Queue",
                points=25,
                prompt="Implement priority-based job scheduling",
                validator=lambda x: "priority" in x.lower(),
            ),
            BenchmarkLevel(
                level=4,
                name="Concurrency Bug Fix",
                points=15,
                prompt="Find and fix the race condition",
                validator=lambda x: "lock" in x.lower() or "Lock" in x,
            ),
            BenchmarkLevel(
                level=5,
                name="Multi-file Refactor",
                points=10,
                prompt="Split into multiple files",
                validator=lambda x: "__init__" in x,
            ),
        ]

    def set_client(self, client: LLMClient) -> None:
        """Set the LLM client for generation."""
        self._client = client

    async def run(self) -> list[BenchmarkResult]:
        """Run all benchmark iterations.

        Returns:
            List of BenchmarkResult for each iteration
        """
        if self._client is None:
            msg = "LLM client not set. Call set_client() first."
            raise RuntimeError(msg)

        self._results = []

        for iteration in range(self.config.iterations):
            level_results = await self._run_iteration(iteration + 1)
            result = BenchmarkResult(iteration=iteration + 1, levels=level_results)
            self._results.append(result)

        return self._results

    async def _run_iteration(self, iteration: int) -> list[LevelResult]:
        """Run a single benchmark iteration."""
        assert self._client is not None

        level_results = []
        for level in self._levels:
            response = await self._client.generate(
                level.prompt,
                **self.config.sampler.to_dict(),
            )

            passed = level.validator(response)
            points = level.points if passed else 0

            level_results.append(
                LevelResult(
                    level=level.level,
                    name=level.name,
                    points=points,
                    passed=passed,
                )
            )

        return level_results

    def generate_report(self) -> dict[str, Any]:
        """Generate benchmark report with statistics.

        Returns:
            Dict with model_id, stats, confidence_interval, and results
        """
        if not self._results:
            msg = "No results. Call run() first."
            raise RuntimeError(msg)

        scores = [r.total_score for r in self._results]
        stats = calculate_stats(scores)
        ci = calculate_confidence_interval(scores)

        return {
            "model_id": self.config.model_id,
            "iterations": self.config.iterations,
            "context_size": self.config.context_size,
            "sampler": self.config.sampler.to_dict(),
            "stats": stats,
            "confidence_interval": ci,
            "results": [
                {
                    "iteration": r.iteration,
                    "total_score": r.total_score,
                    "levels_passed": r.levels_passed,
                }
                for r in self._results
            ],
        }
