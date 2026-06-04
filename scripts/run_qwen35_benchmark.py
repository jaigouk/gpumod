#!/usr/bin/env python3
"""Qwen3.5 provider comparison benchmark runner.

Runs coding benchmarks against multiple providers and stores all artifacts.

Usage:
    python scripts/run_qwen35_benchmark.py --provider aessedai
    python scripts/run_qwen35_benchmark.py --provider all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpumod.benchmarks.coding.levels import (
    LEVEL_REGISTRY,
    PytestValidator,
    get_level,
)
from gpumod.benchmarks.coding.llm_client import LlamaCppClient
from gpumod.benchmarks.coding.metrics_collector import DefaultMetricsCollector
from gpumod.benchmarks.coding.sampler_config import THINKING_CODING
from gpumod.benchmarks.coding.scoring import calculate_confidence_interval, calculate_stats

# ---------------------------------------------------------------------------
# Provider Configuration
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""

    id: str
    name: str
    repo: str
    quant: str
    file: str
    port: int  # llama.cpp server port
    service_id: str  # gpumod service ID


PROVIDERS: dict[str, ProviderConfig] = {
    "aessedai": ProviderConfig(
        id="aessedai",
        name="AesSedai",
        repo="AesSedai/Qwen3.5-35B-A3B-GGUF",
        quant="IQ4_XS",
        file="Qwen3.5-35B-A3B-IQ4_XS.gguf",
        port=7094,
        service_id="qwen35-35b-a3b-aessedai-iq4xs",
    ),
    "bartowski": ProviderConfig(
        id="bartowski",
        name="bartowski",
        repo="bartowski/Qwen_Qwen3.5-35B-A3B-GGUF",
        quant="IQ4_XS",
        file="Qwen_Qwen3.5-35B-A3B-IQ4_XS.gguf",
        port=7095,
        service_id="qwen35-35b-a3b-bartowski-iq4xs",
    ),
    "unsloth": ProviderConfig(
        id="unsloth",
        name="unsloth",
        repo="unsloth/Qwen3.5-35B-A3B-GGUF",
        quant="MXFP4",
        file="Qwen3.5-35B-A3B-UD-MXFP4_MOE.gguf",
        port=7096,
        service_id="qwen35-35b-a3b-unsloth-mxfp4",
    ),
}


# ---------------------------------------------------------------------------
# Result Data Classes
# ---------------------------------------------------------------------------


@dataclass
class LevelArtifact:
    """Artifact for a single level attempt."""

    level: int
    name: str
    points_possible: int
    prompt: str
    response: str
    validation: dict[str, Any]
    points_earned: int
    duration_seconds: float
    timestamp: str


@dataclass
class IterationResult:
    """Result for a single benchmark iteration."""

    iteration: int
    levels: list[LevelArtifact]
    total_score: int
    levels_passed: int
    metrics: dict[str, Any]
    timestamp: str


@dataclass
class BenchmarkRun:
    """Complete benchmark run for a provider."""

    provider: ProviderConfig
    config: dict[str, Any]
    iterations: list[IterationResult]
    summary: dict[str, Any]
    started_at: str
    completed_at: str


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


class Qwen35Benchmark:
    """Runs Qwen3.5 coding benchmarks with full artifact storage."""

    def __init__(
        self,
        provider: ProviderConfig,
        base_url: str = "http://localhost:8080",
        iterations: int = 15,
        output_dir: Path | None = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url
        self.iterations = iterations
        self.output_dir = output_dir or Path(
            "docs/benchmarks/20260226_qwen35_35b_a3b_provider_comparison"
        )
        self.validator = PytestValidator(timeout_seconds=30)
        self.client: LlamaCppClient | None = None
        self.metrics_collector = DefaultMetricsCollector()

    async def run(self) -> BenchmarkRun:
        """Run the complete benchmark."""
        started_at = datetime.now(UTC).isoformat()

        # Initialize client
        self.client = LlamaCppClient(base_url=self.base_url, timeout=180.0)

        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {self.provider.name} ({self.provider.quant})")
        print(f"Repository: {self.provider.repo}")
        print(f"Iterations: {self.iterations}")
        print(f"{'=' * 60}\n")

        iteration_results: list[IterationResult] = []

        for i in range(self.iterations):
            print(f"\n--- Iteration {i + 1}/{self.iterations} ---")

            # Measure VRAM before iteration
            self.metrics_collector.measure_vram()

            result = await self._run_iteration(i + 1)
            iteration_results.append(result)

            print(f"  Score: {result.total_score}/100 ({result.levels_passed}/5 levels)")

        completed_at = datetime.now(UTC).isoformat()

        # Calculate summary statistics
        scores = [r.total_score for r in iteration_results]
        stats = calculate_stats(scores)
        ci = calculate_confidence_interval(scores)

        # Aggregate metrics
        all_tps = []
        all_vram_peak = []
        for r in iteration_results:
            if "mean_tps" in r.metrics:
                all_tps.append(r.metrics["mean_tps"])
            if "vram_peak_mb" in r.metrics:
                all_vram_peak.append(r.metrics["vram_peak_mb"])

        summary = {
            "scores": scores,
            "stats": stats,
            "confidence_interval": ci,
            "tps": {
                "mean": sum(all_tps) / len(all_tps) if all_tps else None,
                "values": all_tps,
            },
            "vram_peak_mb": max(all_vram_peak) if all_vram_peak else None,
        }

        run = BenchmarkRun(
            provider=self.provider,
            config={
                "base_url": self.base_url,
                "iterations": self.iterations,
                "sampler": THINKING_CODING.to_dict(),
                "levels": list(LEVEL_REGISTRY.keys()),
            },
            iterations=iteration_results,
            summary=summary,
            started_at=started_at,
            completed_at=completed_at,
        )

        # Save artifacts first (in case close fails)
        self._save_artifacts(run)

        # Close client
        await self.client.close()

        return run

    async def _run_iteration(self, iteration: int) -> IterationResult:
        """Run a single benchmark iteration."""
        assert self.client is not None

        level_artifacts: list[LevelArtifact] = []
        total_score = 0
        levels_passed = 0

        for level_num in sorted(LEVEL_REGISTRY.keys()):
            level_def = get_level(level_num)

            print(f"  L{level_num}: {level_def.name}...", end=" ", flush=True)

            start_time = time.perf_counter()
            timestamp = datetime.now(UTC).isoformat()

            # Generate response
            try:
                response = await self.client.generate(
                    level_def.prompt,
                    **THINKING_CODING.to_dict(),
                )
            except Exception as e:
                response = f"ERROR: {e}"

            end_time = time.perf_counter()
            duration = end_time - start_time

            # Record metrics
            if self.client.last_timing:
                tokens = self.client.last_timing.get("generated_tokens", 0)
                ttft = self.client.last_timing.get("prompt_ms", 0) / 1000
            else:
                tokens = len(response) // 4 if response else 0
                ttft = duration * 0.1

            self.metrics_collector.record_generation(
                tokens=tokens,
                duration_seconds=duration,
                ttft_seconds=ttft,
            )

            # Extract code from response (handle markdown code blocks)
            code = self._extract_code(response)

            # Validate
            validation_result = self.validator.validate(code, level_def.test_code)
            passed = validation_result.passed
            points = level_def.points if passed else 0

            if passed:
                total_score += points
                levels_passed += 1
                print(f"✓ (+{points})")
            else:
                print(f"✗ ({validation_result.error or 'failed'})")

            artifact = LevelArtifact(
                level=level_num,
                name=level_def.name,
                points_possible=level_def.points,
                prompt=level_def.prompt,
                response=response,
                validation={
                    "passed": validation_result.passed,
                    "pass_rate": validation_result.pass_rate,
                    "tests_passed": validation_result.tests_passed,
                    "tests_total": validation_result.tests_total,
                    "error": validation_result.error,
                },
                points_earned=points,
                duration_seconds=duration,
                timestamp=timestamp,
            )
            level_artifacts.append(artifact)

        # Get iteration metrics
        metrics = self.metrics_collector.get_iteration_metrics()

        return IterationResult(
            iteration=iteration,
            levels=level_artifacts,
            total_score=total_score,
            levels_passed=levels_passed,
            metrics=metrics,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def _extract_code(self, response: str) -> str:
        """Extract Python code from response, handling markdown blocks."""
        if not response:
            return ""

        # Check for markdown code blocks
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                return code_part.strip()

        if "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                code_part = parts[1]
                # Remove language identifier if present
                lines = code_part.strip().split("\n")
                if lines and not lines[0].startswith(("def ", "class ", "import ", "from ")):
                    lines = lines[1:]
                return "\n".join(lines).strip()

        # No code blocks, return as-is
        return response.strip()

    def _save_artifacts(self, run: BenchmarkRun) -> None:
        """Save all artifacts to disk."""
        provider_id = run.provider.id
        artifacts_dir = self.output_dir / "artifacts" / provider_id

        # Create directories
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save full JSON result
        json_path = self.output_dir / f"result_{provider_id}.json"
        with open(json_path, "w") as f:
            json.dump(self._serialize_run(run), f, indent=2)
        print(f"\nSaved: {json_path}")

        # Save generated code per iteration per level
        for iteration in run.iterations:
            iter_dir = artifacts_dir / f"iter_{iteration.iteration:02d}"
            iter_dir.mkdir(exist_ok=True)

            for level in iteration.levels:
                # Save response (full)
                response_path = iter_dir / f"L{level.level}_response.txt"
                with open(response_path, "w") as f:
                    f.write(level.response)

                # Save extracted code
                code = self._extract_code(level.response)
                code_path = iter_dir / f"L{level.level}_code.py"
                with open(code_path, "w") as f:
                    f.write(code)

            # Save iteration summary
            summary_path = iter_dir / "summary.json"
            with open(summary_path, "w") as f:
                json.dump(
                    {
                        "iteration": iteration.iteration,
                        "total_score": iteration.total_score,
                        "levels_passed": iteration.levels_passed,
                        "metrics": iteration.metrics,
                        "timestamp": iteration.timestamp,
                        "levels": [
                            {
                                "level": lv.level,
                                "name": lv.name,
                                "points_earned": lv.points_earned,
                                "validation": lv.validation,
                                "duration_seconds": lv.duration_seconds,
                            }
                            for lv in iteration.levels
                        ],
                    },
                    f,
                    indent=2,
                )

        print(f"Saved artifacts: {artifacts_dir}")

    def _serialize_run(self, run: BenchmarkRun) -> dict[str, Any]:
        """Serialize BenchmarkRun to JSON-compatible dict."""
        return {
            "provider": asdict(run.provider),
            "config": run.config,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "total_score": it.total_score,
                    "levels_passed": it.levels_passed,
                    "metrics": it.metrics,
                    "timestamp": it.timestamp,
                    "levels": [
                        {
                            "level": lv.level,
                            "name": lv.name,
                            "points_possible": lv.points_possible,
                            "points_earned": lv.points_earned,
                            "validation": lv.validation,
                            "duration_seconds": lv.duration_seconds,
                            "timestamp": lv.timestamp,
                            # Store response in artifacts, not in main JSON
                        }
                        for lv in it.levels
                    ],
                }
                for it in run.iterations
            ],
            "summary": run.summary,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Qwen3.5 provider comparison benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--provider",
        choices=["aessedai", "bartowski", "unsloth", "all"],
        required=True,
        help="Provider to benchmark (or 'all' for all providers)",
    )

    parser.add_argument(
        "--base-url",
        default=None,
        help="llama.cpp server URL (default: use provider's configured port)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Number of iterations per provider (default: 15)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/benchmarks/20260226_qwen35_35b_a3b_provider_comparison"),
        help="Output directory for results",
    )

    return parser.parse_args()


def _save_combined_results(all_runs: list[BenchmarkRun], output_dir: Path) -> None:
    """Save combined results to JSON file."""
    combined_path = output_dir / "results_combined.json"
    combined = {
        "providers": [asdict(r.provider) for r in all_runs],
        "comparison": [
            {
                "provider_id": r.provider.id,
                "stats": r.summary["stats"],
                "confidence_interval": r.summary["confidence_interval"],
                "tps": r.summary["tps"],
                "vram_peak_mb": r.summary["vram_peak_mb"],
            }
            for r in all_runs
        ],
        "timestamp": datetime.now(UTC).isoformat(),
    }
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved combined results: {combined_path}")


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Determine providers to run
    providers = list(PROVIDERS.values()) if args.provider == "all" else [PROVIDERS[args.provider]]

    all_runs: list[BenchmarkRun] = []

    for provider in providers:
        # Use provider's port if no explicit base-url given
        base_url = args.base_url or f"http://localhost:{provider.port}"

        benchmark = Qwen35Benchmark(
            provider=provider,
            base_url=base_url,
            iterations=args.iterations,
            output_dir=args.output_dir,
        )

        run = await benchmark.run()
        all_runs.append(run)

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"Summary: {provider.name}")
        print(f"{'=' * 60}")
        print(f"  Mean Score: {run.summary['stats']['mean']:.1f}")
        print(f"  Std Dev: {run.summary['stats']['std']:.1f}")
        print(f"  Min/Max: {run.summary['stats']['min']}/{run.summary['stats']['max']}")
        print(
            f"  95% CI: [{run.summary['confidence_interval']['lower']:.1f}, {run.summary['confidence_interval']['upper']:.1f}]"
        )
        if run.summary["tps"]["mean"]:
            print(f"  Mean TPS: {run.summary['tps']['mean']:.1f}")
        if run.summary["vram_peak_mb"]:
            print(f"  VRAM Peak: {run.summary['vram_peak_mb']} MB")

    # Save combined results
    if len(all_runs) > 1:
        _save_combined_results(all_runs, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
