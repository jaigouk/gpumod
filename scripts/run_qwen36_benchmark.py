#!/usr/bin/env python3
"""Qwen3.6 vs Gemma4 architecture comparison benchmark.

Compares coding performance across different model architectures:
- Qwen3.6-27B (dense, Q4_K_M)
- Qwen3.6-35B-A3B (MoE, UD-Q4_K_S)
- Gemma 4 E4B (dense, BF16)

Reuses v2 methodology from the Qwen3.5 provider comparison benchmark.

Usage:
    python scripts/run_qwen36_benchmark.py --model qwen36-27b
    python scripts/run_qwen36_benchmark.py --model gemma4-e4b
    python scripts/run_qwen36_benchmark.py --model all
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpumod.benchmarks.qwen35.levels import (
    LEVEL_REGISTRY,
    PytestValidator,
    get_level,
)
from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient
from gpumod.benchmarks.qwen35.metrics_collector import DefaultMetricsCollector
from gpumod.benchmarks.qwen35.sampler_config import THINKING_CODING
from gpumod.benchmarks.qwen35.scoring import calculate_confidence_interval, calculate_stats

DEFAULT_OUTPUT_DIR = Path("docs/benchmarks/20260423_qwen36_gemma4_comparison")


@dataclass
class ModelConfig:
    id: str
    name: str
    architecture: str
    repo: str
    quant: str
    file: str
    port: int
    service_id: str


MODELS: dict[str, ModelConfig] = {
    "qwen36-27b": ModelConfig(
        id="qwen36-27b",
        name="Qwen3.6-27B",
        architecture="dense-27B",
        repo="unsloth/Qwen3.6-27B-GGUF",
        quant="Q4_K_M",
        file="Qwen3.6-27B-Q4_K_M.gguf",
        port=7100,
        service_id="qwen36-27b-q4",
    ),
    "qwen36-35b-a3b": ModelConfig(
        id="qwen36-35b-a3b",
        name="Qwen3.6-35B-A3B",
        architecture="moe-35B-A3B",
        repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        quant="UD-Q4_K_S",
        file="Qwen3.6-35B-A3B-UD-Q4_K_S.gguf",
        port=7101,
        service_id="qwen36-35b-a3b-q4",
    ),
    "gemma4-e4b": ModelConfig(
        id="gemma4-e4b",
        name="Gemma 4 E4B",
        architecture="dense-E4B",
        repo="unsloth/gemma-4-E4B-it-GGUF",
        quant="BF16",
        file="gemma-4-E4B-it-BF16.gguf",
        port=7098,
        service_id="gemma4-e4b-bf16",
    ),
}


@dataclass
class LevelArtifact:
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
    iteration: int
    levels: list[LevelArtifact]
    total_score: int
    levels_passed: int
    metrics: dict[str, Any]
    timestamp: str


@dataclass
class BenchmarkRun:
    model: ModelConfig
    config: dict[str, Any]
    iterations: list[IterationResult]
    summary: dict[str, Any]
    started_at: str
    completed_at: str


class ArchitectureBenchmark:
    """Runs coding benchmarks comparing different model architectures."""

    def __init__(
        self,
        model: ModelConfig,
        base_url: str = "http://localhost:8080",
        iterations: int = 15,
        output_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.iterations = iterations
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self.validator = PytestValidator(timeout_seconds=30)
        self.client: LlamaCppClient | None = None
        self.client_timeout = 300.0
        self.metrics_collector = DefaultMetricsCollector()

    async def run(self) -> BenchmarkRun:
        started_at = datetime.now(UTC).isoformat()
        self.client = LlamaCppClient(base_url=self.base_url, timeout=self.client_timeout)

        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {self.model.name} ({self.model.quant})")
        print(f"Architecture: {self.model.architecture}")
        print(f"Repository: {self.model.repo}")
        print(f"Iterations: {self.iterations}")
        print(f"{'=' * 60}\n")

        iteration_results: list[IterationResult] = []

        for i in range(self.iterations):
            print(f"\n--- Iteration {i + 1}/{self.iterations} ---")
            self.metrics_collector.measure_vram()
            result = await self._run_iteration(i + 1)
            iteration_results.append(result)
            print(f"  Score: {result.total_score}/100 ({result.levels_passed}/5 levels)")

        completed_at = datetime.now(UTC).isoformat()

        scores = [r.total_score for r in iteration_results]
        stats = calculate_stats(scores)
        ci = calculate_confidence_interval(scores)

        all_tps: list[float] = []
        all_vram_peak: list[int] = []
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
            model=self.model,
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

        self._save_artifacts(run)
        await self.client.close()
        return run

    async def _run_iteration(self, iteration: int) -> IterationResult:
        assert self.client is not None

        level_artifacts: list[LevelArtifact] = []
        total_score = 0
        levels_passed = 0

        for level_num in sorted(LEVEL_REGISTRY.keys()):
            level_def = get_level(level_num)
            print(f"  L{level_num}: {level_def.name}...", end=" ", flush=True)

            start_time = time.perf_counter()
            timestamp = datetime.now(UTC).isoformat()

            try:
                response = await self.client.generate(
                    level_def.prompt,
                    **THINKING_CODING.to_dict(),
                )
            except Exception as e:
                response = f"ERROR: {e}"

            end_time = time.perf_counter()
            duration = end_time - start_time

            if self.client.last_timing:
                tokens = self.client.last_timing.get("generated_tokens", 0)
                gen_ms = self.client.last_timing.get("generation_ms", 0.0)
                ttft = self.client.last_timing.get("prompt_ms", 0) / 1000
                gen_duration = gen_ms / 1000 if gen_ms > 0 else duration
            else:
                tokens = len(response) // 4 if response else 0
                ttft = duration * 0.1
                gen_duration = duration

            self.metrics_collector.record_generation(
                tokens=tokens,
                duration_seconds=gen_duration,
                ttft_seconds=ttft,
            )

            code = self._extract_code(response)
            validation_result = self.validator.validate(code, level_def.test_code)
            passed = validation_result.passed
            points = level_def.points if passed else 0

            if passed:
                total_score += points
                levels_passed += 1
                print(f"PASS (+{points})")
            else:
                print(f"FAIL ({validation_result.error or 'failed'})")

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
        if not response:
            return ""

        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                return code_part.strip()

        if "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                code_part = parts[1]
                lines = code_part.strip().split("\n")
                if lines and not lines[0].startswith(("def ", "class ", "import ", "from ")):
                    lines = lines[1:]
                return "\n".join(lines).strip()

        return response.strip()

    def _save_artifacts(self, run: BenchmarkRun) -> None:
        model_id = run.model.id
        artifacts_dir = self.output_dir / "artifacts" / model_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.output_dir / f"result_{model_id}.json"
        with open(json_path, "w") as f:
            json.dump(self._serialize_run(run), f, indent=2)
        print(f"\nSaved: {json_path}")

        for iteration in run.iterations:
            iter_dir = artifacts_dir / f"iter_{iteration.iteration:02d}"
            iter_dir.mkdir(exist_ok=True)

            for level in iteration.levels:
                response_path = iter_dir / f"L{level.level}_response.txt"
                with open(response_path, "w") as f:
                    f.write(level.response)

                code = self._extract_code(level.response)
                code_path = iter_dir / f"L{level.level}_code.py"
                with open(code_path, "w") as f:
                    f.write(code)

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
        return {
            "model": asdict(run.model),
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


def _save_combined_results(all_runs: list[BenchmarkRun], output_dir: Path) -> None:
    combined_path = output_dir / "results_combined.json"
    combined = {
        "models": [asdict(r.model) for r in all_runs],
        "comparison": [
            {
                "model_id": r.model.id,
                "architecture": r.model.architecture,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3.6 vs Gemma4 architecture comparison benchmark",
    )
    parser.add_argument(
        "--model",
        choices=["qwen36-27b", "qwen36-35b-a3b", "gemma4-e4b", "all"],
        required=True,
        help="Model to benchmark (or 'all' for sequential run)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override llama.cpp server URL (default: use model's configured port)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Number of iterations per model (default: 15)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    models = list(MODELS.values()) if args.model == "all" else [MODELS[args.model]]
    all_runs: list[BenchmarkRun] = []

    for model in models:
        base_url = args.base_url or f"http://localhost:{model.port}"

        benchmark = ArchitectureBenchmark(
            model=model,
            base_url=base_url,
            iterations=args.iterations,
            output_dir=args.output_dir,
        )

        run = await benchmark.run()
        all_runs.append(run)

        print(f"\n{'=' * 60}")
        print(f"Summary: {model.name} ({model.architecture})")
        print(f"{'=' * 60}")
        print(f"  Mean Score: {run.summary['stats']['mean']:.1f}")
        print(f"  Std Dev: {run.summary['stats']['std']:.1f}")
        print(f"  Min/Max: {run.summary['stats']['min']}/{run.summary['stats']['max']}")
        ci = run.summary["confidence_interval"]
        print(f"  95% CI: [{ci['lower']:.1f}, {ci['upper']:.1f}]")
        if run.summary["tps"]["mean"]:
            print(f"  Mean TPS: {run.summary['tps']['mean']:.1f}")
        if run.summary["vram_peak_mb"]:
            print(f"  VRAM Peak: {run.summary['vram_peak_mb']} MB")

    if len(all_runs) > 1:
        _save_combined_results(all_runs, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
