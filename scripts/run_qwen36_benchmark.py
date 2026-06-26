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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpumod.benchmarks.coding.code_extraction import extract_code
from gpumod.benchmarks.coding.levels import (
    LEVEL_REGISTRY,
    PytestValidator,
    get_level,
)
from gpumod.benchmarks.coding.llm_client import LlamaCppClient
from gpumod.benchmarks.coding.metrics_collector import DefaultMetricsCollector
from gpumod.benchmarks.coding.scoring import calculate_confidence_interval, calculate_stats
from gpumod.benchmarks.model_registry import REGISTRY, ModelSpec
from gpumod.benchmarks.normalizers import CodeAnswerNormalizer, _strip_think_blocks

DEFAULT_OUTPUT_DIR = Path("docs/benchmarks/20260423_qwen36_gemma4_comparison")


def _model_identity(model: ModelSpec) -> dict[str, Any]:
    """Identity-only view for the result JSON ``"model"`` object.

    Keeps the schema identical to the pre-refactor ``asdict(ModelConfig)``
    output (9 keys, sampler as ``to_dict()``) — the ``normalizer`` and
    ``max_tokens`` fields are intentionally excluded (a normalizer is not
    JSON-serializable and was never part of the schema). Consumers like
    ``scripts/run_benchmarks.py`` merge coding results on ``model.id``.
    """
    return {
        "id": model.id,
        "name": model.name,
        "architecture": model.architecture,
        "repo": model.repo,
        "quant": model.quant,
        "file": model.file,
        "port": model.port,
        "service_id": model.service_id,
        "sampler": model.sampler.to_dict(),
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
    model: ModelSpec
    config: dict[str, Any]
    iterations: list[IterationResult]
    summary: dict[str, Any]
    started_at: str
    completed_at: str


class ArchitectureBenchmark:
    """Runs coding benchmarks comparing different model architectures."""

    def __init__(
        self,
        model: ModelSpec,
        base_url: str = "http://localhost:8080",
        iterations: int = 15,
        output_dir: Path | None = None,
        max_tokens: int = 32768,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.iterations = iterations
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self.validator = PytestValidator(timeout_seconds=30)
        self.client: LlamaCppClient | None = None
        # gpumod-76l.3: 900s (was 300s) to give MTP/thinking-mode runs
        # enough budget to finish on L4/L5. At ~10 t/s observed for MTP,
        # 300s only buys ~3000 tokens — easily truncated by long
        # <think> blocks, producing SyntaxError in extracted code.
        self.client_timeout = 900.0
        # gpumod-76l.3: Unsloth recommends 32768 max output tokens for
        # Qwen3.6 general queries. Without this, thinking-mode models can
        # consume the entire ctx-size (40960) and never emit code. Override via
        # --max-tokens for degenerate small/low-bit models that ramble (e.g. E2B Q2).
        self.max_tokens = max_tokens
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
                "sampler": self.model.sampler.to_dict(),
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
                    max_tokens=self.max_tokens,
                    **self.model.sampler.to_dict(),
                )
            except Exception as e:
                response = f"ERROR: {e}"
            # gpumod-76l.3: Prefer `content` (the model's final answer);
            # `reasoning_content` holds the chain-of-thought, which often
            # contains DRAFT code the model later refined or rejected.
            # Picking from reasoning first means we grab a draft and miss
            # the polished final code. Only fall back to reasoning when
            # content has no code fence at all (model burned its budget
            # thinking without reaching final code).
            reasoning = self.client.last_reasoning_content
            # gpumod-nor9: extraction goes through the model's ResponseNormalizer
            # (default CodeAnswerNormalizer when the spec leaves it None). The
            # normalizer strips closed <think> blocks first so the fence search
            # sees only the post-</think> final answer (gpumod-msy8); it falls
            # back to reasoning_content only when content has no fence. Raw
            # `response` is kept for the artifact below.
            normalizer = self.model.normalizer or CodeAnswerNormalizer()
            extract_source = normalizer.extract_answer(response, reasoning)
            # Artifact preserves the RAW response (incl. any <think> trace) plus
            # reasoning_content for post-hoc debugging, even though extraction
            # used the think-stripped final answer.
            artifact_response = (
                f"<reasoning_content>\n{reasoning}\n</reasoning_content>\n\n"
                f"<content>\n{response}\n</content>"
                if reasoning
                else response
            )

            end_time = time.perf_counter()
            duration = end_time - start_time

            draft_n: int | None = None
            draft_n_accepted: int | None = None
            if self.client.last_timing:
                tokens = self.client.last_timing.get("generated_tokens", 0)
                gen_ms = self.client.last_timing.get("generation_ms", 0.0)
                ttft = self.client.last_timing.get("prompt_ms", 0) / 1000
                gen_duration = gen_ms / 1000 if gen_ms > 0 else duration
                draft_n = self.client.last_timing.get("draft_n")
                draft_n_accepted = self.client.last_timing.get("draft_n_accepted")
            else:
                tokens = len(response) // 4 if response else 0
                ttft = duration * 0.1
                gen_duration = duration

            self.metrics_collector.record_generation(
                tokens=tokens,
                duration_seconds=gen_duration,
                ttft_seconds=ttft,
                draft_n=draft_n,
                draft_n_accepted=draft_n_accepted,
            )

            code = extract_code(extract_source)
            validation_result = self.validator.validate(code, level_def.test_code)
            passed = validation_result.passed
            points = level_def.points if passed else 0

            if passed:
                total_score += points
                levels_passed += 1
                print(f"PASS (+{points})")
            else:
                print(f"FAIL ({validation_result.error or 'failed'})")

            # gpumod-76l.3: artifact stores tagged reasoning+content so
            # we can post-mortem mismatch between what the model said and
            # what _extract_code returned, without losing either field.
            artifact = LevelArtifact(
                level=level_num,
                name=level_def.name,
                points_possible=level_def.points,
                prompt=level_def.prompt,
                response=artifact_response,
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

                # gpumod-msy8: strip closed <think> blocks before extracting so
                # the saved L*_code.py matches what the validator actually scored
                # (the validation path strips think; see _strip_think_blocks).
                # The full raw response — including any <think> trace — is still
                # preserved verbatim in L*_response.txt above.
                code = extract_code(_strip_think_blocks(level.response))
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
            "model": _model_identity(run.model),
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
        "models": [_model_identity(r.model) for r in all_runs],
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
        # gpumod-nor9: choices derive from the registry — adding a known-arch
        # model is one REGISTRY entry, no runner edit.
        choices=[*sorted(REGISTRY), "all"],
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
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Max output tokens per request (default: 32768; lower for degenerate models)",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    models = list(REGISTRY.values()) if args.model == "all" else [REGISTRY[args.model]]
    all_runs: list[BenchmarkRun] = []

    for model in models:
        base_url = args.base_url or f"http://localhost:{model.port}"

        benchmark = ArchitectureBenchmark(
            model=model,
            base_url=base_url,
            iterations=args.iterations,
            output_dir=args.output_dir,
            max_tokens=args.max_tokens,
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
