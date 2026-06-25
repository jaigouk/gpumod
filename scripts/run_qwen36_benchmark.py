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
import re
import sys
import time
from dataclasses import asdict, dataclass, field
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
from gpumod.benchmarks.coding.sampler_config import (
    GEMMA_CODING,
    THINKING_CODING,
    VIBETHINKER_CODING,
    SamplerConfig,
)
from gpumod.benchmarks.coding.scoring import calculate_confidence_interval, calculate_stats

DEFAULT_OUTPUT_DIR = Path("docs/benchmarks/20260423_qwen36_gemma4_comparison")

# gpumod-msy8: VibeThinker-3B (and other Qwen2-template reasoning models) emit
# <think>...</think> reasoning into message.content. Their plain ChatML template
# is a "content-only" chat format in llama.cpp, so the reasoning is NOT split
# into reasoning_content even with --reasoning-format deepseek. The model often
# drafts ```python fences INSIDE <think>, so extract_code (first-fence-wins)
# would validate the draft instead of the post-</think> final answer. We strip
# CLOSED <think> blocks from content before the fence search. An unterminated
# <think> (budget exhausted mid-thought) is left intact so the reasoning-fallback
# path can still mine it for a last-resort draft. No-op for models whose content
# has no <think> (e.g. Gemma, which routes thinking to reasoning_content).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_blocks(text: str) -> str:
    """Remove complete <think>...</think> spans from a model response."""
    return _THINK_BLOCK_RE.sub("", text).strip()


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
    # gpumod-h6gs: per-model sampler. Defaults to Qwen's THINKING_CODING so
    # existing entries behave unchanged. Gemma 4 overrides with GEMMA_CODING
    # (temp=1.0, top_p=0.95, top_k=64) per Google's model card recommendation.
    sampler: SamplerConfig = field(default_factory=lambda: THINKING_CODING)


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
    "qwen36-35b-a3b-iq4xs": ModelConfig(
        id="qwen36-35b-a3b-iq4xs",
        name="Qwen3.6-35B-A3B",
        architecture="moe-35B-A3B",
        repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        quant="UD-IQ4_XS",
        file="Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
        port=7099,
        service_id="qwen36-35b-a3b-iq4xs",
    ),
    "qwen36-27b-mtp-q4": ModelConfig(
        id="qwen36-27b-mtp-q4",
        name="Qwen3.6-27B MTP",
        architecture="dense-27B+mtp",
        repo="unsloth/Qwen3.6-27B-MTP-GGUF",
        quant="UD-Q4_K_XL",
        file="Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf",
        port=7102,
        service_id="qwen36-27b-mtp-q4",
    ),
    "qwen36-35b-a3b-mtp-iq4xs": ModelConfig(
        id="qwen36-35b-a3b-mtp-iq4xs",
        name="Qwen3.6-35B-A3B MTP",
        architecture="moe-35B-A3B+mtp",
        repo="unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        quant="UD-IQ4_XS",
        file="Qwen3.6-35B-A3B-MTP-UD-IQ4_XS.gguf",
        port=7103,
        service_id="qwen36-35b-a3b-mtp-iq4xs",
    ),
    "qwen36-35b-a3b-mtp-iq4xs-preserve": ModelConfig(
        id="qwen36-35b-a3b-mtp-iq4xs-preserve",
        name="Qwen3.6-35B-A3B MTP (preserve_thinking)",
        architecture="moe-35B-A3B+mtp",
        repo="unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        quant="UD-IQ4_XS",
        file="Qwen3.6-35B-A3B-MTP-UD-IQ4_XS.gguf",
        port=7104,
        service_id="qwen36-35b-a3b-mtp-iq4xs-preserve",
    ),
    "qwen35-35b-a3b-heretic-mtp-q3kl-preserve": ModelConfig(
        id="qwen35-35b-a3b-heretic-mtp-q3kl-preserve",
        name="Qwen3.5-35B-A3B heretic MTP (preserve_thinking)",
        architecture="moe-35B-A3B+mtp",
        repo="llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF",
        quant="Q3_K_L",
        file="Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-Q3_K_L.gguf",
        port=7105,
        service_id="qwen35-35b-a3b-heretic-mtp-q3kl-preserve",
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
    # gpumod-kpmq.2: small Gemma 4 E2B QAT (2-bit mobile) — fast harness dev model.
    "gemma4-e2b-qat-q2": ModelConfig(
        id="gemma4-e2b-qat-q2",
        name="Gemma 4 E2B IT QAT UD-Q2_K_XL",
        architecture="dense-E2B",
        repo="unsloth/gemma-4-E2B-it-qat-mobile-GGUF",
        quant="QAT UD-Q2_K_XL",
        file="gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf",
        port=7112,
        service_id="gemma4-e2b-qat-q2",
        sampler=GEMMA_CODING,
    ),
    # gpumod-kpmq.5: standard (non-mobile) Gemma 4 E2B QAT — UD-Q4_K_XL (recommended
    # tier). The proper E2B benchmark model; the q2 mobile (2-bit) above is degenerate.
    "gemma4-e2b-qat-q4": ModelConfig(
        id="gemma4-e2b-qat-q4",
        name="Gemma 4 E2B IT QAT UD-Q4_K_XL",
        architecture="dense-E2B",
        repo="unsloth/gemma-4-E2B-it-qat-GGUF",
        quant="QAT UD-Q4_K_XL",
        file="gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
        port=7113,
        service_id="gemma4-e2b-qat-q4",
        sampler=GEMMA_CODING,
    ),
    # gpumod-h6gs: Gemma 4 12B presets, non-speculative (no Gemma 4 12B MTP
    # drafter exists upstream; ggml-org/llama.cpp PR #23398 WIP).
    "gemma4-12b-q4": ModelConfig(
        id="gemma4-12b-q4",
        name="Gemma 4 12B IT UD-Q4_K_XL",
        architecture="dense-12B",
        repo="unsloth/gemma-4-12b-it-GGUF",
        quant="UD-Q4_K_XL",
        file="gemma-4-12b-it-UD-Q4_K_XL.gguf",
        port=7106,
        service_id="gemma4-12b-q4",
        sampler=GEMMA_CODING,
    ),
    "gemma4-12b-q5": ModelConfig(
        id="gemma4-12b-q5",
        name="Gemma 4 12B IT Q5_K_M",
        architecture="dense-12B",
        repo="unsloth/gemma-4-12b-it-GGUF",
        quant="Q5_K_M",
        file="gemma-4-12b-it-Q5_K_M.gguf",
        port=7107,
        service_id="gemma4-12b-q5",
        sampler=GEMMA_CODING,
    ),
    "gemma4-12b-q8": ModelConfig(
        id="gemma4-12b-q8",
        name="Gemma 4 12B IT UD-Q8_K_XL",
        architecture="dense-12B",
        repo="unsloth/gemma-4-12b-it-GGUF",
        quant="UD-Q8_K_XL",
        file="gemma-4-12b-it-UD-Q8_K_XL.gguf",
        port=7108,
        service_id="gemma4-12b-q8",
        sampler=GEMMA_CODING,
    ),
    "gemma4-26b-a4b-q4": ModelConfig(
        id="gemma4-26b-a4b-q4",
        name="Gemma 4 26B-A4B IT UD-IQ4_XS",
        architecture="moe-26B-A4B",
        repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        quant="UD-IQ4_XS",
        file="gemma-4-26B-A4B-it-UD-IQ4_XS.gguf",
        port=7109,
        service_id="gemma4-26b-a4b-q4",
        sampler=GEMMA_CODING,
    ),
    "gemma4-26b-a4b-qat-q4": ModelConfig(
        id="gemma4-26b-a4b-qat-q4",
        name="Gemma 4 26B-A4B IT QAT UD-Q4_K_XL",
        architecture="moe-26B-A4B",
        repo="unsloth/gemma-4-26B-A4B-it-qat-GGUF",
        quant="QAT UD-Q4_K_XL",
        file="gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        port=7110,
        service_id="gemma4-26b-a4b-qat-q4",
        sampler=GEMMA_CODING,
    ),
    # gpumod-kpmq.5: MTP (speculative) variant of the gemma4-26b-a4b-qat — same base
    # GGUF + an MTP drafter (faster gen). Served on the same port 7110 (alternative
    # preset; run one at a time).
    "gemma4-26b-a4b-qat-mtp-q4": ModelConfig(
        id="gemma4-26b-a4b-qat-mtp-q4",
        name="Gemma 4 26B-A4B IT QAT UD-Q4_K_XL + MTP",
        architecture="moe-26B-A4B+mtp",
        repo="unsloth/gemma-4-26B-A4B-it-qat-GGUF",
        quant="QAT UD-Q4_K_XL",
        file="gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        port=7110,
        service_id="gemma4-26b-a4b-qat-mtp-q4",
        sampler=GEMMA_CODING,
    ),
    # gpumod-nq8v spike: SIQ-1-35B Q4_K_M direct llama-server (no preset
    # landed). The runner connects via --base-url override; service_id is
    # metadata only here.
    "siq1-35b-q4km": ModelConfig(
        id="siq1-35b-q4km",
        name="SIQ-1-35B Q4_K_M (no MTP — weights unpublished)",
        architecture="qwen35moe-hybrid-35B-A3B",
        repo="AlexWortega/SIQ-1-35B",
        quant="Q4_K_M",
        file="SIQ-1-35B.Q4_K_M.gguf",
        port=18210,
        service_id="siq1-35b-q4km",
        sampler=THINKING_CODING,
    ),
    # gpumod-qsgl: Qwen-AgentWorld-35B-A3B — hybrid Gated-DeltaNet MoE, llama.cpp arch
    # qwen35moe (same family as siq1 above). Refreshed 2026-06-25 to the official Unsloth
    # Dynamic GGUF (UD-Q4_K_S). The old gguf-my-repo quant needed two --override-kv flags
    # for a conversion defect (declared block_count=41 but shipped 40 blocks); the Unsloth
    # build declares block_count=40 with no nextn key (verified from the GGUF header), so
    # no override is needed. Still started via its preset (`gpumod service start
    # agentworld-35b-a3b-q4`) for the 131072 context + card sampling, not a bare server.
    "agentworld-35b-a3b-q4": ModelConfig(
        id="agentworld-35b-a3b-q4",
        name="Qwen-AgentWorld-35B-A3B UD-Q4_K_S",
        architecture="qwen35moe-hybrid-35B-A3B",
        repo="unsloth/Qwen-AgentWorld-35B-A3B-GGUF",
        quant="UD-Q4_K_S",
        file="Qwen-AgentWorld-35B-A3B-UD-Q4_K_S.gguf",
        port=7111,
        service_id="agentworld-35b-a3b-q4",
        sampler=THINKING_CODING,
    ),
    # gpumod-msy8: VibeThinker-3B — dense Qwen2 3B reasoning-tuned model, Q8_0
    # near-lossless. Architecture/size A/B vs the 26B QAT MoE baseline (NOT a
    # same-model quant A/B). Card caveat: NOT trained for tool-calling/agents;
    # the coding suite is single-turn codegen so the comparison is valid.
    # VIBETHINKER_CODING sampler honors the card's temp=1.0 (top_k 20, see
    # sampler_config.py for the top_k=-1 divergence note).
    "vibethinker-3b-q8": ModelConfig(
        id="vibethinker-3b-q8",
        name="VibeThinker-3B Q8_0",
        architecture="dense-3B",
        repo="prithivMLmods/VibeThinker-3B-GGUF",
        quant="Q8_0",
        file="VibeThinker-3B.Q8_0.gguf",
        port=7115,
        service_id="vibethinker-3b-q8",
        sampler=VIBETHINKER_CODING,
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
            # gpumod-msy8: strip closed <think> blocks from content first so the
            # fence search below sees only the post-</think> final answer (see
            # _strip_think_blocks). Raw `response` is kept for the artifact.
            content_final = _strip_think_blocks(response)
            if "```" in content_final:
                extract_source = content_final
            elif reasoning:
                extract_source = reasoning + "\n\n" + content_final
            else:
                extract_source = content_final
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
        choices=[
            "qwen36-27b",
            "qwen36-35b-a3b",
            "qwen36-35b-a3b-iq4xs",
            "qwen36-27b-mtp-q4",
            "qwen36-35b-a3b-mtp-iq4xs",
            "qwen36-35b-a3b-mtp-iq4xs-preserve",
            "qwen35-35b-a3b-heretic-mtp-q3kl-preserve",
            "gemma4-e4b",
            "gemma4-e2b-qat-q2",
            "gemma4-e2b-qat-q4",
            "gemma4-12b-q4",
            "gemma4-12b-q5",
            "gemma4-12b-q8",
            "gemma4-26b-a4b-q4",
            "gemma4-26b-a4b-qat-q4",
            "gemma4-26b-a4b-qat-mtp-q4",
            "siq1-35b-q4km",
            "agentworld-35b-a3b-q4",
            "vibethinker-3b-q8",
            "all",
        ],
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

    models = list(MODELS.values()) if args.model == "all" else [MODELS[args.model]]
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
