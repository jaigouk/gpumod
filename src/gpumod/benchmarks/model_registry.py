"""Adapter-based benchmark model registry (gpumod-nor9).

A KNOWN-architecture benchmark model is ONE data entry in ``REGISTRY`` — no
runner edit, no duplicate CLI ``choices`` list. The runner derives its
``--model`` choices from ``sorted(REGISTRY)`` and selects a model by id.

Each entry is a frozen ``ModelSpec`` carrying the model's identity plus its
sampler and (optionally) a ``ResponseNormalizer`` for non-default response
handling. ``normalizer=None`` means "use the suite default" — every model
migrated from the original ``MODELS`` dict keeps ``None`` so behavior is
identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gpumod.benchmarks.coding.sampler_config import (
    GEMMA_CODING,
    THINKING_CODING,
    VIBETHINKER_CODING,
    SamplerConfig,
)

if TYPE_CHECKING:
    from gpumod.benchmarks.normalizers import ResponseNormalizer


@dataclass(frozen=True)
class ModelSpec:
    """Immutable benchmark model definition.

    ``SamplerConfig`` is frozen (hashable) so ``sampler=THINKING_CODING`` is a
    valid direct default — no ``field(default_factory=...)`` needed.
    """

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
    sampler: SamplerConfig = THINKING_CODING
    # None -> suite default (CodeAnswerNormalizer for the coding suite). A NEW
    # architecture gets its own normalizer here without touching the runner.
    normalizer: ResponseNormalizer | None = None
    max_tokens: int = 32768


REGISTRY: dict[str, ModelSpec] = {
    "qwen36-27b": ModelSpec(
        id="qwen36-27b",
        name="Qwen3.6-27B",
        architecture="dense-27B",
        repo="unsloth/Qwen3.6-27B-GGUF",
        quant="Q4_K_M",
        file="Qwen3.6-27B-Q4_K_M.gguf",
        port=7100,
        service_id="qwen36-27b-q4",
    ),
    "qwen36-35b-a3b": ModelSpec(
        id="qwen36-35b-a3b",
        name="Qwen3.6-35B-A3B",
        architecture="moe-35B-A3B",
        repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        quant="UD-Q4_K_S",
        file="Qwen3.6-35B-A3B-UD-Q4_K_S.gguf",
        port=7101,
        service_id="qwen36-35b-a3b-q4",
    ),
    "qwen36-35b-a3b-iq4xs": ModelSpec(
        id="qwen36-35b-a3b-iq4xs",
        name="Qwen3.6-35B-A3B",
        architecture="moe-35B-A3B",
        repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        quant="UD-IQ4_XS",
        file="Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
        port=7099,
        service_id="qwen36-35b-a3b-iq4xs",
    ),
    "qwen36-27b-mtp-q4": ModelSpec(
        id="qwen36-27b-mtp-q4",
        name="Qwen3.6-27B MTP",
        architecture="dense-27B+mtp",
        repo="unsloth/Qwen3.6-27B-MTP-GGUF",
        quant="UD-Q4_K_XL",
        file="Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf",
        port=7102,
        service_id="qwen36-27b-mtp-q4",
    ),
    "qwen36-35b-a3b-mtp-iq4xs": ModelSpec(
        id="qwen36-35b-a3b-mtp-iq4xs",
        name="Qwen3.6-35B-A3B MTP",
        architecture="moe-35B-A3B+mtp",
        repo="unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        quant="UD-IQ4_XS",
        file="Qwen3.6-35B-A3B-MTP-UD-IQ4_XS.gguf",
        port=7103,
        service_id="qwen36-35b-a3b-mtp-iq4xs",
    ),
    "qwen36-35b-a3b-mtp-iq4xs-preserve": ModelSpec(
        id="qwen36-35b-a3b-mtp-iq4xs-preserve",
        name="Qwen3.6-35B-A3B MTP (preserve_thinking)",
        architecture="moe-35B-A3B+mtp",
        repo="unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        quant="UD-IQ4_XS",
        file="Qwen3.6-35B-A3B-MTP-UD-IQ4_XS.gguf",
        port=7104,
        service_id="qwen36-35b-a3b-mtp-iq4xs-preserve",
    ),
    "qwen35-35b-a3b-heretic-mtp-q3kl-preserve": ModelSpec(
        id="qwen35-35b-a3b-heretic-mtp-q3kl-preserve",
        name="Qwen3.5-35B-A3B heretic MTP (preserve_thinking)",
        architecture="moe-35B-A3B+mtp",
        repo="llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF",
        quant="Q3_K_L",
        file="Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-Q3_K_L.gguf",
        port=7105,
        service_id="qwen35-35b-a3b-heretic-mtp-q3kl-preserve",
    ),
    "gemma4-e4b": ModelSpec(
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
    "gemma4-e2b-qat-q2": ModelSpec(
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
    "gemma4-e2b-qat-q4": ModelSpec(
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
    "gemma4-12b-q4": ModelSpec(
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
    "gemma4-12b-q5": ModelSpec(
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
    "gemma4-12b-q8": ModelSpec(
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
    "gemma4-26b-a4b-q4": ModelSpec(
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
    "gemma4-26b-a4b-qat-q4": ModelSpec(
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
    "gemma4-26b-a4b-qat-mtp-q4": ModelSpec(
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
    "siq1-35b-q4km": ModelSpec(
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
    "agentworld-35b-a3b-q4": ModelSpec(
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
    "vibethinker-3b-q8": ModelSpec(
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
