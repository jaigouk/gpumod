"""PoC: Compound KV cache estimation formula for hybrid-attention models.

Demonstrates the non-linear, layer-type-aware KV cache estimation on 4
reference models, comparing against the current linear formula.

Usage:
    uv run python docs/research/poc/kv_estimation_compound.py

Spike: gpumod-cf8
Date: 2026-05-24
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Minimal config.json fields needed for KV estimation."""

    name: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int | None = None  # If None, derived from hidden_size / num_attention_heads

    # Hybrid attention fields (Gemma 3/3n style)
    layer_types: list[str] | None = None  # e.g. ["sliding_attention", "full_attention", ...]
    sliding_window: int | None = None  # token count
    sliding_window_pattern: int | None = None  # derive layer_types if not explicit

    # Global-layer overrides (Gemma 4 style -- not present in ref models but formula supports)
    num_global_key_value_heads: int | None = None
    global_head_dim: int | None = None

    # KV sharing (Gemma 3n style)
    num_kv_shared_layers: int = 0
    attention_k_eq_v: bool = False  # K=V halves KV storage

    # TriAttention budget (future)
    triattn_budget: int | None = None

    # Source info
    source: str = ""  # URL or description of where config was obtained

    def effective_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.num_attention_heads == 0:
            return 0
        return self.hidden_size // self.num_attention_heads

    def effective_layer_types(self) -> list[str]:
        """Return the layer_types array, deriving from pattern if needed."""
        if self.layer_types is not None:
            return self.layer_types
        if self.sliding_window_pattern is not None:
            p = self.sliding_window_pattern
            return [
                "sliding_attention" if bool((i + 1) % p) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        # Dense model: all layers are "full_attention" (no sliding window)
        return ["full_attention"] * self.num_hidden_layers


@dataclass
class KVEstimate:
    """KV cache estimate for a model at a given context size."""

    model_name: str
    context_size: int
    kv_mb_linear: float  # current gpumod formula
    kv_mb_compound: float  # new compound formula
    kv_per_1k_linear: float  # MB per 1K tokens (linear)
    kv_per_1k_compound: float  # MB per 1K tokens at this ctx (compound, for display)
    detail: str = ""  # breakdown explanation


# ---------------------------------------------------------------------------
# Current linear formula (from fetchers/huggingface.py:263-299)
# ---------------------------------------------------------------------------


def estimate_kv_linear(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_tokens: int,
    k_eq_v: bool = False,
) -> float:
    """Current gpumod formula: 2 * L * H_kv * d * 2 * ctx / 1024^2.

    The '2' factors are: K+V, and fp16 (2 bytes).
    If k_eq_v, K and V are the same tensor, so only 1x instead of 2x.
    """
    kv_factor = 1 if k_eq_v else 2  # K+V or just one shared tensor
    bytes_total = kv_factor * num_layers * num_kv_heads * head_dim * 2 * context_tokens
    return bytes_total / (1024 * 1024)


# ---------------------------------------------------------------------------
# Compound formula (layer-type-aware, non-linear)
# ---------------------------------------------------------------------------


def estimate_kv_compound(config: ModelConfig, context_tokens: int) -> tuple[float, str]:
    """Compound KV cache estimation: layer-type-aware, non-linear.

    Returns (kv_mb, detail_string).

    Formula:
        KV_total = sum over unique layers of:
            - sliding layers: min(ctx, sliding_window) * per_token_bytes
            - full/global layers: min(ctx, triattn_budget or inf) * per_token_bytes
            - global layers may use different head_dim and num_kv_heads

    "Unique" excludes layers that share KV with another layer
    (num_kv_shared_layers).
    """
    layer_types = config.effective_layer_types()
    hd = config.effective_head_dim()
    kv_heads = config.num_key_value_heads
    kv_factor = 1 if config.attention_k_eq_v else 2  # K+V or K=V shared
    bytes_per_elem = 2  # fp16

    # Count layer types
    n_sliding = sum(1 for lt in layer_types if lt == "sliding_attention")
    n_global = sum(1 for lt in layer_types if lt == "full_attention")
    assert n_sliding + n_global == config.num_hidden_layers, (
        f"Layer type mismatch: {n_sliding}+{n_global} != {config.num_hidden_layers}"
    )

    # Determine how shared layers are distributed.
    # Gemma 3n config says num_kv_shared_layers=15 out of 35 total,
    # meaning 15 layers reuse KV from an upstream layer of the same type.
    # Conservative assumption: shared layers are proportionally distributed.
    shared = config.num_kv_shared_layers
    total = config.num_hidden_layers
    if shared > 0 and total > 0:
        # Proportional distribution (round down to be conservative = overestimate)
        shared_sliding = math.floor(shared * n_sliding / total)
        shared_global = shared - shared_sliding
        # Clamp
        shared_sliding = min(shared_sliding, n_sliding)
        shared_global = min(shared_global, n_global)
    else:
        shared_sliding = 0
        shared_global = 0

    unique_sliding = n_sliding - shared_sliding
    unique_global = n_global - shared_global

    # Per-token bytes for each layer type
    # Sliding layers use the base head_dim and kv_heads
    per_tok_sliding = kv_factor * kv_heads * hd * bytes_per_elem

    # Global layers may use different head_dim and kv_heads (Gemma 4 style)
    global_hd = config.global_head_dim if config.global_head_dim is not None else hd
    global_kv_heads = (
        config.num_global_key_value_heads
        if config.num_global_key_value_heads is not None
        else kv_heads
    )
    per_tok_global = kv_factor * global_kv_heads * global_hd * bytes_per_elem

    # Effective context per layer type
    sw = config.sliding_window if config.sliding_window is not None else context_tokens
    ctx_sliding = min(context_tokens, sw)

    budget = config.triattn_budget if config.triattn_budget is not None else context_tokens
    ctx_global = min(context_tokens, budget)

    # Total bytes
    sliding_bytes = unique_sliding * ctx_sliding * per_tok_sliding
    global_bytes = unique_global * ctx_global * per_tok_global
    total_bytes = sliding_bytes + global_bytes
    total_mb = total_bytes / (1024 * 1024)

    detail = (
        f"sliding: {unique_sliding} layers x min({context_tokens}, {sw}) = {ctx_sliding} tok "
        f"x {per_tok_sliding} B/tok = {sliding_bytes / (1024*1024):.1f} MB; "
        f"global: {unique_global} layers x min({context_tokens}, {budget}) = {ctx_global} tok "
        f"x {per_tok_global} B/tok = {global_bytes / (1024*1024):.1f} MB"
    )

    return total_mb, detail


# ---------------------------------------------------------------------------
# Backward-compatibility verification
# ---------------------------------------------------------------------------


def verify_backward_compat(config: ModelConfig) -> bool:
    """Verify that compound formula matches linear when sw=inf, budget=inf, shared=0.

    This is the critical property: the compound formula must be a strict
    generalization, not a replacement.
    """
    # Neutralize hybrid features
    neutralized = ModelConfig(
        name=config.name + " (neutralized)",
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        layer_types=None,  # All full_attention
        sliding_window=None,  # No sliding window (= inf)
        sliding_window_pattern=None,
        num_global_key_value_heads=None,
        global_head_dim=None,
        num_kv_shared_layers=0,
        attention_k_eq_v=False,
        triattn_budget=None,  # inf
    )

    for ctx in [1000, 8000, 32000, 128000]:
        linear = estimate_kv_linear(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=neutralized.effective_head_dim(),
            context_tokens=ctx,
        )
        compound, _ = estimate_kv_compound(neutralized, ctx)
        if abs(linear - compound) > 0.01:
            print(f"  MISMATCH at ctx={ctx}: linear={linear:.2f} MB, compound={compound:.2f} MB")
            return False
    return True


# ---------------------------------------------------------------------------
# Reference models
# ---------------------------------------------------------------------------

MODELS = [
    ModelConfig(
        name="Gemma 3n E4B",
        num_hidden_layers=35,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_size=2048,
        head_dim=256,
        layer_types=(
            ["sliding_attention"] * 4 + ["full_attention"]
        ) * 7,  # 28 sliding + 7 full = 35
        sliding_window=512,
        num_kv_shared_layers=15,
        attention_k_eq_v=False,
        source="google/gemma-3n-E4B-it config.json (text_config), fetched 2026-05-24",
    ),
    ModelConfig(
        name="Gemma 3 27B",
        num_hidden_layers=62,
        num_attention_heads=32,
        num_key_value_heads=16,
        hidden_size=5376,
        head_dim=128,
        # 5:1 pattern derived from sliding_window_pattern=6 in transformers
        # see: huggingface/transformers v5.1.0 configuration_gemma3.py:286-292
        sliding_window=1024,
        sliding_window_pattern=6,  # every 6th layer is full attention
        num_kv_shared_layers=0,
        attention_k_eq_v=False,
        source="google/gemma-3-27b-it config.json (text_config), fetched 2026-05-24; "
        "layer pattern from transformers v5.1.0 Gemma3TextConfig sliding_window_pattern=6",
    ),
    ModelConfig(
        name="Qwen3 32B",
        num_hidden_layers=64,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_size=5120,
        head_dim=128,
        # Dense model: no sliding window, no hybrid attention
        sliding_window=None,  # config.json has sliding_window=null
        layer_types=None,  # All full attention
        num_kv_shared_layers=0,
        attention_k_eq_v=False,
        source="Qwen/Qwen3-32B config.json, fetched 2026-05-24",
    ),
    ModelConfig(
        name="Llama 3.3 70B",
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_size=8192,
        head_dim=128,
        # Dense model: no sliding window
        sliding_window=None,
        layer_types=None,
        num_kv_shared_layers=0,
        attention_k_eq_v=False,
        source="unsloth/Llama-3.3-70B-Instruct config.json, fetched 2026-05-24 "
        "(meta-llama gated; architecture identical per config)",
    ),
]

CONTEXT_SIZES = [8_000, 32_000, 128_000]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 90)
    print("KV Cache Estimation: Linear (current) vs Compound (proposed)")
    print("=" * 90)

    # Step 1: Backward compatibility verification
    print("\n--- Backward Compatibility Verification ---")
    print("Testing: compound formula with sw=inf, budget=inf, shared=0 MUST match linear")
    for model in MODELS:
        ok = verify_backward_compat(model)
        status = "PASS" if ok else "FAIL"
        print(f"  {model.name}: {status}")

    # Step 2: Compute KV per 1K tokens (linear)
    print("\n--- KV Cache per 1K Tokens (linear formula, current gpumod) ---")
    for model in MODELS:
        hd = model.effective_head_dim()
        kv_per_1k_bytes = 2 * model.num_hidden_layers * model.num_key_value_heads * hd * 2 * 1000
        kv_per_1k_mb = math.ceil(kv_per_1k_bytes / (1024 * 1024))
        print(f"  {model.name}: {kv_per_1k_mb} MB/1K tokens")
        print(
            f"    formula: 2 * {model.num_hidden_layers} * {model.num_key_value_heads} * "
            f"{hd} * 2 * 1000 / 1024^2"
        )

    # Step 3: Reference value table
    print("\n--- Reference Value Table ---")
    header = f"{'Model':<20} {'Ctx':>8} {'Linear MB':>12} {'Compound MB':>14} {'Delta':>10} {'%Saved':>8}"
    print(header)
    print("-" * len(header))

    results: list[KVEstimate] = []
    for model in MODELS:
        hd = model.effective_head_dim()
        kv_per_1k_linear = (
            2 * model.num_hidden_layers * model.num_key_value_heads * hd * 2 * 1000
        ) / (1024 * 1024)

        for ctx in CONTEXT_SIZES:
            linear_mb = estimate_kv_linear(
                num_layers=model.num_hidden_layers,
                num_kv_heads=model.num_key_value_heads,
                head_dim=hd,
                context_tokens=ctx,
            )
            compound_mb, detail = estimate_kv_compound(model, ctx)
            delta = linear_mb - compound_mb
            pct = (delta / linear_mb * 100) if linear_mb > 0 else 0

            est = KVEstimate(
                model_name=model.name,
                context_size=ctx,
                kv_mb_linear=linear_mb,
                kv_mb_compound=compound_mb,
                kv_per_1k_linear=kv_per_1k_linear,
                kv_per_1k_compound=compound_mb / (ctx / 1000),
                detail=detail,
            )
            results.append(est)

            print(
                f"{model.name:<20} {ctx:>8,} {linear_mb:>12,.1f} {compound_mb:>14,.1f} "
                f"{delta:>10,.1f} {pct:>7.1f}%"
            )

    # Step 4: Detail breakdown for hybrid models
    print("\n--- Detail Breakdown (hybrid models only) ---")
    for est in results:
        if est.kv_mb_linear != est.kv_mb_compound:
            print(f"\n  {est.model_name} @ {est.context_size:,} ctx:")
            print(f"    {est.detail}")

    # Step 5: Effective KV/1K at each context size
    print("\n--- Effective KV MB per 1K Tokens (compound formula, varies with context) ---")
    header2 = f"{'Model':<20} {'Ctx':>8} {'Linear/1K':>12} {'Compound/1K':>14}"
    print(header2)
    print("-" * len(header2))
    for est in results:
        print(
            f"{est.model_name:<20} {est.context_size:>8,} "
            f"{est.kv_per_1k_linear:>12.1f} {est.kv_per_1k_compound:>14.1f}"
        )


if __name__ == "__main__":
    main()
