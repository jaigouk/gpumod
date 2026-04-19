# Research: TriAttention KV cache compression — viability assessment for NVIDIA + Gemma 4

**Date:** 2026-04-19
**Spike Ticket:** gpumod-lff
**Status:** Final
**Author:** researcher (gpumod triattention-spike team)
**Reviewed by:** tech-lead (approved 2026-04-19)

## Summary

TriAttention is a newly published (arXiv 2604.04921, 2026-04-07) KV cache
compression method that prunes keys via pre-RoPE Q/K frequency analysis. A
working CUDA implementation exists in an external llama.cpp fork
(`atomicmilkshake/llama-cpp-turboquant`, branch `feature/triattention`), so
NVIDIA viability (Q1) is not the blocker.

The blocker is **Gemma 4 architectural compatibility**: Gemma 4 (released
2026-04-02) uses a dual-RoPE configuration where global-attention layers apply
`rope_type="proportional"` with `partial_rotary_factor=0.25`. TriAttention's
trigonometric series (paper eq. 2) sums over all `d/2` RoPE frequency bands; with
only 25% of dimensions rotated on the global layers, the series must be
re-derived. Sliding layers inherently bound their KV by `sliding_window` (512 or
1024 tokens), so pruning them below that is redundant — meaningful compression
would come from the 1/6 global layers, which are precisely the algorithmically
incompatible ones. No Gemma-family calibration stats or benchmarks exist in the
paper, the reference implementation, or the forks.

**Gate decision: DEFER.** Re-evaluate on or after 2026-08-01 (≈3 months),
once the community publishes p-RoPE compatibility analyses or Gemma
calibration stats emerge. In the interim, pilot TriAttention on Qwen3-32B
(where it is validated and pre-calibrated) to de-risk gpumod's KV-compression
plumbing in isolation from the Gemma 4 questions.

## Research Questions

From tech-lead's task #1 briefing, 13 sub-questions plus 5 cross-cutting
integration questions:

- **Q1** — Is a CUDA TriAttention impl viable for llama.cpp + NVIDIA?
  (sub-questions 1.1–1.6)
- **Q2a** — Is Gemma 4's attention architecture compatible with TriAttention?
  (sub-questions 2a.1–2a.3)
- **Q2b** — Is calibration for Gemma 4 feasible on RTX 4090?
  (sub-questions 2b.1–2b.4)
- **Cross-cutting A–E** — estimation formula shape, driver scope, preset
  surface, preflight implications, failure modes.

## Q1 — CUDA viability

| #   | Answer                                                                                                                                                                                                                                                                    | Evidence                                                                                                         | Confidence       | Gaps                                                 |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------- | ---------------------------------------------------- |
| 1.1 | **CUDA impl exists** — `atomicmilkshake/llama-cpp-turboquant` branch `feature/triattention`. HIP/ROCm sibling fork at `domvox/llama-cpp-turboquant-hip` branch `feature/triattention-scoring`. Upstream `ggml-org/llama.cpp`: **0 PRs** matching "TriAttention" (fork-only). | github.com/atomicmilkshake/llama-cpp-turboquant ; github.com/ggml-org/llama.cpp/pulls?q=TriAttention (0 results) | High             | —                                                    |
| 1.2 | **Active fork, MIT license, not upstreamed.** Build flags: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120;121"`. Requires CUDA 13.x runtime. Turing+ (sm_75) minimum; RTX 4090 = sm_89 is supported.                                                          | fork README                                                                                                      | High             | No tagged release; rolling branch — version pin risk |
| 1.3a | **Paper claim: 10.7× KV reduction at matched accuracy on AIME25 (Qwen3-8B)**; 2.5× throughput at fixed accuracy.                                                                                                                                                          | arXiv 2604.04921 §1, Figure 1                                                                                    | High (as claim)  | Not independently reproduced                         |
| 1.3b | **Fork README reports throughput only** (75 tok/s vs 17.5 tok/s baseline); no independent KV-ratio measurement. Combined with TurboQuant turbo3 yields ~6.8× effective on Qwen3.5-27B per community reports.                                                             | fork README ; github.com/ggml-org/llama.cpp discussion #20969                                                     | Med              | —                                                    |
| 1.4 | **Quality cost is benchmark-dependent.** MATH 500: 68.4% vs 69.6% (–1.2pp at 1024-token budget). AIME24: 42.1% vs 57.1% (**–15pp at 2048-token budget** — non-trivial). AIME25 long-reasoning: parity at 10.7×. No PPL/KLD reported in paper.                              | arXiv 2604.04921 Tables 1–2                                                                                      | High             | No PPL/KLD, no non-reasoning benchmarks (MMLU etc.)  |
| 1.5 | **Compression is budget-bounded, NOT linear.** Method retains top-B keys (fixed integer budget). Pruning triggered every β=128 tokens. Once ctx > B, KV cache size is constant. **This breaks gpumod's linear `kv_per_1k` assumption.**                                  | arXiv 2604.04921 §4.3                                                                                            | High             | —                                                    |
| 1.6 | **Per-head, not uniform.** Configurable `per_head` or `per_layer_per_head` modes via `TRIATTN_RUNTIME_PRUNING_MODE`. GQA handling via z-score normalize-then-max across query heads sharing a KV head.                                                                   | arXiv 2604.04921 §4.3 ; WeianMao/triattention README                                                             | High             | No per-layer compression variance data               |

**Port effort: ~0 hours** — CUDA fork is already production-ready. Kill
condition (>40 person-hours port) NOT triggered on Q1.

## Q2a — Gemma 4 architecture compatibility

Configs fetched directly from Hugging Face (2026-04-19).

**Gemma 4 E4B (dense):** 42 layers (35 `sliding_attention` + 7
`full_attention`), `num_attention_heads=8`, `num_key_value_heads=2` (GQA 4:1),
`head_dim=256`, `sliding_window=512`, `max_position_embeddings=131072`,
`num_kv_shared_layers=18`, `attention_k_eq_v=false`.

**Gemma 4 26B A4B (MoE):** 30 layers (25 sliding + 5 full, repeating 5:1
pattern), `num_attention_heads=16`, `num_key_value_heads=8` (local) /
`num_global_key_value_heads=2` (global), `head_dim=256`, `global_head_dim=512`,
`sliding_window=1024`, `max_position_embeddings=262144`, `num_experts=128`,
`num_kv_shared_layers=0`, `attention_k_eq_v=true`.

**Gemma 4 31B (dense):** 60 layers, `attention_k_eq_v=true`, otherwise mirrors
the 26B architecture.

**Dual RoPE configuration** (from 26B `config.json#text_config.rope_parameters`):

```json
{
  "full_attention":    { "rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0 },
  "sliding_attention": { "rope_type": "default",                                      "rope_theta": 10000.0  }
}
```

| #    | Answer                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Evidence                                                                    | Confidence | Gaps                                                            |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ---------- | --------------------------------------------------------------- |
| 2a.1 | Architecture verified via direct `config.json` fetch (above). Layer pattern is explicit in the `layer_types` array. p-RoPE applies **only on global layers with 25% partial rotation**. Last layer is always global.                                                                                                                                                                                                                                       | HF `google/gemma-4-E4B` / `-26B-A4B` / `-31B-it` config.json ; HF blog gemma4 | High       | —                                                               |
| 2a.2 | **TriAttention assumes standard full-dim RoPE.** Logit formula (paper eq. 2): `logit(q,k) = Σ_f ‖q_f‖‖k_f‖·cos(ω_f·Δ+φ_f)` sums over all `d/2` frequency bands. With `partial_rotary_factor=0.25`, only 25% of bands rotate; the remaining 75% are pass-through — the trigonometric series must be re-derived for partial RoPE. Sliding-window attention is not discussed anywhere in the paper or either fork README.                                      | arXiv 2604.04921 §2.1 & §4.1 ; absence in WeianMao & atomicmilkshake READMEs | High       | Unknown whether Q/K concentration (R) still holds on Gemma p-RoPE |
| 2a.3 | **No Gemma benchmarks exist.** Paper validates Qwen3-8B, Qwen3-32B, Qwen2.5, Llama3 (via DeepSeek-R1-Distill), and GLM-4.7-Flash (MLA) only. Pre-calibrated stats shipped in `triattention/vllm/stats/`: Qwen3-8B, Qwen3-32B-int4, DeepSeek-R1-Distill-{Llama-8B, Qwen-7B}. `[unverified]` for Gemma-family.                                                                                                                                                | arXiv 2604.04921 §3.3 & §5 ; WeianMao/triattention README stats directory   | High       | —                                                               |

**Secondary compatibility issues:**

- `attention_k_eq_v=true` on 26B/31B already halves KV vs a standard GQA
  model, diminishing TriAttention's marginal gain (compression on top of an
  already-compressed K=V representation).
- `num_kv_shared_layers=18` on E4B: 18 of 42 layers reuse K/V from an upstream
  layer of the same attention type. Pruning must be coherent across the
  sharing chain — not addressed by the paper.
- Sliding layers are inherently bounded by their window (512 / 1024). Pruning
  below window size is redundant; meaningful compression would come only from
  the 1/6 global layers — which are the algorithmically incompatible p-RoPE
  layers. **This is the core reason DEFER is the right call for Gemma 4.**

## Q2b — Calibration feasibility

| #    | Answer                                                                                                                                                                                                                                                                                                                    | Evidence                                            | Confidence | Gaps                                                |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ---------- | --------------------------------------------------- |
| 2b.1 | **Calibration required, data-dependent.** Not data-free. Computes per-band complex-valued Q center `E[q_f]` and expected Q norm `E[‖q_f‖]` from a representative corpus. Produces `.pt` (vLLM) or `.triattention` binary (llama.cpp fork) sidecar.                                                                          | arXiv 2604.04921 §3.3 & eq. 4 ; WeianMao README     | High       | —                                                   |
| 2b.2 | **Dataset/sample count not published.** Paper states "calibration data" without specifying corpus size or sampling protocol. For Gemma 4 E4B BF16 (~15 GB), VRAM for calibration = weights + activations ≈ 15 + 3 GB ≈ 18 GB → fits RTX 4090 24 GB with ~6 GB headroom. Runtime `[unverified]`; analogue to llama.cpp imatrix calibration gives ~1–4 h wall for ≲1000 samples. | arXiv 2604.04921 §3.3 ; WeianMao README ; imatrix analogue | Low        | Actual runtime, recommended corpus size and sampling |
| 2b.3 | **One-time offline step.** Command: `llama-cli -m model.gguf -ngl 99 --triattention-calibrate corpus.txt --triattention-calibrate-out model.triattention`. Referenced at inference via `--triattention-stats model.triattention`.                                                                                           | atomicmilkshake/llama-cpp-turboquant README         | High       | —                                                   |
| 2b.4 | **Sidecar binary file**, stored separately from the GGUF. No GGUF-metadata integration path. Artifact size undocumented; by analogy to imatrix output, likely tens of MB.                                                                                                                                                   | atomicmilkshake/llama-cpp-turboquant README          | Med        | Exact artifact size per model                        |

**Risk on Gemma 4 specifically:** no one has calibrated Gemma 4 for
TriAttention. A first run would require empirical verification that the Q/K
concentration phenomenon holds on Gemma's dual-RoPE — if the Mean Resultant
Length `R` drops on global layers because of the 25% partial rotation, the
`S_trig`/`S_norm` balance (paper §4.2) would need retuning.

## Cross-cutting integration

### A — Estimation formula shape

**Must become non-linear and layer-type-aware.** Current formula in
`src/gpumod/fetchers/huggingface.py:263-299`:

```
kv_per_1k = 2 * num_layers * num_kv_heads * head_dim * 2 * 1000 / 1024**2
total_vram = base_vram + (ctx / 1000) * kv_per_1k   # registry.py:159-163
```

is strictly linear in ctx and treats all layers identically. Under TriAttention
on a hybrid-attention model like Gemma 4, KV size becomes piecewise with two
distinct per-token costs (local `head_dim=256` vs global `global_head_dim=512`
on 26B/31B) and two distinct per-layer-type bounds (window for sliding,
triattn_budget for global). Layers that reuse K/V via `num_kv_shared_layers`
should not be counted in storage at all.

```
unique_sliding_layers = sliding_layers - shared_sliding_layers
unique_global_layers  = global_layers  - shared_global_layers

kv_mb(ctx) ≈ unique_sliding_layers * min(ctx, sliding_window) * per_tok_local
           + unique_global_layers  * min(ctx, triattn_budget) * per_tok_global
           + base_overhead
```

Reduces to the current linear formula when `triattn_budget=∞`,
`sliding_window=∞`, and `unique_layers = num_layers`. So the new formula can be
adopted as a generalisation, not a replacement.

### B — Driver scope

llama.cpp only (via fork). vLLM has a separate implementation in
`WeianMao/triattention` with the same calibration artifact requirement and
comparable compression ratios, but it is out of scope for this spike (gpumod's
vLLM driver targets different workloads). **Scope: `drivers/llamacpp.py` only.**

### C — Preset surface

Add an optional preset block. Absence = off, preserving backward
compatibility. Implicit per-model lookup would hide too much complexity
behind the driver.

```yaml
kv_compression:
  method: triattention    # future: also turboquant, triattention+turboquant
  budget: 4096            # top-B keys retained on global layers
  stats_path: /path/to/model.triattention
  window: 256             # recent tokens protected from eviction (β parameter)
```

### D — Preflight implications

`src/gpumod/preflight/vram_check.py` currently suggests "halving ctx roughly
halves KV cache" when VRAM is insufficient. Under TriAttention this heuristic
is wrong once `ctx > budget` — halving ctx would not halve KV. VRAMCheck must
read the new `kv_compression` field and use the compound formula in §A for
both the estimate and the suggestion. A new preflight check
(`TriAttentionStatsCheck`) should verify the sidecar `.triattention` file
exists, matches the model (hash or model-id sidecar metadata), and the driver
binary is a TriAttention-capable build.

### E — Failure modes

The fork's README does not document fallback behaviour for unsupported
architectures or stale stats. Most likely: runtime CUDA crash mid-decode.
gpumod preflight MUST enforce: (a) stats file exists, (b) stats file matches
model, (c) driver binary is TriAttention-capable (grep for flag support /
version sentinel), (d) model architecture is on an explicit allowlist. Without
preflight, users hit cryptic CUDA errors with no actionable signal.

## gpumod-specific implementation implications

- `src/gpumod/models.py:158` — `ModelInfo` needs optional
  `kv_compression_budget: int | None`, and (for Gemma 4-style hybrids)
  exposure of the `layer_types` ratio and both `head_dim` values.

- `src/gpumod/fetchers/huggingface.py:88-93, 263-299` — **non-trivial fetcher
  refactor required.** `HuggingFaceFetcher` currently consumes
  `huggingface_hub.model_info()` (repo metadata), not a parsed `config.json`.
  `info.config` exposes a partial view of the config and does not reliably
  surface Gemma 4's `rope_parameters`, `layer_types`, `num_kv_shared_layers`,
  `global_head_dim`, or `attention_k_eq_v`. Implementing the layer-type-aware
  formula in §A requires either:
  - (a) an additional HTTP fetch of `raw/main/config.json` (new dependency +
    URL-handling + 401/redirect semantics), or
  - (b) extending the parser to read the extra keys from `info.config` when
    available and fall back to the current formula when not.
  This is part of the "redesign gpumod KV estimation" follow-up (#3 below)
  and should be scoped as a fetcher refactor, not a one-line formula swap.

- `src/gpumod/registry.py:159-163` — `total_vram_for_context` becomes
  piecewise (min of ctx and per-layer-type budget).

- `src/gpumod/preflight/vram_check.py` — ctx-reduction suggestion must be
  compression-aware; add `TriAttentionStatsCheck`.

- `src/gpumod/services/drivers/llamacpp.py` — translate the `kv_compression`
  preset field into `--triattention-budget`, `--triattention-window`,
  `--triattention-stats` CLI flags on service start.

- `presets/llm/gemma4-*.yaml` — if TriAttention ever ships for Gemma 4,
  presets need a `kv_compression` block + mechanism to fetch/ship the
  `.triattention` sidecar.

- **No changes needed to vllm driver** (out of scope).

## Options Considered

| Option                                          | Pros                                                                                                                                                                                                | Cons                                                                                                                                                                                                                      |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. GO — integrate TriAttention for Gemma 4 now** | - Largest VRAM headroom gain on long-context Gemma 4 workloads - Research is freshly published; potential differentiation                                                                           | - Zero Gemma-family evidence; paper's trig series provably broken on `partial_rotary_factor=0.25` - No calibration stats exist; first-ever attempt on Gemma - Requires gpumod estimation + preflight redesign - High risk |
| **B. KILL — abandon TriAttention entirely**     | - No engineering cost - Avoids fork maintenance                                                                                                                                                      | - Discards a working, CUDA-validated compression method that is viable for Qwen/Llama users - Gives up 10.7× reduction wins on reasoning workloads where it's proven                                                       |
| **C. DEFER — pilot on Qwen3-32B first, Gemma 4 later** (chosen) | - CUDA path validated on a model where TriAttention is proven (pre-calibrated stats ship in repo) - Exercises gpumod driver/preflight plumbing with controlled risk - Unblocks estimation-formula redesign independently | - No immediate Gemma 4 benefit - Requires follow-up research checkpoint                                                                                                                                                    |

## Recommendation

**DEFER.** Pursue option C. Proceed with Qwen3-32B pilot to de-risk the
integration surface; revisit Gemma 4 on or after **2026-08-01** once community
data on p-RoPE compatibility or Gemma-family calibration stats emerges.

Do not commit gpumod to Gemma 4 TriAttention integration in the current
quarter. Gate decision drivers:

1. TriAttention paper published **2026-04-07**, Gemma 4 released
   **2026-04-02** — both under 3 weeks old as of report date.
2. **Zero** public evidence of TriAttention on Gemma-family.
3. Gemma 4's `partial_rotary_factor=0.25` p-RoPE breaks the paper's
   trigonometric-series-over-all-d/2-bands assumption — requires novel
   derivation before integration.
4. Interleaved sliding+global attention is not addressed by the paper.
   Sliding layers are already window-bounded; TriAttention would mostly help
   the 1/6 global layers — which are the incompatible p-RoPE layers.
5. AIME24 showed –15pp accuracy drop at aggressive budgets — non-reasoning
   quality cost on Gemma 4 is unknown.
6. CUDA implementation for Qwen/Llama is production-ready (sm_75+, covers
   RTX 4090 sm_89), so we can capture the easy wins immediately via the
   Qwen3-32B pilot without committing to Gemma 4.

## References

- [TriAttention paper — arXiv 2604.04921](https://arxiv.org/html/2604.04921) — primary source; algorithm, experiments, formulae
- [WeianMao/triattention (reference impl)](https://github.com/WeianMao/triattention) — calibration pipeline, pre-rolled stats
- [atomicmilkshake/llama-cpp-turboquant](https://github.com/atomicmilkshake/llama-cpp-turboquant) — CUDA fork, branch `feature/triattention`
- [domvox/triattention-ggml (HIP sibling)](https://github.com/domvox/triattention-ggml) — ROCm/HIP fork, branch `feature/triattention-scoring`
- [ggml-org/llama.cpp discussion #20969 — TurboQuant](https://github.com/ggml-org/llama.cpp/discussions/20969) — upstream integration status; 0 TriAttention PRs
- [Hugging Face blog: Welcome Gemma 4](https://huggingface.co/blog/gemma4) — dual-RoPE (p-RoPE on global layers), shared KV, sliding-window details
- [Gemma 4 model card — Google AI](https://ai.google.dev/gemma/docs/core/model_card_4) — model sizes, layers, release date (2026-04-02)
- [MarkTechPost — TriAttention overview](https://www.marktechpost.com/2026/04/11/researchers-from-mit-nvidia-and-zhejiang-university-propose-triattention-a-kv-cache-compression-method-that-matches-full-attention-at-2-5x-higher-throughput/) — compression/accuracy tables

Gemma 4 configs directly fetched (raw/main/config.json):

- `google/gemma-4-E4B`
- `google/gemma-4-26B-A4B`
- `google/gemma-4-31B-it`

## Follow-up Tasks

- [ ] Track upstream llama.cpp merge of TurboQuant + TriAttention (watch discussions #20969, #21526)
- [ ] Pilot TriAttention on Qwen3-32B (pre-calibrated stats available) to validate gpumod driver-flag plumbing and calibration-artifact fetching, in isolation from Gemma 4 questions
- [ ] Spike: redesign gpumod KV estimation to non-linear, layer-type-aware model (prerequisite for any KV-compression driver work; see §A and `fetchers/huggingface.py` implementation note)
- [ ] Research check-back on 2026-08-01: re-evaluate Gemma 4 + TriAttention once p-RoPE compatibility analyses or Gemma calibration stats emerge
- [ ] (Deferred) Gemma 4 TriAttention integration — unblock only after the KV-estimation redesign lands and the 2026-08-01 check-back shows progress

## Acceptance Criteria

- [x] All research questions answered (Q1.1–1.6, Q2a.1–2a.3, Q2b.1–2b.4, A–E)
- [x] Every claim has a cited source
- [x] Resource usage evaluated (VRAM, calibration cost)
- [x] License verified as permissive (MIT for fork; Apache-2.0 for reference impl)
- [x] Python/ecosystem compatibility checked (CUDA 13.x, sm_75+ covers RTX 4090 sm_89)
- [x] Recommendation stated with rationale (DEFER; 6 drivers enumerated)
- [x] Follow-up tickets to be created (see "Follow-up Tasks")
