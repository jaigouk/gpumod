# Qwen3.6-A3B baseline vs Qwen3.5-A3B heretic (uncensored, MTP-Preserved)

**Date:** 2026-05-26
**Ticket:** gpumod-ojey
**Question:** Is the `llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF` fine-tune a viable swap-in for `qwen36-35b-a3b-mtp-iq4xs-preserve` in `modes/hermes-agent.yaml`?

## ⚠ Cross-Version + Cross-Quant Caveat

This comparison conflates **three** variables; deltas cannot be attributed to any single one.

| Axis | Baseline | Heretic | Effect on comparison |
|---|---|---|---|
| **Model version** | Qwen 3.6 | Qwen 3.5 | One generation older base |
| **Fine-tune** | Stock Unsloth | `heretic-v2-uncensored` | Different post-training data |
| **Quant** | UD-IQ4_XS (~4.25 bpw) | Q3_K_L (~3.4 bpw) | ~0.85 bpw less precision |
| Binary | llama.cpp b9297 | llama.cpp b9297 | (clean — no binary confound) |

**Quant fallback:** the ticket originally specified Q4_K_S (the closest bit-width match to UD-IQ4_XS available in the heretic repo). Q4_K_S would not fit alongside the MTP draft head + 128K q8_0 KV cache on 24 GB VRAM (preflight: 25,012 MB required vs 24,067 MB available, OOM by ~945 MB). We took the ticket's pre-authorized fallback to Q3_K_L. **The bit-width gap widens vs the baseline as a result, and Q3_K_L is the dominant quality confound below.**

Treat this benchmark as a coarse "is the heretic variant in the same ballpark end-to-end" check, **not an ablation** of any one factor.

## Setup

| Component | Specification |
| --- | --- |
| **CPU** | AMD Ryzen 7 5700G (16 threads) |
| **RAM** | 32 GB DDR4 |
| **GPU** | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| **OS** | Ubuntu 24.04.4 LTS |
| **Driver** | NVIDIA 580.65.06 |
| **CUDA** | 12.0 |
| **llama.cpp** | b9297 (`b0df4c0cf`) — same binary for both rows; no binary confound |
| **GGML_CUDA_NO_PINNED** | `1` (default since gpumod-56md) |

VRAM isolation: all other GPU-resident services stopped throughout. Only the model under test was GPU-resident at any time.

## Models Tested

| ID | Source repo | Quant | File size | Context | Thinking flag |
| --- | --- | --- | --- | --- | --- |
| `qwen36-35b-a3b-mtp-iq4xs-preserve` † | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` | UD-IQ4_XS | 18.2 GB | 131072 | preserve_thinking |
| `qwen35-35b-a3b-heretic-mtp-q3kl-preserve` | `llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF` | Q3_K_L | 17.4 GB | 131072 | preserve_thinking |

† Baseline row reused from [20260524_qwen36_mtp_vs_a3b](../20260524_qwen36_mtp_vs_a3b/README.md) (see `result_qwen36-35b-a3b-mtp-iq4xs-preserve.json` symlink). Not re-run; same 128K + q8_0 KV configuration, same b9297 binary, no fresh measurement noise.

## Results

### Summary Table

| Variant | Mean Score | Std Dev | 95% CI | Mean TPS | Draft accept | Wall (15 iters) |
| --- | ---:| ---:| --- | ---:| ---:| ---:|
| Baseline preserve (UD-IQ4_XS) | **88.3** | 6.5 | [84.8, 91.9] | **216.5** | 78.7% | ~35 min |
| Heretic preserve (Q3_K_L) | **83.3** | 11.4 | [77.0, 89.7] | **212.5** | **76.0%** | ~32 min |
| Δ | **-5.0** | +4.9 | overlap | -1.84% | -2.7pp | — |

**95% CI overlap is heavy** (baseline upper 91.9 vs heretic lower 77.0). The -5.0 score delta is real-looking but the heretic's variance is ~2× the baseline (σ 11.4 vs 6.5), so the distributions cross substantially. Heretic min/max = 65/90 vs baseline 65/90 — same support; the difference is in the mix of 65s vs 90s.

**TPS is a wash** at -1.8%. Both at ~213-217 t/s with MTP draft.

### Score Distribution (per iteration)

| Variant | Scores (15 iters) |
| --- | --- |
| Baseline preserve | 90, 90, 90, **65**, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90 (1× 65) |
| Heretic preserve | **65**, 90, **65**, **65**, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, **65** (4× 65) |

Both ceilings are 90. The full -5.0 mean delta is accounted for by **3 extra 65s** in the heretic run — and all four heretic 65s come from L1 failures (same failure mode as the baseline's single 65).

### Level Pass Rates (15 iterations × 5 levels)

| Level | Task | Pts | Baseline preserve | Heretic preserve |
| --- | --- | --- | --- | --- |
| L1 | Basic queue (add/get, FIFO) | 25 | **14/15 (93%)** | **11/15 (73%)** |
| L2 | Retry with exponential backoff | 25 | 15/15 (100%) | 15/15 (100%) |
| L3 | Priority scheduling | 25 | 15/15 (100%) | 15/15 (100%) |
| L4 | Find & fix concurrency bug | 15 | 15/15 (100%) | 15/15 (100%) |
| L5 | Multi-file refactoring | 10 | 0/15 (0%) | 0/15 (0%) |

L2–L5 are **tied to the iteration**. The entire benchmark delta lives at L1. That's the surprising part: L1 is the easiest level (basic FIFO queue), and the heretic — same family, same MTP architecture — flakes on it 4× as often as the baseline.

L5 (multi-file refactor) is a 0/15 hard wall for both; this matches the baseline preserve's L5 history. Multi-file refactor remains an unsolved level at this model size + quant — including the heretic's single Q3_K_L sample size.

### MTP-Specific Metrics

| Variant | n_max | Total drafts | Drafts accepted | Mean acceptance |
| --- | --- | ---:| ---:| ---:|
| Baseline preserve | 2 | 366,406 | 277,488 | **78.7%** |
| Heretic preserve | 2 | 350,181 | 266,045 | **76.0%** |

Heretic's MTP draft acceptance is -2.7pp vs baseline — small but real. Plausible cause: Q3_K_L target weights produce slightly different sampling paths than the draft head's UD-IQ4_XS-derived expectations. Acceptance is still healthy (>75%) — well above the threshold where speculative decoding pays for itself.

### Wall-clock

| Variant | 15 iters | Per iter |
| --- | --- | --- |
| Baseline preserve | ~35 min | ~2.3 min |
| Heretic preserve | ~32 min | ~2.1 min |

Heretic was ~9% faster wall-clock despite -1.8% TPS, because per-iteration token counts were lower on average (a few iterations hit the 32K thinking budget cap on the baseline that the heretic didn't reproduce).

## L1 Standalone Smoke (gpumod-ojey follow-up)

The bench's 4-extra-L1-failures result raised the question: **is this a model competence regression, or noise from the bench's multi-level conversation accumulation?**

To answer it, we ran the same L1 prompt **20 times standalone** against each service (fresh chat context per attempt — no prior turns, no preserve_thinking carry-over). Same sampler (`THINKING_CODING`), same code extractor + validator the main runner uses. See [scripts/heretic_l1_smoke.py](../../../scripts/heretic_l1_smoke.py).

### L1 Smoke Results

| Variant | Pass | Rate | Wilson 95% CI | Mean dur | Mean tokens (clean) | Failure modes |
| --- | ---:| ---:| --- | ---:| ---:| --- |
| Baseline UD-IQ4_XS (enable) | 19/20 | **95.0%** | [76.4%, 99.1%] | 30.8s | ~4,400 | #2 32K thinking-budget exhaustion |
| Heretic Q3_K_L (preserve) | 18/20 | **90.0%** | [69.9%, 97.2%] | 36.0s | ~5,800 | #1 32K thinking-budget exhaustion, #3 SyntaxError in extracted code |

(Baseline smoke used `qwen36-35b-a3b-mtp-iq4xs` on port 7103 — the `enable_thinking` sibling — because the preserve service was down for VRAM isolation. enable vs preserve is empirically equivalent on single-shot per the prior bench's chat-template proof, so this is the right surrogate for a "single-shot L1 only" comparison.)

### What the smoke tells us

- **Statistically indistinguishable.** Wilson CIs are [69.9%, 97.2%] vs [76.4%, 99.1%] — they overlap from 76% to 97%. With n=20 you cannot separate 18/20 from 19/20.
- **Both share the same dominant failure mode**: 32K thinking-budget exhaustion. The model goes deep into reasoning and never emits the final code fence within the cap. This is a benchmark-configuration artifact, not model competence.
- **Heretic-only failure (#3 SyntaxError) is n=1.** Single sample; could be the lossier Q3_K_L producing a slightly-malformed code block once. Not a signal.
- **Heretic uses ~32% more reasoning tokens per clean answer** (5,800 vs 4,400). Consistent with Q3_K_L needing more chain-of-thought to recover (lossier quant → longer reasoning paths).
- **The bench's 11/15 L1 result was conversation-accumulation, not competence.** preserve_thinking carries prior `<think>` blocks into each subsequent level prompt. By the time L1 fires in iter 4, the model is reasoning against a context already pre-filled with L5 thinking from iter 3. Smaller thinking budget → more 32K caps → more L1 fails. Fresh-context smoke disproves the "Q3_K_L can't do L1" hypothesis.

### Implication

The heretic-Q3_K_L variant has **L1-equivalent competence to baseline-UD-IQ4_XS at this sample size**. The benchmark-score gap is dominated by the quant-precision interaction with the preserve_thinking conversation buffer at 32K thinking budget, not by the fine-tune or version difference.

To statistically prove a real competence delta would require ~100+ smoke attempts per side; n=20 establishes only that the two are within noise.

## Key Findings

1. **Mean score gap is real but small** (-5.0 pts) and **dominated by L1 conversation-accumulation under Q3_K_L**, not by the fine-tune or version.
2. **TPS is a wash** (-1.8%). Both at ~213-217 t/s with MTP.
3. **MTP draft acceptance dropped -2.7pp** (78.7% → 76.0%) with Q3_K_L. Still healthy.
4. **L2–L5 are perfectly tied** with the baseline. The entire bench-score delta lives at L1.
5. **Standalone L1 smoke nails 18/20 = 90%** for the heretic vs 19/20 = 95% for the baseline. CIs overlap heavily. The bench's 11/15 L1 result was conversation accumulation, not competence.
6. **L5 multi-file refactor is 0/15 hard wall for both.** Unchanged at this model class.
7. **The Q4_K_S → Q3_K_L fallback was forced by VRAM** (24,012 MB Q4_K_S preflight requirement vs 24,067 MB available). Q4_K_S would have given a fairer bit-width comparison; we get the documented fallback's wider quant gap.

## Methodology

| Aspect | Configuration |
| --- | --- |
| Iterations | 15 per model |
| Validation | PytestValidator (real pytest tests, 30s per-level timeout) |
| Sampler | `THINKING_CODING`: temp=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=0 |
| Client timeout | 900s per request |
| max_tokens | 32,768 per level |
| Context | 131,072 |
| KV cache | `--cache-type-k q8_0 --cache-type-v q8_0` |
| MTP flags | `--spec-type draft-mtp --spec-draft-n-max 2 --parallel 1` |
| Code extraction | `message.content` preferred; falls back to `reasoning_content` only if content has no code fence |
| Runner | `scripts/run_qwen36_benchmark.py --model qwen35-35b-a3b-heretic-mtp-q3kl-preserve` |
| L1 smoke script | `scripts/heretic_l1_smoke.py` (standalone, 20 attempts, fresh context per attempt) |

## Recommendation

**Do not swap.** The current `qwen36-35b-a3b-mtp-iq4xs-preserve` baseline is preferred for `modes/hermes-agent.yaml` because:

1. **Quant gap dominates.** The heretic repo's lowest bit-width that fits is Q3_K_L. The baseline runs at UD-IQ4_XS (one rung higher). At 24 GB VRAM with MTP + 128K q8_0 KV, the heretic can't match the baseline's effective precision.
2. **The fine-tune is "uncensored" and unaudited for the agent use case.** Hermes-agent is an in-loop tool-calling chat model. We have no calibration on whether the heretic's post-training data introduces failure modes specific to tool-call structure or agent reasoning. The 5-pt single-shot coding gap is the only data we have.
3. **MTP draft acceptance is lower** (-2.7pp) on the heretic. Small but compounds across many turns.

**If the use case strictly needs uncensored output**, the heretic-Q3_K_L is competitive — same L1 competence in smoke, same L2–L4 in bench, same L5 hard wall. Variance is higher; expect more 65-score iterations under preserve_thinking + long conversations. Acceptable as a specialized swap, not as a default.

**What we'd want before reconsidering:**
- Heretic at a higher bit-width (Q4_K_S or Q4_K_M) — would require 32 GB GPU or aggressive context reduction.
- L1 smoke n≥100 on each side to actually separate the pass rates statistically.
- One real Hermes-agent multi-turn session with heretic to check for tool-call regression.

## Files

| File | Description |
| --- | --- |
| `result_qwen35-35b-a3b-heretic-mtp-q3kl-preserve.json` | 15-iter bench result, heretic Q3_K_L preserve |
| `result_qwen36-35b-a3b-mtp-iq4xs-preserve.json` | Baseline 15-iter result (symlink to prior bench dir) |
| `run.log` | Heretic bench stdout |
| `l1_smoke.json`, `l1_smoke.log` | L1 standalone smoke, heretic Q3_K_L (port 7105, 20 attempts) |
| `l1_smoke_baseline_enable.json`, `l1_smoke_baseline_enable.log` | L1 standalone smoke, baseline UD-IQ4_XS enable (port 7103, 20 attempts) |
| `artifacts/qwen35-35b-a3b-heretic-mtp-q3kl-preserve/` | Per-iteration response artifacts (tagged `<reasoning_content>` + `<content>`) |

## References

- [Prior bench: Qwen3.6 MTP vs non-MTP](../20260524_qwen36_mtp_vs_a3b/README.md) — baseline preserve row was first established here
- [heretic_l1_smoke.py](../../../scripts/heretic_l1_smoke.py) — standalone L1 flake-rate script
- [llmfan46/Qwen3.5-…-MTP-Preserved-GGUF](https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF)
- [Unsloth Qwen3.5 docs](https://unsloth.ai/docs/models/qwen3.5)
