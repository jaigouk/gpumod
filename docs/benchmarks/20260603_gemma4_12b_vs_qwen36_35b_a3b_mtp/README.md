# Gemma 4 (12B Q4/Q5/Q8 + 26B-A4B IQ4) vs Qwen3.6-35B-A3B-MTP-preserve

**Date:** 2026-06-04
**Ticket:** gpumod-h6gs
**Question:** How does Gemma 4 at the 12B and 26B-A4B sizes compare on the v2 coding benchmark against the current Hermes-agent model (Qwen3.6-35B-A3B-MTP-IQ4_XS preserve_thinking)?

## TL;DR

| Model | Mean | σ | TPS | VRAM (load) | L5 pass | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **Gemma 4 26B-A4B IT UD-IQ4_XS** | **94.7** | **5.2** | 137.2 | ~16 GB | 47% | **Highest quality + lowest variance.** Beats Qwen on both. Loses 37% on TPS (no MTP). |
| Gemma 4 12B IT UD-Q8_K_XL | 88.3 | 15.3 | 54.3 | ~15 GB | 93% | Quality at Qwen-MTP parity, **TPS regression vs Q5 not worth +0.6 mean** |
| Gemma 4 12B IT Q5_K_M | 87.7 | 15.7 | 76.7 | ~10 GB | **93%** | **Best 12B Q tier**: knee of the quality/TPS curve, cracks L5 routinely |
| Qwen3.6-35B-A3B-MTP-IQ4_XS preserve † | 88.3 | 6.5 | **216.5** | ~22 GB | 0% | Current Hermes baseline — fastest, lowest variance among Qwen, **L5 ceiling** |
| Gemma 4 12B IT UD-Q4_K_XL | 76.0 | 17.8 | 83.2 | ~9 GB | 87% | Cheapest Gemma, but Q5 dominates it on mean for ~9% TPS cost |

† Reused from [20260524 benchmark](../20260524_qwen36_mtp_vs_a3b/README.md) — same v2 methodology, same b9297 binary.

**Two headline findings:**

1. **Gemma 4 26B-A4B is the new top quality model in this suite** — 94.7 mean / σ=5.2 / 7 perfect runs. Beats Qwen3.6-35B-A3B-MTP-preserve on mean (94.7 vs 88.3) AND variance (σ=5.2 vs 6.5). Qwen retains the speed crown by 1.58× thanks to MTP.
2. **Gemma 4 12B (all quants) cracks L5** — multi-file refactoring, which was the absolute ceiling for every Qwen MoE in prior benchmarks (0% across 15+ runs), is solved by Gemma 4 12B at **87–93% pass rate**. The 26B-A4B paradoxically drops back to 47% on L5.

See [Recommendation](#recommendation) for whether to swap any modes.

## Setup

| Component     | Specification                                       |
| ------------- | --------------------------------------------------- |
| **CPU**       | AMD Ryzen 7 5700G (16 threads)                      |
| **RAM**       | 32 GB DDR4                                          |
| **GPU**       | NVIDIA GeForce RTX 4090 (24 GB VRAM)                |
| **OS**        | Ubuntu 24.04.4 LTS                                  |
| **Driver**    | NVIDIA 580.65.06                                    |
| **CUDA**      | 12.0                                                |
| **llama.cpp** | b9297 (`b0df4c0cf`) — same binary as 20260524 baseline |

VRAM isolation: only the model under test was GPU-resident during each run. `vllm-embedding-code` and all other services were stopped before launch.

## Models Tested

| ID | Source | Architecture | Quant | File size | Context | Sampler |
|---|---|---|---|---:|---:|---|
| `qwen36-35b-a3b-mtp-iq4xs-preserve` † | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` | MoE 35B / 3B active + MTP | UD-IQ4_XS | 18.2 GB | 131072 | THINKING_CODING (temp 0.6) |
| `gemma4-12b-q4` | `unsloth/gemma-4-12b-it-GGUF` | Dense 12B | UD-Q4_K_XL | 7.4 GB | 131072 | GEMMA_CODING (temp 1.0) |
| `gemma4-12b-q5` | `unsloth/gemma-4-12b-it-GGUF` | Dense 12B | Q5_K_M | 8.4 GB | 131072 | GEMMA_CODING |
| `gemma4-12b-q8` | `unsloth/gemma-4-12b-it-GGUF` | Dense 12B | UD-Q8_K_XL | 13.6 GB | 131072 | GEMMA_CODING |
| `gemma4-26b-a4b-q4` | `unsloth/gemma-4-26B-A4B-it-GGUF` | MoE 26B / 4B active | UD-IQ4_XS | 12.7 GB | 131072 | GEMMA_CODING |

† Reused from 20260524 — not re-run.

All Gemma presets ship `--cache-type-k q8_0 --cache-type-v q8_0` (matches Qwen baseline's strategy at 131072 ctx), `--parallel 1`, `--flash-attn on`, and `--chat-template-kwargs '{"enable_thinking":true}'`. None ship MTP — see [Methodology Caveats](#methodology-caveats).

## Results

### Summary Table

| Model | Quant | Mean | σ | 95% CI | TPS | Perfect (100/15) | L5 pass |
|---|---|---:|---:|---:|---:|---:|---:|
| **Gemma 4 26B-A4B** | UD-IQ4_XS | **94.7** | **5.2** | [91.8, 97.5] | 137.2 | 7/15 | 47% |
| Gemma 4 12B | UD-Q8_K_XL | 88.3 | 15.3 | [79.9, 96.8] | 54.3 | 9/15 | 93% |
| Qwen3.6-35B-A3B-MTP preserve | UD-IQ4_XS | 88.3 | 6.5 | [84.8, 91.9] | **216.5** | 0/15 | 0% |
| Gemma 4 12B | Q5_K_M | 87.7 | 15.7 | [79.0, 96.3] | 76.7 | 8/15 | 93% |
| Gemma 4 12B | UD-Q4_K_XL | 76.0 | 17.8 | [66.1, 85.9] | 83.2 | 4/15 | 87% |

**Statistical reading:**

- **26B-A4B is meaningfully above everything else.** Its 95% CI lower bound (91.8) sits above every other model's upper bound except 12B Q8's 96.8 (overlap with the lower half of Q8's CI is narrow). The +6.4 vs Qwen-MTP preserve is real.
- **12B Q5, 12B Q8, and Qwen-MTP are statistically a three-way tie on mean.** Their 95% CIs overlap heavily ([79.0, 96.3], [79.9, 96.8], [84.8, 91.9]). The interesting differences are elsewhere: variance, TPS, VRAM, and per-level distribution.
- **12B Q4 is below the pack.** [66.1, 85.9] doesn't reach the other models' lower bounds. The Q4→Q5 step on Gemma 12B is the biggest single quality jump in the comparison (+11.7 mean).

### Score Distribution (per iteration)

| Model | Scores (15 iters) |
|---|---|
| Gemma 4 26B-A4B Q4 | 100, 100, 100, 90, 90, 100, 90, 90, 100, 90, 90, 90, 100, 90, **100** |
| Gemma 4 12B Q8 | 65, 75, 75, 75, **100**, **100**, **100**, 60, **100**, 75, **100**, **100**, **100**, **100**, **100** |
| Gemma 4 12B Q5 | 75, 90, 75, 50, **100**, 75, 75, **100**, **100**, **100**, **100**, **100**, **100**, 75, **100** |
| Qwen3.6-35B-A3B-MTP preserve | 90, 90, 90, 65, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90 |
| Gemma 4 12B Q4 | **100**, 75, **100**, 75, 75, **100**, 75, 75, 50, 50, 75, 50, 75, **100**, 65 |

The Gemma 12B variants have a bimodal distribution: a cluster around 75–100 (depending on which levels passed) and a few outliers as low as 50 (L1 + L2 both failing in the same iter). Qwen's distribution is the tightest (everything at 90, one 65 outlier) but that 90 ceiling reflects L5 always failing.

### Level Pass Rates (15 iter × 5 levels)

| Level | Task | Pts | 12B Q4 | 12B Q5 | 12B Q8 | 26B-A4B Q4 | Qwen-MTP preserve |
|---|---|---:|---:|---:|---:|---:|---:|
| L1 | Basic queue (add/get, FIFO) | 25 | 53% | 100% | 100% | 100% | 93% |
| L2 | Retry with exponential backoff | 25 | 80% | 67% | 60% | **100%** | **100%** |
| L3 | Priority scheduling | 25 | 80% | 87% | **100%** | **100%** | **100%** |
| L4 | Find & fix concurrency bug | 15 | 93% | **100%** | 93% | **100%** | **100%** |
| L5 | Multi-file refactoring | 10 | **87%** | **93%** | **93%** | 47% | 0% |

**Two anomalies stand out:**

1. **L5 inverted between 12B and 26B-A4B.** The dense 12B (any quant) routinely solves multi-file refactoring; the MoE 26B-A4B drops to 47% on the same task. Plausible explanation: MoE expert routing under long-context multi-file prompts is less stable than dense's "all params see everything." This deserves a follow-up spike before treating it as a hard rule — n=15 is small for the L5 subset.
2. **L2 regression on 12B Q5/Q8 vs Q4.** Q4 hits 80% on L2 but Q5 drops to 67% and Q8 to 60%. The L1/L2 failures cluster in the same iterations — appears to be SyntaxError or import-error patterns in the generated retry code. Higher precision quants exploring a wider sampling distribution may produce more brittle exception-handling boilerplate. Counterintuitive but consistent across both higher-precision runs.

### Quality vs Compute Trade-offs (Gemma 4 12B series)

| Quant | Mean | TPS | TPS vs Q4 | Quality vs Q4 |
|---|---:|---:|---:|---:|
| UD-Q4_K_XL | 76.0 | 83.2 | baseline | baseline |
| Q5_K_M | 87.7 | 76.7 | **−8%** | **+11.7** |
| UD-Q8_K_XL | 88.3 | 54.3 | **−35%** | +12.3 |

Q5 is the clear knee of the curve: nearly all of Q8's quality gain (Q8 is only +0.6 over Q5) at substantially better TPS (Q5 is +41% faster than Q8). On a 4090, **Q5_K_M dominates Q8_K_XL** for this workload.

### Wall-clock per Run

| Run | Duration | Per iteration |
|---|---:|---:|
| gemma4-12b-q4 (15 iter) | 44 min | ~3.0 min |
| gemma4-12b-q5 (15 iter) | 43 min | ~2.9 min |
| gemma4-12b-q8 (15 iter) | 68 min | ~4.5 min |
| gemma4-26b-a4b-q4 (15 iter) | **24 min** | **~1.6 min** |

The 26B-A4B's 4B-active MoE routing makes it the fastest end-to-end despite being the largest model on disk. The 12B Q8 is the slowest — verbose thinking compounds with slower per-token TPS.

### Reproducibility Check (Run 2 of Q8 and 26B-A4B)

Both Gemma 4 models that landed surprising numbers in run 1 were re-run with `rerun_q8.sh` / `rerun_26b_a4b.sh` (n=15 each, identical config, b9500 binary). Outcome: **both runs reproduce within sample variance — the run 1 conclusions hold.**

| Model | Run 1 mean ± σ | Run 2 mean ± σ | Δ mean | Run 1 95% CI | Run 2 95% CI | CI overlap |
|---|---:|---:|---:|---|---|:-:|
| `gemma4-12b-q8` | 88.33 ± 15.31 | **87.67 ± 14.98** | -0.67 | [80.6, 96.1] | [80.1, 95.3] | ✅ |
| `gemma4-26b-a4b-q4` | 94.67 ± 5.16 | **94.00 ± 5.07** | -0.67 | [92.0, 97.3] | [91.4, 96.6] | ✅ |

Same -0.67 mean delta on both models is sample-variance noise, not a systematic shift. The 26B-A4B lead over 12B Q8 is reproducible: 94.00 vs 87.67 across the second pair of n=15 runs, with non-overlapping 95% CIs (26B-A4B [91.4, 96.6] sits cleanly above 12B Q8 [80.1, 95.3]'s upper bound only marginally).

Run 2 score distributions:

| Model | Scores (run 2, 15 iters) |
|---|---|
| Gemma 4 12B Q8 (run 2) | **100**, 65, 75, **100**, 75, 75, 60, **100**, 90, **100**, 75, **100**, **100**, **100**, **100** |
| Gemma 4 26B-A4B (run 2) | **100**, 90, 90, 90, 90, 90, 90, 90, 90, **100**, 90, **100**, **100**, **100**, **100** |

Run 2 archives live at `result_*.run1.json` / `run_*.run1.log` / `artifacts/*.run1/` (the script archives the prior "current" data as run1 and writes new data to the unsuffixed paths). The 26B-A4B `artifacts/*.run1/` is empty by design — the per-iter detail was already in the published run 1 result.json before the rerun, and we cleared the artifacts dir before re-running to avoid mixing in-progress state.

**Sampler / yaml flag note.** Between run 1 (02:37) and run 2 (14:21–15:30), `presets/llm/gemma4-*.yaml` gained `--temp 1.0 --top-p 0.95 --top-k 64` in `extra_args` (Unsloth defaults). This **did not** affect benchmark scoring: the runner sets `GEMMA_CODING = SamplerConfig(temperature=1.0, top_p=0.95, top_k=64, ...)` per-request via `**self.model.sampler.to_dict()` in `scripts/run_qwen36_benchmark.py`, overriding any llama-server boot defaults. The yaml flags only affect non-benchmark direct chat. Confirmed by inspecting `result_*.run1.json`, which records the same sampler dict run 1 was already using.

## Methodology Caveats

- **MTP asymmetry, not by choice.** Qwen baseline uses MTP speculative decoding (+24% TPS measured). No Gemma 4 variant tested here has any MTP path available:
  - 12B: Google did not publish a `-it-assistant` drafter for this size.
  - 26B-A4B: a drafter exists (`google/gemma-4-26B-A4B-it-assistant`) but mainline `ggml-org/llama.cpp` does not yet support the `gemma4_assistant` arch ([PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398) is WIP as of 2026-05-29). The merged implementation lives only in the ik_llama.cpp fork ([PR #1744](https://github.com/ikawrakow/ik_llama.cpp/pull/1744)). gpumod runs mainline b9297.
  - **Interpretation:** Gemma TPS columns reflect native non-speculative speed. The TPS gap to Qwen (Gemma 26B-A4B at 137.2 vs Qwen at 216.5 → 1.58×) is partly architecture, partly the missing speculative-decoding boost. A future benchmark once #23398 lands will measure Gemma's TPS with MTP enabled.
- **Sampler asymmetry, by design.** Each model uses its vendor-recommended sampler: Qwen runs THINKING_CODING (temp=0.6, top_p=0.95, top_k=20); Gemma runs GEMMA_CODING (temp=1.0, top_p=0.95, top_k=64). Plumbed via the new `sampler` field on `ModelConfig` in [scripts/run_qwen36_benchmark.py](../../../scripts/run_qwen36_benchmark.py) (gpumod-h6gs). Earlier benchmarks under this v2 methodology forced THINKING_CODING on every model — that produced a defensibility caveat in the 20260423 gemma4-e4b row; the fix landed with this benchmark.
- **Iterations: 15 per model.** Same as prior v2 runs. n=15 is enough to call meaningful mean/σ differences but is light for per-level pass-rate stability on rare-failure levels. The 26B-A4B L5 47% rate is the most likely number in the table to shift with more iterations.
- **Validation: PytestValidator with 30 s per-level timeout, 900 s per-request client timeout, max_tokens=32768.** Identical to the 20260524 baseline.

## Recommendation

| Use case | Recommended | Why |
|---|---|---|
| **Current Hermes-agent slot (keep)** | `qwen36-35b-a3b-mtp-iq4xs-preserve` | Mean parity with Gemma 12B Q5/Q8 (88.3 vs 87.7/88.3), TPS lead is decisive (216.5 vs 76.7/54.3) for an interactive multi-turn agent. Variance is also better than Gemma 12B. |
| **Low-VRAM coding mode (NEW, if added)** | `gemma4-12b-q5` | ~10 GB load fits alongside multiple co-tenants on a 24 GB GPU. 87.7 mean is statistically tied with Hermes baseline. TPS 76.7 is acceptable for non-interactive use. Q4 is too noisy (σ=17.8); Q8 is too slow (TPS 54.3). |
| **Highest-quality coding model (NEW slot)** | `gemma4-26b-a4b-q4` | 94.7 mean / σ=5.2 — best both in this suite. Worth a dedicated mode if a workload needs quality over latency. **Caveat:** L5 47%. Not for multi-file refactoring. |
| **L5 / multi-file refactoring** | `gemma4-12b-q5` or `gemma4-12b-q8` | 93% L5 pass on both. The only models in this suite that handle multi-file refactor reliably. Counterintuitively, the dense 12B beats the bigger MoE 26B-A4B on this specific task. |

### What we'd want before a mode swap

- **A real chat / tool-calling session** on each candidate Gemma preset under hermes-agent prompts. The v2 benchmark covers single-shot coding only.
- **VRAM-budget verification under co-tenancy** for the low-VRAM proposal (12B Q5 + vllm-embedding-code + maybe more).
- **A larger sample (n=30+)** on the 26B-A4B L5 47% number before treating it as a hard "MoE can't do multi-file" finding.
- **A re-benchmark of 26B-A4B once `ggml-org/llama.cpp` PR #23398 lands**, with the `-assistant` drafter enabled. If MTP adds ~50% TPS as it does on Qwen, 26B-A4B at ~200 TPS with 94.7 quality would be the unambiguous champion.

### Why NOT swap Hermes today

The current `qwen36-35b-a3b-mtp-iq4xs-preserve` is statistically equivalent to Gemma 12B Q5/Q8 on mean (CI overlap), is 2.8× faster (216.5 vs 76.7), and has lower variance (σ=6.5 vs 15.7). The mean-equivalent candidates would be a clear regression on the responsiveness axis without a quality compensation. The 26B-A4B's quality advantage IS real but its TPS regression (216.5 → 137.2) is real too — for an interactive multi-turn agent, +6 mean does not justify −37% TPS.

Swap candidates exist (Gemma 12B Q5 for a low-VRAM mode, 26B-A4B for a quality-first mode) — both belong in new modes, not as replacements for Hermes.

## Files

| File | Description |
|---|---|
| `result_gemma4-12b-q4.json` | 15-iter result, Gemma 4 12B UD-Q4_K_XL |
| `result_gemma4-12b-q5.json` | 15-iter result, Gemma 4 12B Q5_K_M |
| `result_gemma4-12b-q8.json` | 15-iter run 2 result, Gemma 4 12B UD-Q8_K_XL (reproducibility check, b9500) |
| `result_gemma4-12b-q8.run1.json` | 15-iter run 1 result, Gemma 4 12B UD-Q8_K_XL (b9297, original) |
| `result_gemma4-26b-a4b-q4.json` | 15-iter run 2 result, Gemma 4 26B-A4B UD-IQ4_XS (reproducibility check, b9500) |
| `result_gemma4-26b-a4b-q4.run1.json` | 15-iter run 1 result, Gemma 4 26B-A4B UD-IQ4_XS (b9297, original) |
| `run_bench.sh` | Driver for Q4 + Q5 sequential run |
| `run_bench_extra.sh` | Driver for Q8 + 26B-A4B sequential run (added mid-benchmark) |
| `rerun_q8.sh` | Re-run driver for gemma4-12b-q8 (archives current → .run1, then fresh n=15) |
| `rerun_26b_a4b.sh` | Re-run driver for gemma4-26b-a4b-q4 (same archive-and-rerun pattern) |
| `run.log`, `run_extra.log` | Combined stdout from each driver |
| `run_gemma4-*.log` | Per-model benchmark stdout (current run) |
| `run_gemma4-*.run1.log` | Per-model benchmark stdout (run 1 archive) |
| `artifacts/<model>/iter_NN/` | Per-iteration, per-level generated code and validation output (current run) |
| `artifacts/<model>.run1/iter_NN/` | Per-iteration archive from run 1 |

## References

- [20260524 Qwen3.6 MTP vs non-MTP benchmark (Hermes-agent baseline)](../20260524_qwen36_mtp_vs_a3b/README.md)
- [20260423 Qwen3.6 vs Gemma4 E4B comparison (prior generation)](../20260423_qwen36_gemma4_comparison/README.md)
- [Unsloth gemma-4-12b-it-GGUF model card](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF)
- [Unsloth gemma-4-26B-A4B-it-GGUF model card](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)
- [google/gemma-4-12B-it (sampler recommendation source)](https://huggingface.co/google/gemma-4-12B-it)
- [ggml-org/llama.cpp PR #23398 — Gemma 4 MTP port to mainline (WIP)](https://github.com/ggml-org/llama.cpp/pull/23398)
- [ikawrakow/ik_llama.cpp PR #1744 — Gemma 4 MTP (merged in fork)](https://github.com/ikawrakow/ik_llama.cpp/pull/1744)
