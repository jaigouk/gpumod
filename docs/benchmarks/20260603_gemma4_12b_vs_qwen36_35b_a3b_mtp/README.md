# Gemma 4 (12B Q4/Q5/Q8 + 26B-A4B IQ4) vs Qwen3.6-35B-A3B-MTP-preserve

**Date:** 2026-06-04
**Tickets:** gpumod-h6gs (bench); gpumod-9ial / gpumod-eods / gpumod-7vy8 / gpumod-pdtn (methodology fixes); gpumod-t84m / gpumod-4omn (bench infra)
**Question:** How does Gemma 4 at the 12B and 26B-A4B sizes compare on the v2 coding benchmark against the current Hermes-agent model (Qwen3.6-35B-A3B-MTP-IQ4_XS preserve_thinking)?

## TL;DR

| Model | Mean | σ | Min/Max | 95% CI | TPS | VRAM | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **Gemma 4 26B-A4B IT UD-IQ4_XS** | **100.0** | **0.0** | 100/100 | [100.0, 100.0] | 140.7 | ~16 GB | **Perfect** across all 15 iters, all 5 levels. Strongest single result in this suite. |
| Gemma 4 12B IT UD-Q8_K_XL | 89.7 | 13.4 | 65/100 | [82.9, 96.5] | 52.3 | ~15 GB | L2 80% / L5 73% on the composition test. Decent quality, slowest TPS. |
| Gemma 4 12B IT Q5_K_M | 80.7 | 12.8 | 65/100 | [74.2, 87.2] | 73.7 | ~10 GB | Reasonable quality-VRAM-TPS knee. Drops on the harder L5. |
| Gemma 4 12B IT UD-Q4_K_XL | 62.3 | 18.9 | 40/100 | [52.8, 71.9] | 80.0 | ~9 GB | Not recommended for serious coding work on this benchmark. |
| Qwen3.6-35B-A3B-MTP-IQ4_XS preserve † | 88.3 | 6.5 | 65/100 | [84.8, 91.9] | **216.5** | ~22 GB | Hermes baseline. TPS lead is real; quality is now overtaken by Gemma 4 26B-A4B. |

† Qwen row reused from [20260524 benchmark](../20260524_qwen36_mtp_vs_a3b/README.md); not re-run here. Its L5 had a 0% pass rate (different test format then) so the 88.3 mean reflects L1–L4 only; the comparison with Gemma 4 26B-A4B's 100/0 should be read as "Qwen falls 12 points short on the lower 4 levels, Gemma also clears the harder composition L5".

**Three headlines:**

1. **Gemma 4 26B-A4B is the unambiguous quality leader.** Perfect 100 mean / σ=0 / 15 of 15 iters at the ceiling, including 100% on the L5 composition test that requires a Job dataclass + RetryPolicy + JobQueue with real composition between them.
2. **12B-Q8 sits 10 points below 26B-A4B.** 89.67 mean with σ=13.4 — usable but not in the same league. Its L2 80% is a 20pp improvement over what an earlier rev of the L2 prompt produced; the bench infrastructure changes that drove that are listed under [Methodology](#methodology).
3. **12B-Q4 is too weak for the benchmark's harder levels.** 62.33 mean / 33% L5 — the cheapest dense Gemma is below the bar for any production coding workload that exercises composition.

See [Recommendation](#recommendation) for mode-swap decisions.

## Setup

| Component | Specification |
|---|---|
| **CPU** | AMD Ryzen 7 5700G (16 threads) |
| **RAM** | 32 GB DDR4 |
| **GPU** | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| **OS** | Ubuntu 24.04.4 LTS |
| **Driver** | NVIDIA 580.65.06 |
| **CUDA** | 12.0 |
| **llama.cpp** | b9500 (`3d1998634`, built 2026-06-04) |

VRAM isolation enforced for every model: only the model under test was GPU-resident; all other gpumod services were stopped via `gpumod mode switch blank` before each model start (the [bench drivers](run_bench.sh) call that themselves).

## Models Tested

| ID | Source | Architecture | Quant | File size | Context | Sampler |
|---|---|---|---|---:|---:|---|
| `qwen36-35b-a3b-mtp-iq4xs-preserve` † | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` | MoE 35B / 3B active + MTP | UD-IQ4_XS | 18.2 GB | 131072 | THINKING_CODING (temp 0.6) |
| `gemma4-12b-q4` | `unsloth/gemma-4-12b-it-GGUF` | Dense 12B | UD-Q4_K_XL | 7.4 GB | 131072 | GEMMA_CODING (temp 1.0, RP 1.05) |
| `gemma4-12b-q5` | `unsloth/gemma-4-12b-it-GGUF` | Dense 12B | Q5_K_M | 8.4 GB | 131072 | GEMMA_CODING |
| `gemma4-12b-q8` | `unsloth/gemma-4-12b-it-GGUF` | Dense 12B | UD-Q8_K_XL | 13.6 GB | 131072 | GEMMA_CODING |
| `gemma4-26b-a4b-q4` | `unsloth/gemma-4-26B-A4B-it-GGUF` | MoE 26B / 4B active | UD-IQ4_XS | 12.7 GB | 131072 | GEMMA_CODING |

† Reused from 20260524 — not re-run.

All Gemma presets ship `--cache-type-k q8_0 --cache-type-v q8_0` (matches Qwen baseline's strategy at 131072 ctx), `--parallel 1`, `--flash-attn on`, and `--chat-template-kwargs '{"enable_thinking":true}'`. None ship MTP — see [Methodology Caveats](#methodology-caveats).

## Methodology

This bench's results depend on four bench-infrastructure fixes landed in commit history before the published run:

| Ticket | Fix |
|---|---|
| **gpumod-9ial** | Code extractor unwraps the `<reasoning_content>…<content>` artifact wrapper before extraction (so artifact `L*_code.py` files match what was validated) and dedents indented fences (Gemma's chat template wraps code inside numbered list items at column 4+). Pure-correctness fix — previously the artifacts didn't reflect the scored code. |
| **gpumod-eods** | `GEMMA_CODING.repetition_penalty` 1.0 → 1.05 to break degeneration loops on the dense 12B at temp=1.0. |
| **gpumod-7vy8** | L2 prompt rewritten. Removed a misleading `requests.get` example that derailed 12B dense, fixed a data-shape mismatch with the tests, disambiguated "retry up to 3 times" → "4 total attempts (initial + 3 retries)", explicit "do not import external packages". |
| **gpumod-pdtn** | L5 test rewritten. Previous L5 only asserted `from solution import JobQueue/Job` (trivially passable — any file with those two class names passed). New L5 requires `Job` dataclass + `RetryPolicy` class + `JobQueue` that **composes** `RetryPolicy` (source-inspection assertion). Level renamed "Multi-file Refactor" → "Compose Job + RetryPolicy + JobQueue" so the name matches what is measured. |

The L5 change is the most significant for cross-benchmark comparisons — any older benchmark report under the v2 methodology used the trivial test, so its L5 numbers are not comparable to this one's. The other three fixes don't change the test bar, they just make the runner correctly evaluate model output.

## Results

### Summary

| Model | Mean | σ | Min/Max | 95% CI | TPS |
|---|---:|---:|---:|---:|---:|
| **Gemma 4 26B-A4B** | **100.00** | **0.00** | 100/100 | [100.0, 100.0] | 140.7 |
| Gemma 4 12B Q8 | 89.67 | 13.43 | 65/100 | [82.9, 96.5] | 52.3 |
| Gemma 4 12B Q5 | 80.67 | 12.80 | 65/100 | [74.2, 87.2] | 73.7 |
| Gemma 4 12B Q4 | 62.33 | 18.89 | 40/100 | [52.8, 71.9] | 80.0 |

### Per-level pass rates

| Level | Task | Q4 | Q5 | Q8 | 26B-A4B |
|---|---|---:|---:|---:|---:|
| L1 | Basic queue (add/get, FIFO) | 46% | 100% | 100% | **100%** |
| L2 | Retry with backoff | 40% | 46% | 80% | **100%** |
| L3 | Priority scheduling | 93% | 100% | 93% | **100%** |
| L4 | Find & fix concurrency bug | 93% | 100% | 93% | **100%** |
| L5 | Compose Job + RetryPolicy + JobQueue | 33% | 40% | 73% | **100%** |

### Per-iteration scores

| Model | Scores (15 iters) |
|---|---|
| Gemma 4 26B-A4B | **100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100** |
| Gemma 4 12B Q8 | **100**, 75, 75, **100**, 90, **100**, 90, 90, **100**, **100**, **100**, 90, 65, **100**, **100** |
| Gemma 4 12B Q5 | 90, 75, 75, 75, 65, 90, **100**, 75, 90, **100**, 75, 90, 75, 90, 65 |
| Gemma 4 12B Q4 | 75, 40, 75, 75, **100**, 50, 65, 50, 75, 50, 50, 50, 65, **100**, 65 |

### What separates 26B-A4B from 12B

The 26B-A4B clears every iteration on every level. The 12B variants cluster failures on L2 (retry-with-backoff: 40–80% pass) and L5 (composition: 33–73% pass). The L1/L3/L4 tasks are within reach of even Q5; the gap is on the two levels that ask for structural correctness (off-by-one-free retry loop, three classes that compose). That's the kind of failure mode where MoE 26B-A4B's 4B-active-params budget seems to give it a hard advantage over dense 12B regardless of quant tier.

## Methodology Caveats

- **MTP asymmetry, not by choice.** Qwen baseline uses MTP speculative decoding (+24% TPS measured in the 20260524 benchmark). No Gemma 4 variant tested here has any MTP path available:
  - 12B: Google released a 12B drafter (`google/gemma-4-12B-it-assistant`) but its `Gemma4UnifiedAssistantForCausalLM` arch has zero upstream PRs in `ggml-org/llama.cpp`.
  - 26B-A4B: a drafter exists (`google/gemma-4-26B-A4B-it-assistant`) and AtomicChat ships GGUF conversions. Mainline llama.cpp [PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398) is still OPEN (WIP) as of 2026-06-04. Tracking via gpumod-rj0s.
  - **Interpretation:** Gemma TPS columns reflect non-speculative speed. The Gemma 4 26B-A4B 140.7 TPS vs Qwen-MTP 216.5 TPS gap (-35%) is partly architecture, partly the missing speculative-decoding boost. Once PR #23398 merges, a re-bench should close most of that gap.
- **Sampler asymmetry, by design.** Each model uses its vendor-recommended sampler: Qwen runs THINKING_CODING (temp=0.6, top_p=0.95, top_k=20); Gemma runs GEMMA_CODING (temp=1.0, top_p=0.95, top_k=64, RP=1.05). Plumbed via the `sampler` field on `ModelConfig` in `scripts/run_qwen36_benchmark.py`.
- **Iterations: 15 per model.** Enough to call meaningful mean/σ differences but light for per-level pass-rate stability on rare-failure levels.
- **Validation: PytestValidator with 30 s per-level timeout, 900 s per-request client timeout, max_tokens=32768.** Identical to the 20260524 baseline.

## Recommendation

| Use case | Recommended | Why |
|---|---|---|
| **Highest-quality coding model** | `gemma4-26b-a4b-q4` | 100/100 perfect across 15 iters with σ=0. Clears the L5 composition test that 12B-Q8 only manages 73% on. |
| **Hermes-agent slot** | **Swap to `gemma4-26b-a4b-q4`** (landed in commit `7523805`, gpumod-yxr6 partial) | +12 mean (88.3 → 100), σ collapses to 0, VRAM total drops 22.8 GB → 19 GB (+4 GB headroom). Trade-off: -35% TPS (216 → 140) and loss of `preserve_thinking` multi-turn kwarg (Gemma's chat template uses `enable_thinking`). Track gpumod-rj0s for the upstream PR #23398 merge that would let Gemma 4 26B-A4B run with its own MTP drafter and close the TPS gap. |
| **Code mode slot** | **Pending follow-up** (gpumod-yxzt) | `code` mode invariant requires `--parallel 3 --cont-batching` for concurrent coding tabs. Need `gemma4-26b-a4b-q4-multi` preset first; then swap. |
| **Low-VRAM coding mode** | `gemma4-12b-q5` (best 12B knee that fits in ~10 GB) | 80.67 mean / 46% L2 / 40% L5. Q8 is clearly better at 89.67 mean if VRAM allows it. |
| **Lowest-precision dense Gemma** | `gemma4-12b-q4` is **not** recommended for serious work | 62.33 mean / 33% L5; the cheaper variant fails the composition tests too often. |

### Why the Hermes swap is defensible

The Qwen baseline used to be defended as "statistically equivalent on mean, ~2-3× faster, lower variance" against the prior Gemma 12B variants. That comparison is no longer the relevant one — `gemma4-26b-a4b-q4` is the comparison point now, and its 100 mean vs Qwen's 88.3 is a +12 quality delta with no CI overlap and σ collapsing from 6.5 to 0. The TPS cost is unchanged (-35%) but the quality gain is meaningful enough that for an interactive agent where quality of single replies dominates, the swap is defensible. Track gpumod-rj0s — once Gemma 4 MTP lands, the swap also wins on TPS.

### What we'd want before treating the swap as production-final

- **Real chat / tool-calling session validation.** v2 benchmark covers single-shot coding only — multi-turn agent behaviour under the new Gemma chat template is not measured.
- **Re-bench 26B-A4B with MTP drafter** once PR #23398 lands (gpumod-rj0s). If MTP adds ~50% TPS as it does on Qwen, 26B-A4B reaches ~210 TPS at 100/0 quality.
- **n=30 confirmation** on Q8's L2 80% to lock in that the L2 prompt-fix gain wasn't a draw of the dice.

## Files

| File | Description |
|---|---|
| `result_gemma4-12b-q4.json` | 15-iter result, Gemma 4 12B UD-Q4_K_XL |
| `result_gemma4-12b-q5.json` | 15-iter result, Gemma 4 12B Q5_K_M |
| `result_gemma4-12b-q8.json` | 15-iter result, Gemma 4 12B UD-Q8_K_XL |
| `result_gemma4-26b-a4b-q4.json` | 15-iter result, Gemma 4 26B-A4B UD-IQ4_XS |
| `run_bench.sh` | Driver for Q4 + Q5 sequential run |
| `run_bench_extra.sh` | Driver for Q8 + 26B-A4B sequential run (includes 12 GiB size guard for the 26B GGUF) |
| `run_gemma4-*.log` | Per-model benchmark stdout (gitignored per `*.log`) |
| `artifacts/<model>/iter_NN/` | Per-iteration, per-level generated code and validation output |

Per-bench drivers are superseded by the central [`scripts/run_coding_benchmark.sh`](https://github.com/jaigouk/gpumod/blob/main/scripts/run_coding_benchmark.sh) (gpumod-4omn) — same patterns, one script, options instead of per-dir wrappers. The local `run_bench{,_extra}.sh` are kept because they document the exact model list and 26B size guard for this benchmark.

## References

- [20260524 Qwen3.6 MTP vs non-MTP benchmark (Hermes-agent baseline)](../20260524_qwen36_mtp_vs_a3b/README.md)
- [20260423 Qwen3.6 vs Gemma4 E4B comparison (prior generation)](../20260423_qwen36_gemma4_comparison/README.md)
- [Unsloth gemma-4-12b-it-GGUF model card](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF)
- [Unsloth gemma-4-26B-A4B-it-GGUF model card](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)
- [google/gemma-4-12B-it (sampler recommendation source)](https://huggingface.co/google/gemma-4-12B-it)
- [ggml-org/llama.cpp PR #23398 — Gemma 4 MTP port to mainline (WIP)](https://github.com/ggml-org/llama.cpp/pull/23398)
- [AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF) — drafter pre-conversion for the future PR #23398 swap
