# Gemma 4 26B-A4B — QAT UD-Q4_K_XL vs imatrix UD-IQ4_XS

**Date:** 2026-06-07
**Tickets:** gpumod-rjkx (bench); gpumod-p2gj (CLI exit-code fix landed mid-bench)
**Question:** Does Unsloth's QAT-derived Gemma 4 26B-A4B GGUF preserve or improve on the current Hermes-agent baseline (imatrix UD-IQ4_XS), and is it worth swapping [`modes/hermes-agent.yaml:28`](https://github.com/jaigouk/gpumod/blob/main/modes/hermes-agent.yaml#L28)?

## TL;DR

| Variant | Mean | σ | Min/Max | 95% CI | TPS | VRAM idle | GGUF size | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Gemma 4 26B-A4B QAT UD-Q4_K_XL** | **100.00** | **0.00** | 100/100 | [100.0, 100.0] | **168.9** | 17567 MB | 14.25 GB | **Quality tied, +20% TPS — swap to this** |
| Gemma 4 26B-A4B imatrix UD-IQ4_XS † | 100.00 | 0.00 | 100/100 | [100.0, 100.0] | 140.7 | 15665 MB | 12.7 GB | Hermes baseline (perfect quality, see 20260603) |

† Imatrix row reused from [`docs/benchmarks/20260603_gemma4_12b_vs_qwen36_35b_a3b_mtp/result_gemma4-26b-a4b-q4.json`](../20260603_gemma4_12b_vs_qwen36_35b_a3b_mtp/result_gemma4-26b-a4b-q4.json) (same b9500 binary, same `GEMMA_CODING` sampler, same `--cache-type-k/v q8_0`, same `enable_thinking`, same v2 coding suite). Different host-load window — flag for TPS interpretation (see [Methodology Caveats](#methodology-caveats)).

**Three headlines:**

1. **Quality is preserved exactly.** QAT scores 100/100 on every one of 15 iterations with σ=0 — bit-for-bit the same headline number as the imatrix baseline. All five levels (including the L5 source-inspection composition test) pass 15/15. QAT does what it says on the tin: at the precision it was trained against, accuracy holds.
2. **TPS is +20% (+28 TPS).** QAT 168.9 mean vs imatrix 140.7 — a far larger delta than the σ noise floor (QAT σ=0.73, imatrix σ=0.32, combined ≪ 1 TPS). The real reason is more nuanced than "QAT is faster": Unsloth's QAT GGUF ships as a Q4_K_XL (K-quant, no runtime lookups), while the imatrix variant uses IQ4_XS (i-quant with per-block lookup tables that cost cycles per token). The +20% mostly comes from the quant-format swap; QAT itself "earns" the right to ship at the higher-throughput Q4_K layout without losing accuracy.
3. **The cost is +1.9 GB VRAM and +1.5 GB GGUF.** Steady-state VRAM 17567 vs 15665 MB. Still comfortably inside the 24 GB ceiling — with multimodal mmproj BF16 (~1.1 GB) loaded, peak VRAM during a benchmark iteration was ~18 GB. Six GB of headroom remain for image-token KV growth on multi-image conversations.

See [Recommendation](#recommendation) for the mode-swap decision.

## Setup

| Component | Specification |
|---|---|
| **CPU** | AMD Ryzen 7 5700G (16 threads) |
| **RAM** | 32 GB DDR4 |
| **GPU** | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| **OS** | Ubuntu 24.04.4 LTS |
| **Driver** | NVIDIA 580.65.06 |
| **CUDA** | 12.0 |
| **llama.cpp** | b9500 (`3d1998634`, built 2026-06-04) — same binary as the 20260603 imatrix run, no quant-format changes between |
| **Bench commit** | `157febf` (HEAD at run time) — QAT preset + runner edits + driver pre-fix; CLI exit-code fix landed mid-bench |

VRAM isolation enforced — `gpumod mode switch blank` stopped `vllm-embedding-code` before the QAT model start; nothing else was GPU-resident during the run (see [run_bench.sh](run_bench.sh)).

## Why this benchmark

Unsloth shipped a QAT (Quantization-Aware Training) GGUF for Gemma 4 26B-A4B at [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF) on top of `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`. Per Unsloth's [QAT docs](https://unsloth.ai/docs/models/gemma-4/qat), they ship only one quant tier (UD-Q4_K_XL) because "precisions higher than the uploaded UD-Q4_K_XL version degrade accuracy" — QAT-derived weights are most consistent at the precision the training targeted.

The current Hermes-agent mode ([`modes/hermes-agent.yaml:28`](https://github.com/jaigouk/gpumod/blob/main/modes/hermes-agent.yaml#L28)) uses `gemma4-26b-a4b-q4` (imatrix UD-IQ4_XS, 12.7 GB GGUF). That row in the 20260603 bench scored **100/100 across 15 iterations with σ=0** — a tough baseline to improve on for quality. The question for QAT is:

- Does it match imatrix's perfect quality? (Expected: yes — QAT training preserves accuracy at the target precision.)
- Does it cost more or less TPS? (Unknown — UD-Q4_K_XL is a different K-quant block layout than IQ4_XS.)
- Is the +1.5 GB GGUF size (14.25 vs 12.7 GB) and +1.2 GB VRAM headroom hit acceptable?
- Net: keep imatrix, switch to QAT, or split (e.g. QAT for hermes-agent single-slot, imatrix for code mode's multi-slot variant)?

## Models tested

| ID | Source | Architecture | Quant | File size | Context | Sampler | mmproj |
|---|---|---|---|---:|---:|---|---|
| `gemma4-26b-a4b-qat-q4` | `unsloth/gemma-4-26B-A4B-it-qat-GGUF` | MoE 26B / 4B active | QAT UD-Q4_K_XL | 14.25 GB | 131072 | GEMMA_CODING (temp 1.0, top_p 0.95, top_k 64, RP 1.05) | BF16 (reused) |
| `gemma4-26b-a4b-q4` † | `unsloth/gemma-4-26B-A4B-it-GGUF` | MoE 26B / 4B active | imatrix UD-IQ4_XS | 12.7 GB | 131072 | GEMMA_CODING | BF16 |

† Imatrix row not re-run for this bench. Reused from 20260603.

Both presets ship identical flags: `--parallel 1 --threads 16 --cache-type-k q8_0 --cache-type-v q8_0 --temp 1.0 --top-p 0.95 --top-k 64 --chat-template-kwargs '{"enable_thinking":true}' --mmproj $HOME/bin/gemma-4-26B-A4B-it-mmproj-BF16.gguf`. The QAT preset reuses the imatrix repo's `mmproj-BF16.gguf` (Unsloth ships the same vision encoder in both repos; reusing the existing file keeps the multimodal path byte-identical between arms). Presets:

- [`presets/llm/gemma4-26b-a4b-qat-q4.yaml`](https://github.com/jaigouk/gpumod/blob/main/presets/llm/gemma4-26b-a4b-qat-q4.yaml) — new for this bench, port 7110, `vram_mb: 20500`
- [`presets/llm/gemma4-26b-a4b-q4.yaml`](https://github.com/jaigouk/gpumod/blob/main/presets/llm/gemma4-26b-a4b-q4.yaml) — Hermes baseline, port 7109, `vram_mb: 19500`

## Methodology

The bench harness is the v2 coding suite from `scripts/run_qwen36_benchmark.py`. Every iteration runs five levels:

| Level | Task | Points |
|---|---|---:|
| L1 | Basic queue (add/get, FIFO) | 25 |
| L2 | Retry with exponential backoff | 25 |
| L3 | Priority scheduling | 25 |
| L4 | Find & fix concurrency bug | 15 |
| L5 | Compose Job + RetryPolicy + JobQueue (source-inspection) | 10 |

Validation: `PytestValidator` with 30 s per-level timeout, 900 s per-request client timeout, `max_tokens=32768`. The L5 source-inspection assertion (gpumod-pdtn fix landed 2026-06-04) is the hardest test in the suite — any model that scores 100 across 15 iters on L5 is doing real composition, not pattern-matching.

QAT-specific test additions:

- New entry in [`scripts/run_qwen36_benchmark.py`](https://github.com/jaigouk/gpumod/blob/main/scripts/run_qwen36_benchmark.py) MODELS dict with `sampler=GEMMA_CODING`.
- New CLI choice `gemma4-26b-a4b-qat-q4`.
- Driver script [`run_bench.sh`](run_bench.sh): `mode switch blank` → 15 s quiesce wait → start QAT service → wait for `/health` → 15 iters → stop. The 15 s quiesce wait is a hard-won learning from this bench's first failed launch (see [Lessons](#lessons-learned-mid-bench)).

## Results

### Summary

| Variant | Mean | σ | Min/Max | 95% CI | TPS mean | TPS min/max | TPS σ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **QAT UD-Q4_K_XL** | **100.00** | **0.00** | 100/100 | [100.0, 100.0] | **168.88** | 166.8 / 169.8 | 0.73 |
| imatrix IQ4_XS † | 100.00 | 0.00 | 100/100 | [100.0, 100.0] | 140.72 | 140.0 / 141.2 | 0.32 |
| **Δ (QAT − imatrix)** | 0.00 | 0.00 | — | — | **+28.16 (+20.0 %)** | — | — |

### Per-level pass rates

| Level | Task | QAT UD-Q4_K_XL | imatrix IQ4_XS |
|---|---|---:|---:|
| L1 | Basic queue | 100% (15/15) | 100% (15/15) |
| L2 | Retry with backoff | 100% (15/15) | 100% (15/15) |
| L3 | Priority scheduling | 100% (15/15) | 100% (15/15) |
| L4 | Concurrency bug fix | 100% (15/15) | 100% (15/15) |
| L5 | Compose Job + RetryPolicy + JobQueue | 100% (15/15) | 100% (15/15) |

### Per-iteration scores

| Variant | Scores (15 iters) |
|---|---|
| QAT UD-Q4_K_XL | **100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100** |
| imatrix IQ4_XS † | **100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100** |

### Per-iteration TPS

| Variant | TPS (15 iters) |
|---|---|
| QAT UD-Q4_K_XL | 168.5, 168.5, 168.7, 168.4, 169.4, 169.1, 169.8, 168.5, 168.9, 169.0, 169.3, 169.8, 169.4, 169.4, 166.8 |
| imatrix IQ4_XS † | 141.0, 141.2, 140.8, 140.8, 140.8, 140.9, 140.9, 140.7, 140.9, 140.0, 140.2, 140.4, 140.7, 140.6, 140.9 |

### VRAM and load characteristics

| Variant | GGUF on disk | mmproj | VRAM idle (steady-state) | TTFT warm |
|---|---:|---:|---:|---:|
| QAT UD-Q4_K_XL | 14.25 GB | 1.13 GB (BF16, shared) | 17567 MB | 52 ms |
| imatrix IQ4_XS † | 12.7 GB | 1.13 GB (BF16) | 15665 MB | 53 ms |
| **Δ** | **+1.55 GB** | 0 | **+1902 MB (+12 %)** | **−1 ms (noise)** |

Steady-state VRAM is reported after the first iteration; iter-1 was identical (within 28 MB). Iter-1 TTFT on QAT was 73 ms (warm-cache miss on first request); imatrix iter-1 TTFT was 82 ms. Both arms have a flat-line VRAM curve — no growth across 15 iters, no leak signal. Total bench duration (QAT, 15 iters): **27.6 min** (started 2026-06-06T22:34:35Z, completed 2026-06-06T23:02:10Z).

## Methodology Caveats

- **Imatrix baseline reused, not re-run.** TPS comparisons must allow for the fact that the imatrix run happened on 2026-06-04 under a (potentially) different host-load profile. Same b9500 binary and same `GEMMA_CODING` sampler, but I/O / CPU contention could differ. Quality (pass-rate, score) is host-load-insensitive and fully comparable. If the TPS delta is small (< 5%), assume noise; if it's large, the cause is more likely the quant pipeline than the host.
- **Sampler is GEMMA_CODING per Google's model card.** Temp 1.0 / top_p 0.95 / top_k 64 / repetition_penalty 1.05. Both arms use the same values.
- **`enable_thinking=true` is enabled on both arms.** Gemma's chat template routes the reasoning portion of the response to a `reasoning_content` field on the OpenAI-compat response; `content` holds the post-thinking answer. The benchmark's `LlamaCppClient` ([`src/gpumod/benchmarks/coding/llm_client.py:93`](https://github.com/jaigouk/gpumod/blob/main/src/gpumod/benchmarks/coding/llm_client.py#L93)) reads `content` (not `reasoning_content`), so the score reflects the post-thinking answer only.
- **MTP is not in play for either arm.** Mainline llama.cpp [PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398) (Gemma 4 MTP) is still WIP. Both QAT and imatrix run at native non-speculative speed. Track gpumod-rj0s.
- **Single run, 15 iters.** Enough for tight mean/σ separation. The imatrix arm has σ=0 (no spread to detect), so any QAT std > 0 immediately signals a quant-pipeline accuracy difference (not noise).

## Lessons learned mid-bench

The first bench launch failed — the gpumod CLI silently exited 0 on a Lifecycle error, so the `set -e` harness didn't catch a quiesce-window collision. Root cause and fix were extracted into a separate ticket and shipped before the re-launch:

- **gpumod-p2gj** — `error_handler` ([`src/gpumod/cli.py:296-304`](https://github.com/jaigouk/gpumod/blob/main/src/gpumod/cli.py#L296-L304)) now raises `typer.Exit(code=1)` after printing the error, so shell automation (`set -e`, `$?`, pipefail) sees a non-zero exit code. New test [`tests/unit/test_cli_error_handler.py`](https://github.com/jaigouk/gpumod/blob/main/tests/unit/test_cli_error_handler.py); 16 existing tests across 5 files updated from `exit_code == 0  # error_handler catches it` to `exit_code != 0  # gpumod-p2gj: …`. All four gates pass (2381 pytest, mypy --strict, ruff, format).

The bench's [run_bench.sh](run_bench.sh) also added a 15 s quiesce wait after `mode switch blank` (the mode switch stops the embedding server; that stop is itself heavy-GPU-eligible and trips the start-side quiesce gate on the QAT model).

## Recommendation

**Swap `modes/hermes-agent.yaml:28` from `gemma4-26b-a4b-q4` to `gemma4-26b-a4b-qat-q4`.**

The decision criteria from gpumod-rjkx were: "keep imatrix, switch to QAT, or split". QAT wins on the only axes that matter:

| Criterion | imatrix IQ4_XS | QAT UD-Q4_K_XL | Verdict |
|---|---|---|---|
| Score mean | 100.00 | 100.00 | tied |
| Score σ | 0.00 | 0.00 | tied |
| L5 composition pass-rate | 15/15 | 15/15 | tied |
| TPS mean | 140.7 | **168.9** | **+28.2 (+20 %)** |
| VRAM idle | 15.3 GB | 17.2 GB | imatrix wins by 1.9 GB |
| GGUF on disk | 12.7 GB | 14.25 GB | imatrix wins by 1.55 GB |
| Multimodal path | mmproj BF16 OK | mmproj BF16 OK (reused) | tied |
| Sampler / context / flags | identical | identical | tied |
| llama.cpp binary | b9500 | b9500 | tied |

VRAM cost is comfortably under the 24 GB ceiling: peak ~18 GB with mmproj leaves >6 GB headroom for image-KV growth, which is enough for any realistic Hermes-agent multimodal conversation. The +1.55 GB disk hit is irrelevant on a 5.3 TB pool.

The +20 % TPS is delivered to every Hermes-agent request — a real, persistent latency improvement for the operator's primary daily-driver model. No regressions in any measured dimension.

**Code mode follow-up.** This bench only covers the single-slot multimodal hermes-agent config. The multi-slot code-mode preset (`gemma4-26b-a4b-q4-multi`, port 7109, `--parallel 3 --cont-batching`) would need its own QAT mirror plus an `install-all` re-render to verify the change. Not in scope here; file a follow-up if the operator wants it after a few days of hermes-agent QAT runtime confirms no regressions.

**MTP follow-up unchanged.** gpumod-rj0s still tracks PR #23398. Once Gemma 4 MTP lands, an `assistant`-arch drafter pairs with whichever quant variant is in production at that point. QAT and MTP are orthogonal.

## Files

- [`run_bench.sh`](run_bench.sh) — driver (QAT only)
- [`result_gemma4-26b-a4b-qat-q4.json`](result_gemma4-26b-a4b-qat-q4.json) — QAT raw results
- `run_gemma4-26b-a4b-qat-q4.log` — QAT run log (gitignored per `*.log`; local-only)
- [`artifacts/`](artifacts/) — per-iteration prompt + response + extracted code
- Imatrix baseline: [`../20260603_gemma4_12b_vs_qwen36_35b_a3b_mtp/result_gemma4-26b-a4b-q4.json`](../20260603_gemma4_12b_vs_qwen36_35b_a3b_mtp/result_gemma4-26b-a4b-q4.json)

## Related

- gpumod-rjkx — this bench
- gpumod-p2gj — CLI exit-code fix (landed mid-bench)
- gpumod-rj0s — Gemma 4 MTP tracking (upstream PR #23398, future TPS boost)
- gpumod-h6gs — predecessor bench (Gemma 4 12B/26B-A4B vs Qwen3.6-35B-A3B-MTP); imatrix baseline lives here
- [Unsloth Gemma 4 QAT docs](https://unsloth.ai/docs/models/gemma-4/qat)
- [`modes/hermes-agent.yaml:28`](https://github.com/jaigouk/gpumod/blob/main/modes/hermes-agent.yaml#L28) — current single-slot LLM selector (would change if QAT wins)
