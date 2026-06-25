# Qwen-AgentWorld-35B-A3B vs MoE peers — v2 coding suite

**Date:** 2026-06-25
**Tickets:** gpumod-qsgl (epic), gpumod-qsgl.3 (this bench); follows gpumod-qsgl.2 (load-test) + gpumod-qsgl.4 (runner wiring + Q2c)
**Question:** Now that AgentWorld loads + generates in gpumod (via the per-preset `--override-kv` conversion-defect fix), how does it compare on the coding suite to the 35B-A3B MoE peers and the Gemma 4 hermes-agent baseline — and should it earn a place in any mode?

## TL;DR

| Variant | Mean | σ | Min/Max | 95% CI | TPS | VRAM idle | GGUF | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Qwen-AgentWorld-35B-A3B Q4_K_M** | **50.00** | 15.81 | 40/90 | [41.3, 58.7] | 188.9 | 21387 MB | 19.7 GB | **Not a coding model — do not adopt for code/agent modes** |
| Qwen3.6-35B-A3B UD-Q4_K_S | 98.67 | 3.52 | 90/100 | [96.7, 100.6] | 172.4 | 20699 MB | ~20 GB | Strong all-rounder |
| Qwen3.6-35B-A3B MTP UD-IQ4_XS | 96.67 | 8.80 | 75/100 | [91.8, 101.5] | **227.8** | 19778 MB | ~17 GB | Throughput winner (MTP) |
| Gemma 4 26B-A4B QAT UD-Q4_K_XL | **100.00** | **0.00** | 100/100 | [100, 100] | 169.8 | 19565 MB | 14.25 GB | Quality winner (hermes-agent baseline) |

**Three headlines:**

1. **The gpumod integration is validated end-to-end.** AgentWorld ran 15 full iterations through the preset/systemd path across a **2 h 37 m** four-arm run (00:12 → 02:49) with **zero crashes**. The `--override-kv` conversion-defect fix (`block_count=40`, `nextn_predict_layers=0`) held in production for the entire run. This is the real deliverable: a `qwen35moe`/MTP model that ships broken GGUFs is now a first-class, repeatable gpumod service.
2. **AgentWorld is not a coding model.** It scores **50.00 (σ 15.81)** versus the peers' 96.67–100. The gap is **accuracy, not speed** — at 188.9 TPS it is *faster* than both Gemma (169.8) and non-MTP Qwen3.6 (172.4). This matches its design: AgentWorld is a "native language **world model**" for environment simulation, not a code/instruct model.
3. **Its failures are structured, and partly a prompt-alignment artifact.** AgentWorld passes the harder L3 (priority queue) and L4 (concurrency bug-fix) **15/15** but is flaky on the *easier* L1/L2 and never passes L5. The L1/L2 flakiness is largely an **interface-naming mismatch on an underspecified prompt** (see [Why AgentWorld scores 50](#why-agentworld-scores-50)), so "50/100" overstates a raw coding deficit — but the peers align with that implicit convention far more reliably, so a real gap remains.

## Setup

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 4090, 24 GB (24564 MB) |
| Host RAM | ~30 GiB usable (32 GB DDR4) |
| **llama.cpp** | **version 9784 (`8be759e6f`)**, `git describe` b9782-2-g8be759e6f, built 2026-06-24 (gpumod-qsgl.1) — **all four arms on this one binary** |
| Bench commit | HEAD at run time (runner: AgentWorld + SIQ entries, gpumod-qsgl.4) |
| Stability defaults | `GGML_CUDA_NO_PINNED=1` (template default); preflight RAM/VRAM checks |
| Isolation | `gpumod mode switch blank` before the run; one arm GPU-resident at a time (see [run_bench.sh](run_bench.sh)) |

Every arm was **re-run on b9784** (not reused from older benches) — the prior `gemma4-26b-a4b-qat-q4` JSON (20260606, b9500) is *not* reused here, so the comparison is binary-clean.

## Models tested

| ID | Source | Architecture | Quant | GGUF | Sampler | Port |
|---|---|---|---|---:|---|---:|
| `agentworld-35b-a3b-q4` | `gaoqianshen/Qwen-AgentWorld-35B-A3B-Q4_K_M-GGUF` | qwen35moe hybrid Gated-DeltaNet + MTP, 256 exp | Q4_K_M | 19.7 GB | THINKING_CODING | 7111 |
| `qwen36-35b-a3b` | `unsloth/Qwen3.6-35B-A3B-GGUF` | MoE 35B / A3B | UD-Q4_K_S | ~20 GB | THINKING_CODING | 7101 |
| `qwen36-35b-a3b-mtp-iq4xs` | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` | MoE 35B / A3B + MTP | UD-IQ4_XS | ~17 GB | THINKING_CODING | 7103 |
| `gemma4-26b-a4b-qat-q4` | `unsloth/gemma-4-26B-A4B-it-qat-GGUF` | MoE 26B / A4B | QAT UD-Q4_K_XL | 14.25 GB | GEMMA_CODING | 7110 |

AgentWorld must be started via its **preset** so the two `--override-kv` flags apply; a bare `llama-server --base-url` would omit them and fail to load. See [docs/research/20260624_agentworld_qwen35moe_mtp/FINDINGS.md](../../research/20260624_agentworld_qwen35moe_mtp/FINDINGS.md).

## Methodology

v2 coding suite (`scripts/run_qwen36_benchmark.py`), 15 iterations/arm, 5 levels:

| Level | Task | Points |
|---|---|---:|
| L1 | Basic queue (add/get, FIFO) | 25 |
| L2 | Retry with exponential backoff | 25 |
| L3 | Priority scheduling | 25 |
| L4 | Find & fix concurrency bug | 15 |
| L5 | Compose Job + RetryPolicy + JobQueue (source-inspection) | 10 |

`PytestValidator`, 30 s/level timeout, 900 s client timeout, `max_tokens=32768`. Scoring reads the `content` field (post-thinking answer); reasoning traces in `reasoning_content` are not scored. Each arm started via `gpumod service start`, polled `/health`, then 15 iters, then stopped + 15 s quiesce.

## Results

### Summary

| Variant | Mean | σ | Min/Max | 95% CI | TPS mean | TPS σ | VRAM idle | TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AgentWorld Q4_K_M | 50.00 | 15.81 | 40/90 | [41.3, 58.7] | 188.9 | 0.76 | 21387 MB | 84 ms |
| Qwen3.6-35B-A3B Q4_K_S | 98.67 | 3.52 | 90/100 | [96.7, 100.6] | 172.4 | 0.67 | 20699 MB | 85 ms |
| Qwen3.6-35B-A3B MTP IQ4_XS | 96.67 | 8.80 | 75/100 | [91.8, 101.5] | 227.8 | 2.90 | 19778 MB | 98 ms |
| Gemma 4 26B-A4B QAT Q4_K_XL | 100.00 | 0.00 | 100/100 | [100, 100] | 169.8 | 0.64 | 19565 MB | 57 ms |

### Per-level pass rate (of 15)

| Level | AgentWorld | Qwen3.6-35B-A3B | Qwen3.6 MTP | Gemma 4 QAT |
|---|---:|---:|---:|---:|
| L1 Basic Queue | **4/15** | 15/15 | 14/15 | 15/15 |
| L2 Retry w/ Backoff | **2/15** | 15/15 | 14/15 | 15/15 |
| L3 Priority Queue | 15/15 | 15/15 | 15/15 | 15/15 |
| L4 Concurrency Bug Fix | 15/15 | 15/15 | 15/15 | 15/15 |
| L5 Compose (hardest) | **0/15** | 13/15 | 15/15 | 15/15 |

### Per-iteration scores (15 iters)

| Variant | Scores |
|---|---|
| AgentWorld | 40, 65, 65, 40, 40, 40, 40, 40, 40, 40, 90, 40, 65, 40, 65 |
| Qwen3.6-35B-A3B | 100, 90, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 90, 100, 100 |
| Qwen3.6 MTP | 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 75, 75, 100, 100, 100 |
| Gemma 4 QAT | 100 × 15 |

AgentWorld's floor of **40** = L3 (25) + L4 (15) every iteration; the spread to 65/90 is whether L1 and/or L2 happened to land. It never scored L5.

## Why AgentWorld scores 50

The per-level pattern is counterintuitive — acing the *harder* L3/L4 while flunking the *easier* L1/L2. Inspecting the artifacts ([artifacts/agentworld-35b-a3b-q4/](artifacts/agentworld-35b-a3b-q4/)) explains it:

- **L1/L2 are underspecified, and AgentWorld guesses the missing method name inconsistently.** The L1 prompt requires `add_job(job_id, data)` and `get_result(job_id)` but **never names the method that sets a completed job's result**. The hidden test calls `set_result(...)`. AgentWorld's reasoning explicitly flags the ambiguity ("there's no `process_job` or `complete_job` method specified!") and then guesses — `complete_job` in the 11 failing iters, `set_result` in the 4 passing ones. The code is otherwise correct (FIFO `deque`, right signatures); only the result-setter name differs. The Qwen3.6/Gemma peers converge on `set_result` ~14–15/15, so they align with the implicit convention far more reliably. **Conclusion:** ~half of AgentWorld's deficit is prompt-alignment, not raw coding inability — but the alignment gap vs the peers is real and consistent.
- **L5 (composition) is a genuine miss: 0/15.** The hardest level (compose `Job` + `RetryPolicy` + `JobQueue`, source-inspected) never passed. This is a real capability gap, consistent with the peers' near-perfect L5.
- **Net read:** AgentWorld can write correct standalone data structures (L3/L4 = 15/15) but is weakly aligned with instruct-style interface conventions and cannot do the multi-component composition the peers handle. That is exactly what you'd expect from an **environment-simulation world model** rather than a code/instruct model.

## Methodology caveats

- **AgentWorld is off-domain.** It is a "native language world model" for long-horizon environment simulation; a coding suite measures something it was not built for. The 50/100 quantifies "don't use it for code," not "the model is bad."
- **L1/L2 ambiguity inflates the gap.** As above, the result-setter method name is unspecified; a stricter prompt would likely lift AgentWorld's L1/L2 and narrow (not erase) the gap. The peers were measured on the identical prompt, so the *comparison* is fair even though the absolute number is prompt-sensitive.
- **VRAM is `vram_idle_mb` (steady-state), not a true peak.** The runner did not populate `vram_peak_mb` this run; values are the per-iteration idle reading. AgentWorld's 21387 MB matches the Q2c measurement at `context_size=65536`.
- **Samplers differ by family (as designed).** The three Qwen arms use THINKING_CODING (temp 0.6 / top_k 20); Gemma uses GEMMA_CODING (temp 1.0 / top_k 64 / rep 1.05) per Google's card. Cross-family TPS/quality comparison carries the usual sampler caveat.
- **Single run, 15 iters/arm.** Enough for tight CI separation; Gemma σ=0 again.

## Recommendation

**Do not adopt AgentWorld for any code or agent mode.** On the suite it is dominated by every peer (50 vs 96.67–100), and its strengths (world-modeling, long-horizon simulation) are not what hermes-agent / code / multi-agent modes need. Keep `gemma4-26b-a4b-qat-q4` as the hermes-agent quality baseline (100/100 σ0) and `qwen36-35b-a3b-mtp-iq4xs` as the throughput option (227.8 TPS).

What this bench *does* earn AgentWorld: a **validated, repeatable gpumod integration**. If a future task needs a world-model (environment simulation, trajectory rollout), the preset + override-kv fix is proven and the service path is solid. File a separate ticket if/when such a use case appears; it should be benched on a simulation-appropriate eval, not this coding suite.

## Files

- [run_bench.sh](run_bench.sh) — 4-arm driver (all arms on b9784)
- [result_agentworld-35b-a3b-q4.json](result_agentworld-35b-a3b-q4.json), [result_qwen36-35b-a3b.json](result_qwen36-35b-a3b.json), [result_qwen36-35b-a3b-mtp-iq4xs.json](result_qwen36-35b-a3b-mtp-iq4xs.json), [result_gemma4-26b-a4b-qat-q4.json](result_gemma4-26b-a4b-qat-q4.json)
- [artifacts/](artifacts/) — per-iteration prompt + response + extracted code
- `run_*.log` — per-arm run logs (gitignored)

## Related

- gpumod-qsgl.2 — load-test spike (the override-kv conversion-defect fix)
- gpumod-qsgl.4 — runner wiring + service-path validation + Q2c (context 65536)
- [docs/research/20260624_agentworld_qwen35moe_mtp/FINDINGS.md](../../research/20260624_agentworld_qwen35moe_mtp/FINDINGS.md) — architecture + defect class + fix
- [docs/benchmarks/20260606_gemma4_26b_qat_vs_imatrix/README.md](../20260606_gemma4_26b_qat_vs_imatrix/README.md) — methodology + format reference; prior Gemma QAT baseline
