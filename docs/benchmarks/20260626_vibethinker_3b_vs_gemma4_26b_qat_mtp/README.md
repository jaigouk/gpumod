# VibeThinker-3B Q8_0 vs Gemma 4 26B-A4B QAT-MTP — coding suite (L1–L5)

**Date:** 2026-06-26
**Ticket:** gpumod-msy8
**Question:** Is a tiny (3 B) dense reasoning-tuned model competitive with the
26 B-A4B MoE daily-driver (`gemma4-26b-a4b-qat-mtp-q4`) on the v2 coding suite —
and if not, where does a model like VibeThinker actually fit?

## TL;DR

| Model | Coding mean | σ | Min/Max | 95% CI | TPS | VRAM idle | GGUF | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **gemma4-26b-a4b-qat-mtp-q4** | **99.3** | 2.58 | 90/100 | [97.9, 100.8] | **233.2** | 21323 MB | 13.3 GB | **Daily-driver stays — coding crown intact** |
| VibeThinker-3B Q8_0 | 74.3 | 15.45 | 40/100 | [65.8, 82.9] | 194.5 | **4971 MB** | **3.29 GB** | **Not a coding/agent replacement; capable for its size, but high-variance and not faster** |

**Three headlines:**

1. **The 3 B trails the 26 B by 25 points — and is not even faster.** VibeThinker
   scores **74.3** vs the daily-driver's **99.3**. The intuitive trade ("smaller
   model, lower quality, but much faster") only half-holds: it loses 25 quality
   points **and** runs *slower* in raw TPS (194.5 vs 233.2, **−16.6 %**). The
   26 B wins throughput because it ships with an MTP speculative drafter and
   only activates ~4 B params/token, while the 3 B dense pays full forward
   passes and emits a very long `<think>` trace per answer. The only axes
   VibeThinker wins are footprint: **4.3× less VRAM** (5.0 vs 21.3 GB) and
   **4× smaller on disk** (3.29 vs 13.3 GB).
2. **Variance is the real story, not the mean.** VibeThinker's σ is **15.45**
   (min 40, max 100) against Gemma's **2.58** (min 90, max 100) — **6× the
   spread**. At temp 1.0 a 3 B reasoning model is wildly inconsistent: it scored
   a perfect **100** on iter 13 and a **40** on iter 4 on the *same* five tasks.
   You cannot rely on any single response. The 95 % CIs do not overlap
   ([65.8, 82.9] vs [97.9, 100.8]), so the quality gap is real, not noise.
3. **It fails exactly where the card says it would.** L1/L3/L4 are strong
   (15/15, 14/15, 14/15); the model collapses on **L2 retry-with-backoff
   (6/15)** and **L5 composition (3/15)** — multi-class, multi-step structure.
   VibeThinker's card states it was tuned for *verifiable single-answer
   reasoning* (competition math/coding) and **"was not trained on tool-calling
   or agent-based programming data"**
   ([model card](https://huggingface.co/WeiboAI/VibeThinker-3B)). The L2/L5
   failures are that limitation showing up: it solves self-contained problems,
   not composed systems.

See [Recommendation](#recommendation).

## Setup

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB |
| Host RAM | 31 GiB |
| Driver / CUDA | 580.65.06 / 13.0 |
| **llama.cpp** | **version 9784 (`8be759e6f`)** — both arms on this one binary |
| Stability defaults | `GGML_CUDA_NO_PINNED=1` (template default); preflight RAM/VRAM checks |
| Isolation | `gpumod mode switch blank` before the run; one arm GPU-resident at a time (see [run_bench.sh](run_bench.sh)) |

VRAM isolation enforced — `mode switch blank` + 15 s quiesce before the
VibeThinker service start; nothing else was GPU-resident (GPU idle at 15 MiB
before/after). VibeThinker runs on port **7115**, not the daily-driver port
7110, so no hermes-agent orchestrator client could reconnect to the
model-under-test mid-run.

## Why this benchmark

VibeThinker-3B ([WeiboAI/VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B))
is a dense **Qwen2** 3 B model (lineage Qwen2.5-3B → Qwen2.5-Coder-3B →
VibeThinker-3B; verified via `fetch_model_config`: `architectures=[Qwen2ForCausalLM]`,
`is_moe=false`, `context_length=131072`). It is fine-tuned for *verifiable
reasoning* — competition-style math, coding, and STEM — and the operator
downloaded the Q8_0 GGUF from
[prithivMLmods/VibeThinker-3B-GGUF](https://huggingface.co/prithivMLmods/VibeThinker-3B-GGUF)
(3.29 GB, near-lossless).

The comparison target is the current hermes-agent daily-driver,
`gemma4-26b-a4b-qat-mtp-q4` (MoE 26 B / 4 B-active + MTP speculative drafter).
This is **not** a same-model quant A/B (as the
[20260606 QAT-vs-imatrix bench](../20260606_gemma4_26b_qat_vs_imatrix/README.md)
was). It is an **architecture/size A/B**: a tiny dense reasoning model vs the
production MoE. Q8_0 was chosen for VibeThinker to keep quantization near-lossless
so any quality gap reflects architecture and scale, not the quant.

## Models tested

| ID | Source | Architecture | Quant | GGUF | Context | Sampler | Port |
|---|---|---|---|---:|---:|---|---:|
| `vibethinker-3b-q8` | prithivMLmods/VibeThinker-3B-GGUF | dense Qwen2 3 B | Q8_0 | 3.29 GB | 65536 | VIBETHINKER_CODING (temp 1.0, top_p 0.95, top_k 20) | 7115 |
| `gemma4-26b-a4b-qat-mtp-q4` † | unsloth/gemma-4-26B-A4B-it-qat-GGUF | MoE 26 B / A4B + MTP | QAT UD-Q4_K_XL | 13.3 GB | 262144 | GEMMA_CODING (temp 1.0, top_p 0.95, top_k 64, RP 1.05) | 7110 |

† **Gemma arm reused, not re-run.** Result taken from
[`../20260625_unified/result_gemma4-26b-a4b-qat-mtp-q4.json`](../20260625_unified/result_gemma4-26b-a4b-qat-mtp-q4.json)
(copied into this folder as `result_gemma4-26b-a4b-qat-mtp-q4.json`): **same
llama.cpp binary 9784 (`8be759e6f`)**, same v2 coding suite (L1–L5), same
GEMMA_CODING sampler, same 15 iterations, run 2026-06-25 (one day prior). Quality
(score/pass-rate) is host-load-insensitive and fully comparable; the 1-day
host-load window is flagged for TPS only (see [Caveats](#methodology-caveats)).

Presets:

- [`presets/llm/vibethinker-3b-q8.yaml`](https://github.com/jaigouk/gpumod/blob/main/presets/llm/vibethinker-3b-q8.yaml) — new for this bench, port 7115, `vram_mb: 6000`
- [`presets/llm/gemma4-26b-a4b-qat-mtp-q4.yaml`](https://github.com/jaigouk/gpumod/blob/main/presets/llm/gemma4-26b-a4b-qat-mtp-q4.yaml) — daily-driver, port 7110

## Methodology

The harness is the v2 coding suite from
[`scripts/run_qwen36_benchmark.py`](https://github.com/jaigouk/gpumod/blob/main/scripts/run_qwen36_benchmark.py).
Every iteration runs five levels:

| Level | Task | Points |
|---|---|---:|
| L1 | Basic queue (add/get, FIFO) | 25 |
| L2 | Retry with exponential backoff | 25 |
| L3 | Priority scheduling | 25 |
| L4 | Find & fix concurrency bug | 15 |
| L5 | Compose Job + RetryPolicy + JobQueue (source-inspection) | 10 |

Validation: `PytestValidator`, 30 s per-level timeout, 900 s per-request client
timeout, `max_tokens=32768`. 15 iterations per arm. Both arms use
`--cache-type-k q8_0 --cache-type-v q8_0` for KV-cache fairness.

VibeThinker-specific additions:

- New `VIBETHINKER_CODING` sampler in
  [`sampler_config.py`](https://github.com/jaigouk/gpumod/blob/main/src/gpumod/benchmarks/coding/sampler_config.py)
  (temp 1.0, top_p 0.95, top_k 20).
- New MODELS entry + CLI choice `vibethinker-3b-q8`.
- A `<think>`-stripping step in the extraction path (see
  [Caveats](#methodology-caveats) — this was required for a *fair* score).

## Results

### Summary

| Variant | Mean | σ | Min/Max | 95% CI | TPS mean | TPS min/max | TPS σ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **gemma4-26b-a4b-qat-mtp-q4** † | **99.33** | 2.58 | 90/100 | [97.9, 100.8] | **233.22** | — | — |
| VibeThinker-3B Q8_0 | 74.33 | 15.45 | 40/100 | [65.8, 82.9] | 194.50 | 191.3 / 196.1 | 1.21 |
| **Δ (VibeThinker − Gemma)** | **−25.00** | +12.87 | — | — | **−38.7 (−16.6 %)** | — | — |

### Per-level pass rates

| Level | Task | gemma4-26b-a4b-qat-mtp-q4 † | VibeThinker-3B Q8_0 |
|---|---|---:|---:|
| L1 | Basic queue | 100% (15/15) | 100% (15/15) |
| L2 | Retry with backoff | 100% (15/15) | **40% (6/15)** |
| L3 | Priority scheduling | 100% (15/15) | 93% (14/15) |
| L4 | Concurrency bug fix | 100% (15/15) | 93% (14/15) |
| L5 | Compose Job + RetryPolicy + JobQueue | 93% (14/15) | **20% (3/15)** |

### Per-iteration scores

| Variant | Scores (15 iters) |
|---|---|
| gemma4-26b-a4b-qat-mtp-q4 † | 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, **90**, 100, 100 |
| VibeThinker-3B Q8_0 | 75, 75, 90, **40**, 65, 65, 65, 90, 75, 65, 65, 65, **100**, 90, 90 |

### VRAM and footprint

| Variant | GGUF on disk | VRAM idle (steady-state) | TPS | Context |
|---|---:|---:|---:|---:|
| gemma4-26b-a4b-qat-mtp-q4 † | 13.3 GB | 21323 MB | 233.2 | 262144 |
| VibeThinker-3B Q8_0 | 3.29 GB | 4971 MB | 194.5 | 65536 |
| **Δ** | **−10.0 GB** | **−16352 MB (−4.3×)** | −38.7 | — |

VibeThinker run: 15 iters in **~33 min** (2026-06-25T22:20:36Z →
22:53:26Z); ~2.2 min/iter wall-clock. The Gemma arm ran 15 iters in ~21 min
(~1.4 min/iter) — the small model is also slower *end-to-end* because it emits a
large `<think>` trace before every answer.

## Methodology Caveats

- **`<think>` extraction — required for a fair score (engineering finding).**
  VibeThinker emits `<think>…</think>` reasoning into `message.content`, and it
  frequently drafts ` ```python ` fences *inside* the `<think>` block. The
  bench's `extract_code` is first-fence-wins, so without intervention it would
  validate the model's **draft** instead of the post-`</think>` final answer.
  `--reasoning-format deepseek` does **not** help: VibeThinker's plain Qwen2
  ChatML template is a "content-only" chat format in llama.cpp, so the reasoning
  parser never engages (verified 2026-06-26 — `reasoning_content` stays empty).
  The fix is runner-side: `_strip_think_blocks` removes closed `<think>` spans
  before extraction. Audited: the validated code is always clean final code; the
  L2/L5 failures are **genuine model errors** (e.g. `def process_job(... Callable)[None] -> bool:`
  stray-syntax; `from dataclass import dataclass` / `@Dataclass` wrong import
  names) — not extraction artifacts.
- **Sampler top_k diverges from the card.** The VibeThinker card recommends
  temp 1.0 / top_p 0.95 / **top_k −1 (disabled)**. We use **top_k 20** (the
  project coding default) to bound the candidate set against the degeneration a
  temp-1.0 reasoning model can fall into. Temp and top_p match the card. The
  observed σ 15.45 is high; a top_k −1 arm might differ, but is unlikely to
  close a 25-point gap.
- **Tool-calling caveat.** The card explicitly states VibeThinker "was not
  trained on tool-calling or agent-based programming data." The v2 coding suite
  is **single-turn codegen scored by pytest** — not tool-calling — so the
  comparison is legitimate. But it also means VibeThinker is structurally
  unsuited to the hermes-agent *orchestrator* role, independent of this score.
- **Architecture/size A/B, not same-model.** Unlike the 20260606 QAT-vs-imatrix
  bench, the two arms differ in size (3 B vs 26 B), architecture (dense vs MoE),
  family (Qwen2 vs Gemma), and decoding (plain vs MTP-speculative). Read the
  result as "small reasoning model vs production MoE," not as an isolated
  single-variable test.
- **Gemma baseline reused (1-day-old, same binary).** Quality is fully
  comparable; TPS carries a 1-day host-load caveat (same binary 9784, so the
  caveat is small). VibeThinker's TPS was measured fresh in this window.

## Recommendation

**Keep `gemma4-26b-a4b-qat-mtp-q4` as the hermes-agent daily-driver. Do not adopt
VibeThinker-3B as a coding or agent model.**

| Criterion | Gemma 26B-A4B-MTP | VibeThinker-3B | Winner |
|---|---|---|---|
| Coding mean | 99.3 | 74.3 | Gemma (+25) |
| Consistency (σ) | 2.58 | 15.45 | Gemma (6× tighter) |
| L5 composition | 14/15 | 3/15 | Gemma |
| TPS | 233.2 | 194.5 | Gemma (+16.6 %) |
| End-to-end latency/iter | ~1.4 min | ~2.2 min | Gemma |
| VRAM idle | 21.3 GB | **5.0 GB** | VibeThinker (4.3×) |
| GGUF on disk | 13.3 GB | **3.29 GB** | VibeThinker (4×) |
| Tool-calling / agent fit | yes (daily-driver) | **no (per card)** | Gemma |

VibeThinker is genuinely capable *for a 3 B* — it reliably solves the
self-contained levels (L1/L3/L4) and even hit a clean 100 once. But against the
production MoE it loses on every axis that matters for the daily-driver role
(quality, consistency, throughput, composition, tool-calling) and wins only on
footprint.

**Where a model like this *could* fit:** a low-VRAM (5 GB) engine for
*self-contained, verifiable, single-shot* reasoning tasks where you can afford
to sample-and-verify (its competition-math/coding sweet spot), the 26 B is
overkill, and there is no tool-calling/composition requirement. That is a
different niche from hermes-agent. It is **not** a drop-in for the orchestrator,
and the high per-response variance means any production use must wrap it in a
verifier or best-of-N, never trust a single generation.

## Files

- [`run_bench.sh`](run_bench.sh) — driver (VibeThinker arm; Gemma reused)
- [`result_vibethinker-3b-q8.json`](result_vibethinker-3b-q8.json) — VibeThinker raw results
- [`result_gemma4-26b-a4b-qat-mtp-q4.json`](result_gemma4-26b-a4b-qat-mtp-q4.json) — reused Gemma baseline (from 20260625_unified)
- [`artifacts/`](artifacts/) — per-iteration prompts + responses + extracted code (`<think>`-stripped to match validation)
- `run_vibethinker-3b-q8.log`, `run_bench_console.log` — run logs (gitignored per `*.log`; local-only)

## Related

- gpumod-msy8 — this bench
- [20260606 Gemma 4 QAT vs imatrix](../20260606_gemma4_26b_qat_vs_imatrix/README.md) — same-model quant A/B (contrast)
- [20260625 unified](../20260625_unified/README.md) — source of the reused Gemma MTP baseline
- [VibeThinker-3B model card](https://huggingface.co/WeiboAI/VibeThinker-3B)
- [prithivMLmods/VibeThinker-3B-GGUF](https://huggingface.co/prithivMLmods/VibeThinker-3B-GGUF) — Q8_0 source
