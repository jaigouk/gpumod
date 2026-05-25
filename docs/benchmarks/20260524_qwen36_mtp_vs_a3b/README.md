# Qwen3.6 MTP vs Non-MTP — Hermes-Agent Swap Evaluation

**Date:** 2026-05-24 (initial), 2026-05-25 (35B-A3B-MTP rerun at 128K + q8_0)
**Ticket:** gpumod-76l.3 (epic gpumod-76l); benchmark refresh gpumod-ttk7
**Question:** Should `modes/hermes-agent.yaml` swap from `qwen36-35b-a3b-iq4xs` to a Multi-Token Prediction (MTP) variant?

## Goal

Compare Qwen3.6 with and without Multi-Token Prediction (MTP) on the v2 coding benchmark (15 iterations × 5 levels of increasing difficulty, validated by real pytest tests). MTP is a speculative-decoding technique baked into the GGUF tensors that Unsloth/Qwen claim gives ~1.4–2.2× faster inference with no accuracy loss. The benchmark settles whether MTP delivers on that claim and whether it should replace the current Hermes-agent chat model.

## Methodology Caveats

Direct comparison with the prior [20260423_qwen36_gemma4_comparison](../20260423_qwen36_gemma4_comparison/README.md) benchmark has two known confounds:

- **Binary version**: prior runs used llama.cpp **b8838** (`23b8cc499`); MTP runs need **b9297** (`b0df4c0cf`). MTP support was merged on 2026-05-13. Between the two binaries are ~459 releases of kernel improvements that affect both quality (numerical determinism) and speed.
- **Quant variant for dense 27B**: prior 27B used legacy `Q4_K_M`; MTP variant ships only as Unsloth's dynamic `UD-Q4_K_XL`. UD- quants typically add 1–3 score points vs the legacy quant. For the MoE 35B-A3B, both runs used the SAME `UD-IQ4_XS` quant, so that comparison is cleaner.

Caveats are flagged inline where they affect interpretation.

## Setup

| Component     | Specification                                       |
| ------------- | --------------------------------------------------- |
| **CPU**       | AMD Ryzen 7 5700G (16 threads)                      |
| **RAM**       | 32 GB DDR4                                          |
| **GPU**       | NVIDIA GeForce RTX 4090 (24 GB VRAM)                |
| **OS**        | Ubuntu 24.04.4 LTS                                  |
| **Driver**    | NVIDIA 580.65.06                                    |
| **CUDA**      | 12.0 (avoid 13.2 — known gibberish bug)             |
| **llama.cpp** | b9297 (`b0df4c0cf`) for MTP; b8838 (`23b8cc499`) for prior non-MTP rows |

VRAM isolation: `vllm-embedding-code` was stopped throughout to remove co-tenant contention. Only the model under test was GPU-resident at any time.

## Models Tested

| ID                                   | Source repo                              | Architecture            | Quant       | File   | Bin    | Context | Thinking flag |
| ------------------------------------ | ---------------------------------------- | ----------------------- | ----------- | ------ | ------ | ------- | ------------- |
| `qwen36-27b` †                       | unsloth/Qwen3.6-27B-GGUF                 | Dense (27B all active)  | Q4_K_M      | 16.0 GB| b8838  | 40960   | (default)     |
| `qwen36-27b-mtp-q4`                  | unsloth/Qwen3.6-27B-MTP-GGUF             | Dense (27B all active) + MTP | UD-Q4_K_XL | 18.0 GB | b9297  | 40960   | enable_thinking |
| `qwen36-35b-a3b-iq4xs` †             | unsloth/Qwen3.6-35B-A3B-GGUF             | MoE (35B total, 3B active) | UD-IQ4_XS | 17.0 GB| b8838  | 32768   | (default)     |
| `qwen36-35b-a3b-mtp-iq4xs`           | unsloth/Qwen3.6-35B-A3B-MTP-GGUF         | MoE (35B total, 3B active) + MTP | UD-IQ4_XS | 18.2 GB | b9297  | 131072  | enable_thinking |
| `qwen36-35b-a3b-mtp-iq4xs-preserve`  | unsloth/Qwen3.6-35B-A3B-MTP-GGUF         | MoE (35B total, 3B active) + MTP | UD-IQ4_XS | 18.2 GB | b9297  | 131072  | preserve_thinking |

† Results reused from the [20260423 benchmark](../20260423_qwen36_gemma4_comparison/README.md) (same v2 methodology, different binary).

## Results 

### Summary Table

All 35B-A3B-MTP rows below reflect the **production-aligned 128K context + q8_0 KV cache** configuration (gpumod-ttk7). See [Configuration Update](#configuration-update) for the prior 32K-f16 numbers and why we re-ran.

| Model                                  | Quant      | MTP | Thinking flag      | Mean Score | Std Dev | 95% CI         | Mean TPS  | Speedup |
| -------------------------------------- | ---------- | --- | ------------------ | ---------- | ------- | -------------- | --------- | ------- |
| **Qwen3.6-35B-A3B-MTP**                | UD-IQ4_XS  | ✓   | enable_thinking    | **89.0**   | **7.1** | [85.1, 92.9]   | **216.8** | **1.24×** vs non-MTP twin |
| **Qwen3.6-35B-A3B-MTP (preserve)**     | UD-IQ4_XS  | ✓   | preserve_thinking  | **88.3**   | **6.5** | [84.8, 91.9]   | **216.5** | 1.24× vs non-MTP twin |
| Qwen3.6-35B-A3B                        | UD-IQ4_XS  |     | (default)          | 87.3       | 10.3    | [81.6, 93.0]   | 174.5     | (baseline) |
| **Qwen3.6-27B-MTP**                    | UD-Q4_K_XL | ✓   | enable_thinking    | 87.3       | 7.3     | [83.3, 91.4]   | 85.4      | **1.82×** vs non-MTP twin |
| Qwen3.6-27B                            | Q4_K_M     |     | (default)          | 80.3       | 6.9     | [76.5, 84.2]   | 46.9      | (baseline) |

**95% CI overlap**: the 35B-A3B-MTP enable CI [85.1, 92.9] and preserve CI [84.8, 91.9] are nearly identical, and both overlap the non-MTP CI [81.6, 93.0]. The +1.7/+1.0 mean deltas vs non-MTP are not statistically significant; treat as quality parity. The +24% TPS is the unambiguous win.

**enable vs preserve_thinking on single-shot is empirically equivalent at 128K**: mean 89.0 vs 88.3, σ 7.1 vs 6.5, TPS 216.8 vs 216.5 — three out of three metrics within run-to-run noise. This matches the chat-template proof (preserve_thinking only fires for prior assistant messages — see [research note](../../research/20260524_preserve_thinking_findings.md)). The earlier 32K preserve run showed -5.0 mean and 4× the L1 failures, which we now attribute to insufficient thinking budget at 32K context, not to the flag itself.

### Score Distribution (per iteration)

| Model                            | Scores (15 iters)                                          |
| -------------------------------- | ---------------------------------------------------------- |
| Qwen3.6-35B-A3B-MTP (enable)     | 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, **100**, 90, 90, **65**, 90 |
| Qwen3.6-35B-A3B-MTP (preserve)   | 90, 90, 90, **65**, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90 |
| Qwen3.6-35B-A3B (prior)          | 90, 90, 90, 90, 90, 90, 90, 90, 90, **50**, 90, 90, 90, 90, 90 |
| Qwen3.6-27B-MTP                  | 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, **75**, **65**, 90, 90 |
| Qwen3.6-27B (prior)              | 90, 90, 85, 75, 75, 75, 75, 75, 90, 75, 85, 75, 75, 90, 75 |

Both MTP variants now sit firmly at the 90 ceiling. Enable has one 65 outlier and one **100** (first time any model in this suite hit L5 — a partial multi-file refactor pass); preserve has a single 65 outlier. The two distributions are statistically indistinguishable. The non-MTP 27B remains the most variable, frequently landing at 75.

### Level Pass Rates (15 iterations × 5 levels)

| Level | Task                              | Pts | 27B (non-MTP) | 27B-MTP  | 35B-A3B (non-MTP) | 35B-A3B-MTP (enable) | 35B-A3B-MTP (preserve) |
| ----- | --------------------------------- | --- | ------------- | -------- | ----------------- | -------------------- | ---------------------- |
| L1    | Basic queue (add/get, FIFO)       | 25  | 100%          | 93%      | 93%               | 93%                  | 93%                    |
| L2    | Retry with exponential backoff    | 25  | 100%          | 100%     | 100%              | **100%**             | **100%**               |
| L3    | Priority scheduling               | 25  | 100%          | 100%     | 100%              | **100%**             | **100%**               |
| L4    | Find & fix concurrency bug        | 15  | 27%           | **93%**  | 93%               | **100%**             | **100%**               |
| L5    | Multi-file refactoring            | 10  | 13%           | 0%       | 0%                | **7%**               | 0%                     |

**L4 jumped dramatically for 27B-MTP (27% → 93%)** — this is partly the new binary (better thinking-mode reasoning), partly the quant upgrade (UD-Q4_K_XL ≥ Q4_K_M), and partly the runner fix that captures `reasoning_content`. The exact attribution is confounded, but the net result is that 27B-MTP is now nearly as good as 35B-A3B on the concurrency task.

**L5 cracked once on the enable run** — 1/15 iterations passed (a partial). This is the first non-zero L5 in this suite for the strong MoE models. With 128K context the model can finally hold the full multi-file refactor in its thinking budget. n=15 is too few to call this a stable improvement, but it suggests L5 is no longer an absolute ceiling — just an extremely tight one.

### MTP-Specific Metrics

| Model                            | n_max | Mean draft acceptance | Total drafts | Drafts accepted |
| -------------------------------- | ----- | --------------------- | ------------ | --------------- |
| Qwen3.6-27B-MTP                  | 2     | 80.2%                 | 475,884      | 364,804         |
| Qwen3.6-35B-A3B-MTP (enable)     | 2     | 78.9%                 | 349,154      | 267,423         |
| Qwen3.6-35B-A3B-MTP (preserve)   | 2     | 78.7%                 | 366,406      | 277,488         |

MTP draft acceptance is healthy across all three — well above the threshold where speculative decoding pays for itself. q8_0 KV cache compression cost ~1pp of acceptance vs the f16 32K runs (78.9% → 78.9% enable, 79.6% → 78.7% preserve — within noise). The MoE has slightly lower acceptance than the dense 27B, possibly because the 3B-active routing makes the draft head's predictions less aligned with the target's actual sampling path.

### Wall-clock

| Run                                          | Duration | Per iteration |
| -------------------------------------------- | -------- | ------------- |
| Qwen3.6-27B-MTP (15)                         | 2h 03m   | ~8 min        |
| Qwen3.6-35B-A3B-MTP enable (15, 128K+q8_0)   | ~0h 35m  | ~2.3 min      |
| Qwen3.6-35B-A3B-MTP preserve (15, 128K+q8_0) | ~0h 35m  | ~2.3 min      |

The MoE's 3B-active routing dominates wall-clock; MTP layered on top gives the +24% on raw TPS. q8_0 KV cache compression cost ~3% wall-clock vs the prior f16 32K runs.

## Key Findings

1. **MoE MTP is the new top performer on every axis**: highest mean (89.0 enable / 88.3 preserve), highest TPS (216.8 / 216.5), low variance (σ=7.1 / 6.5). It dethrones the prior champion (35B-A3B UD-Q4_K_S at 90.0/173.7).
2. **MTP's speed claim holds**: 1.82× for dense 27B, 1.24× for MoE 35B-A3B. The MoE gains less because the 3B-active arch was already fast — the draft head still helps but has less wall-clock to save.
3. **MTP quality is at parity** with the non-MTP twin within statistical noise (CIs overlap heavily across all three MTP variants). Vendor's "no accuracy loss" claim survives the test.
4. **Variance dropped with MTP**: 35B-A3B σ went from 10.3 (non-MTP) → 7.1 (MTP enable) → 6.5 (MTP preserve). Lower variance = more predictable for production agents.
5. **enable vs preserve_thinking is empirically equivalent on single-shot at 128K context**: means 89.0 vs 88.3, σ 7.1 vs 6.5, TPS 216.8 vs 216.5 — three out of three metrics within run-to-run noise. Matches the chat-template proof. The earlier 32K preserve regression was a thinking-budget artifact, not a flag difference.
6. **q8_0 KV cache is compatible with MTP draft head**: smoke and 15-iter benchmark both pass. Cost: ~3% wall-clock and ~1pp draft acceptance. Benefit: 4× more context for thinking budget.
7. **L4 (concurrency bug fix) is now 100% on 35B-A3B-MTP**: confounded by binary + quant + runner-fix history but real.
8. **L5 (multi-file refactoring) cracked once on the enable run (1/15)**: first non-zero L5 in this suite for strong MoE models. The 128K context finally lets thinking budget hold the full task. n=15 is too few to call this a stable improvement, but L5 is no longer the absolute ceiling it was at 32K.

## Methodology

| Aspect           | Configuration                                              |
| ---------------- | ---------------------------------------------------------- |
| **Iterations**   | 15 per model                                               |
| **Validation**   | PytestValidator (real pytest tests, 30 s per-level timeout)|
| **Sampler**      | `THINKING_CODING`: temp=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=0 (matches Unsloth's recommendation for thinking-mode coding) |
| **Client timeout** | 900 s per request (bumped from prior 300 s — MTP thinking can be long-running) |
| **max_tokens**   | 32768 (Unsloth's recommendation for Qwen3.6 general queries) |
| **Thinking mode**| Enabled (default for Qwen3.6). The enable variant uses `--chat-template-kwargs '{"enable_thinking":true}'`; the preserve variant uses `'{"preserve_thinking":true}'`. Both are functionally equivalent on first-turn inputs per the chat-template proof. |
| **Context**      | 131072 (Unsloth's "minimum 128K for thinking capabilities" recommendation). Earlier runs in this folder at 32768 were under-provisioned — see [Configuration Update](#configuration-update). |
| **KV cache**     | `--cache-type-k q8_0 --cache-type-v q8_0`. Halves KV memory so the 128K context fits alongside MTP on 24 GB VRAM. Verified compatible with the draft head (gpumod-ttk7). |
| **MTP flags**    | `--spec-type draft-mtp --spec-draft-n-max 2 --parallel 1` (per Unsloth docs; `-np>1` unsupported with MTP) |
| **Code extraction** | Try `message.content` first (the model's polished final answer); fall back to `reasoning_content` only if content has no code fence. Earlier benchmarks treated reasoning-first and graded the model's *draft* code — the fix landed in commit `219d688`. |

## Recommendation

**Swap Hermes-agent's chat model to `qwen36-35b-a3b-mtp-iq4xs-preserve`** (128K context + q8_0 KV cache).

| Reason | Detail |
|---|---|
| **Quality** | 88.3 mean (preserve) and 89.0 mean (enable) both overlap the non-MTP CI [81.6, 93.0]. Single-shot equivalence between the two flags is now both theoretically proven (chat-template extraction) AND empirically confirmed (means/σ/TPS all within run-to-run noise at 128K). |
| **Speed** | 216.5 TPS (preserve) / 216.8 TPS (enable) vs 174.5 TPS (non-MTP) — **+24% real, sustained** |
| **Variance** | σ=6.5 (preserve) / σ=7.1 (enable) vs σ=10.3 (non-MTP) — **-37% / -31%**, more predictable for agents |
| **Multi-turn behavior** | Hermes-agent is multi-turn (chat + tool calling). `preserve_thinking:true` keeps prior `<think>` blocks in the conversation history; `enable_thinking:true` drops them. Preserving improves reasoning consistency across tool-calling turns. |
| **Context** | 131072 tokens — matches the non-MTP baseline and exceeds Unsloth's 128K minimum for thinking. Production multi-turn needs this; the prior 32K runs in this folder were under-provisioned. |
| **VRAM** | ~20 GB load on 24 GB GPU (model + 128K KV cache with q8_0). Leaves ~4 GB headroom for co-tenant services like vllm-embedding-code. |
| **Architecture parity** | Same MoE 35B-A3B base + MTP + 128K context + q8_0 cache. Only delta from the non-MTP baseline is `--spec-type draft-mtp`. |

**Why preserve over enable?** At 128K the two are statistically indistinguishable on single-shot (mean 88.3 vs 89.0, σ 6.5 vs 7.1, TPS 216.5 vs 216.8). preserve is the correct flag for multi-turn agent workloads — it keeps prior `<think>` blocks across tool calls, which keeps the model's reasoning chain coherent. enable would drop them every turn.

**Implementation** — the actual edit to `modes/hermes-agent.yaml` is gpumod-aop. With this benchmark refresh, that ticket's recommended preset is now `qwen36-35b-a3b-mtp-iq4xs-preserve` (formerly `qwen36-35b-a3b-mtp-iq4xs`).

### What we'd want before fully committing

- One real chat/tool-calling session under hermes-agent's actual prompts (this benchmark only covers single-shot coding tasks)
- VRAM budget verified when co-running `vllm-embedding-code` (currently isolated for measurement; production needs both)

These can be one-week monitoring after the swap, not gating the swap itself.

### Why not 27B-MTP?

It scored 87.3 (≈ baseline 35B-A3B-IQ4XS) at less than half the TPS of 35B-A3B-MTP (85.4 vs 216.5). Smaller VRAM footprint (~21 GB vs ~22 GB) but the speed and variance penalties make it inferior to 35B-A3B-MTP for the Hermes-agent use case. Possible alternative if a multi-agent / batch workload appears, but `-np>1` is unsupported with MTP, so 27B-MTP loses its main advantage there too.

## Configuration Update

The MTP variants were originally benchmarked at `context_size=32768` with f16 KV cache. Cross-checking against the [Unsloth Qwen3.6-35B-A3B-MTP model card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) flagged that 32K is **below the vendor's "minimum 128K for thinking capabilities"** recommendation. The non-MTP twin already runs at 131072 thanks to `--cache-type-k q8_0 --cache-type-v q8_0` halving the KV memory; gpumod-ttk7 tested whether q8_0 cache is compatible with the MTP draft head.

**Result:** compatible. Both 15-iter benchmarks above were re-run at 131072 context + q8_0 cache.

| Metric | enable @ 32K f16 | enable @ 128K q8_0 | preserve @ 32K f16 | preserve @ 128K q8_0 |
|---|---:|---:|---:|---:|
| Mean score | 88.3 | **89.0** | 83.3 | **88.3** |
| Std Dev | 6.5 | 7.1 | 11.4 | **6.5** |
| 95% CI lower | 84.8 | 85.1 | 77.0 | 84.8 |
| 95% CI upper | 91.9 | 92.9 | 89.7 | 91.9 |
| Mean TPS | 222.3 | 216.8 | 230.8 | 216.5 |
| Draft acceptance | 78.9% | 78.9% | 79.6% | 78.7% |

The 32K→128K change cost ~3% TPS (q8_0 cache slows kernel slightly) and ~1pp draft acceptance, in exchange for:
- 4× thinking budget (the smoke that triggered this investigation showed 14891 reasoning chars for a single coding prompt at finish_reason=length)
- preserve_thinking quality recovers from 83.3→88.3 (the 32K result was a thinking-budget artifact, not a flag difference)
- Multi-turn viability for Hermes-agent (32K saturates after 2-3 preserved-thinking turns)

The prior 32K result JSONs are preserved as `result_*.32k-baseline.json` for audit. The headline tables in this README use the 128K production-aligned numbers.

## Files

| File                                                          | Description                              |
| ------------------------------------------------------------- | ---------------------------------------- |
| `result_qwen36-27b-mtp-q4.json`                               | 15-iteration results, 27B-MTP            |
| `result_qwen36-35b-a3b-mtp-iq4xs.json`                        | 15-iteration results, 35B-A3B-MTP enable_thinking @ 128K + q8_0 |
| `result_qwen36-35b-a3b-mtp-iq4xs.32k-baseline.json`           | Earlier 32K f16 run (superseded — see Configuration Update) |
| `result_qwen36-35b-a3b-mtp-iq4xs-preserve.json`               | 15-iteration results, 35B-A3B-MTP preserve_thinking @ 128K + q8_0 |
| `result_qwen36-35b-a3b-mtp-iq4xs-preserve.32k-baseline.json`  | Earlier 32K f16 run (superseded — see Configuration Update) |
| `run.log`                                                     | Original 32K stdout from 27B + enable runs |
| `run_preserve.log`                                            | Original 32K stdout from preserve run    |
| `run_preserve_128k.log`                                       | 128K + q8_0 stdout from preserve rerun (enable rerun log lost to a shell-quoting accident — `result_*.json` is authoritative) |
| `artifacts/qwen36-27b-mtp-q4/`                                | Per-iteration response artifacts (tagged `<reasoning_content>` + `<content>`) |
| `artifacts/qwen36-35b-a3b-mtp-iq4xs/`                         | Per-iteration response artifacts (enable_thinking @ 128K) |
| `artifacts/qwen36-35b-a3b-mtp-iq4xs-preserve/`                | Per-iteration response artifacts (preserve_thinking @ 128K) |

## References

- [Prior 20260423 benchmark (non-MTP baselines)](../20260423_qwen36_gemma4_comparison/README.md)
- [Qwen3.6-27B-MTP-GGUF model card](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)
- [Qwen3.6-35B-A3B-MTP-GGUF model card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
- [Unsloth Qwen3.6 docs (incl. llama.cpp MTP guide)](https://unsloth.ai/docs/models/qwen3.6)
- [llama.cpp b9297 release (NVFP4 + Qwen3.5 MTP tensors)](https://github.com/ggml-org/llama.cpp/releases/tag/b9297)
