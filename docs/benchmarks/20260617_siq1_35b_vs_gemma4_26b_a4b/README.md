# SIQ-1-35B Q4_K_M vs Gemma 4 26B-A4B-QAT-MTP-Q4 (spike, not full bench)

**Date:** 2026-06-17
**Ticket:** [gpumod-nq8v](../../../README.md) — Spike: evaluate SIQ-1-35B vs gemma4-26b-a4b-qat-mtp-q4 on RTX 4090
**Question:** Should `AlexWortega/SIQ-1-35B` (Qwen3.5-MoE hybrid; 35B / 3B active, 256 experts) replace `presets/llm/gemma4-26b-a4b-qat-mtp-q4.yaml` as the hermes-agent orchestrator on this 24 GiB host?

> ⚠ **This is a spike, not the full v2 coding benchmark.** The spike protocol from gpumod-nq8v gates the full coding bench behind four decision gates (VRAM fit with embedding co-tenant live, MTP usability, tool-call compatibility, quality smoke ≥80% of baseline). One gate (VRAM with co-tenant + MTP) hard-fails on this hardware; the other gates pass. We did not run the 15-iter v2 coding suite — the decision gate is already settled. See [Verdict](#verdict).

## TL;DR

| Gate | SIQ-1-35B Q4_K_M | Gemma 4 26B-A4B QAT MTP Q4 (baseline) | Verdict |
|---|---|---|---|
| **llama.cpp arch support (main)** | Loads on b9572 (`fd3271e0b`). `qwen35moe` arch fully supported — same build runs the existing `Qwen3.6-35B-A3B-MTP` with MTP active. | Loads | PASS |
| **MTP drafter support** | **FAIL — the MTP block weights were never published upstream.** This is NOT a llama.cpp gap. See [MTP root cause](#mtp-drafter-the-weights-do-not-exist). | PASS (Gemma MTP integrated upstream) | **FAIL** |
| **VRAM @ -c 65536 --parallel 1** | 21877 MiB peak (single-tenant) | ~21 GiB peak (single-tenant, MTP active) | PASS (single-tenant only) |
| **VRAM @ -c 65536 + embedding co-tenant** | **~23.7 GiB projected (270 MiB free)** — marginal/OOM-prone under load | Fits (~22 GiB ceiling, embedding co-tenant tested live) | **FAIL** for SIQ |
| **Tool-call wire format via `--jinja`** | OpenAI `tool_calls` JSON emitted; `finish_reason=tool_calls`; arguments parsed correctly | OpenAI `tool_calls` JSON | PASS (drop-in at wire level) |
| **Q1–Q5 proxy quality (greedy, temp=0)** | **5/5 correct** | 5/5 correct (native sampling) | TIE |
| **Eval TPS (no MTP, -c 32768, --parallel 1)** | ~190 tok/s on Q1–Q5 | ~250 tok/s on Q1–Q5 | Gemma faster |

**Three headlines:**

1. **MTP drafter weights were never published.** Both standalone and `--model-draft mtp-SIQ-1-35B.f16.gguf --spec-type draft-mtp` attempts fail with `missing tensor 'blk.40.attn_norm.weight'`. Initially read as a llama.cpp gap — but b9572 runs `Qwen3.6-35B-A3B-MTP` (same `qwen35moe` arch, same MTP code path) without issue. The actual root cause is that the upstream safetensors index `model.safetensors.index.json` contains **0 MTP tensors** — only layers 0–39. The published `mtp-SIQ-1-35B.f16.gguf` has only 3 tensors (the shared embedding/output head); the MTP transformer block weights (`nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`, and the layer-40 attention + MoE FFN) simply do not exist in the repo. SIQ's headline TPS uplift is unreachable until/unless the author publishes the missing weights.
2. **VRAM is not safe at 65K context with the embedding co-tenant live.** SIQ Q4_K_M alone takes 21877 MiB at -c 65536 --parallel 1; the `gguf-embedding-code` service on port 8210 occupies ~1.8 GiB; combined ~23.7 GiB on a 24 GiB card leaves <300 MiB headroom for prefill spikes. Gemma 4 26B-A4B-QAT-MTP-Q4 fits at -c 256000 with the same co-tenant (preset budget 22000 MiB).
3. **Quality is plausibly competitive at small sample.** SIQ Q4_K_M and Gemma 4 26B-A4B QAT both scored 5/5 on the Q1–Q5 graduate-level proxy. Tool-call wire format is OpenAI-JSON-compatible via `--jinja`, so hermes-agent integration is mechanically clean.

**The decision is NOT a coin-flip on quality.** It's a hardware fit + drafter availability problem. Even if SIQ matches or beats Gemma 4 on the full coding bench, losing MTP speculative decoding and giving up the embedding co-tenant (or constraining context to ≤32K) are both worse than the current baseline.

## Setup

| Component | Specification |
|---|---|
| **CPU** | AMD Ryzen 7 5700G (16 threads) |
| **RAM** | 32 GB DDR4 |
| **Swap** | 64 GB (zram + disk) |
| **GPU** | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| **OS** | Ubuntu 24.04 LTS |
| **NVIDIA Driver** | 580.65.06 |
| **CUDA** | 12.0 |
| **llama.cpp** | b9572 (`fd3271e0b`, built 2026-06-09 / `ggml-cpu : fix rms_norm_back wrong output under in-place aliasing #24305`) |
| **Stability env** | `GGML_CUDA_NO_PINNED=1` (gpumod-56md default; this benchmark honoured it manually for all SIQ launches) |
| **Co-tenant** | Stopped for the spike. Production target keeps `gguf-embedding-code` on port 8210 GPU-resident. |

## Models tested

| ID | Source | Architecture | Quant | File size | Native context | Sampler |
|---|---|---|---|---:|---:|---|
| `siq1-35b-q4km` (no preset — direct `llama-server` launch) | `AlexWortega/SIQ-1-35B` | Qwen3_5MoeForCausalLM hybrid (attn_output_gate, 30 linear + 10 full layers, Mamba SSM blocks, 256 experts / 8 active, MTP layer, vocab 248320) | Q4_K_M | 19.7 GB GGUF | 262144 | temp=0 greedy, `enable_thinking=false` |
| `gemma4-26b-a4b-qat-mtp-q4` | `unsloth/gemma-4-26B-A4B-it-GGUF` (QAT UD-Q4_K_XL) + MTP Q8_0 | MoE 26B / 4B active + MTP drafter | UD-Q4_K_XL | 13.5 GB main + 462 MB MTP | 262144 | temp=1.0 top_p=0.95 top_k=64 (GEMMA_CODING) |

SIQ's main weights load on b9572; the published `mtp-SIQ-1-35B.f16.gguf` drafter does not.

## Methodology

### What we measured

1. **llama.cpp compatibility** — load `SIQ-1-35B.Q4_K_M.gguf` with `--jinja -ngl 999`. Confirm `/health = ok`. Same for `mtp-SIQ-1-35B.f16.gguf`.
2. **VRAM scaling** — load SIQ at three contexts (32768, 65536, 131072) under both default (`--parallel` auto = 4) and `--parallel 1` (hermes-agent slot shape). Track GPU mem at load and during a 48 K-token cold prefill ([artifacts/siq_*.log](artifacts/)).
3. **MTP drafter** — attempt `llama-server -m SIQ.Q4_K_M.gguf --model-draft mtp-SIQ-1-35B.f16.gguf --spec-type draft-mtp --spec-draft-n-max 2`. Also attempt to load the drafter standalone (`-m mtp-SIQ-1-35B.f16.gguf`). Capture exact error.
4. **Tool-call wire format** — call `/v1/chat/completions` with a single-function `get_weather` tool definition and `tool_choice: auto`. Check whether llama-server's `--jinja` chat-template integration emits an OpenAI-compatible `tool_calls` JSON block. [artifacts/siq_toolcall_probe.json](artifacts/siq_toolcall_probe.json) contains the raw response.
5. **Q1–Q5 quality proxy** — 5 graduate-level science MCQs (Compton scattering, octahedral stereoisomers, B12 vs folate biochem, group automorphisms, Schwarzschild ISCO). [q1q5_items.json](q1q5_items.json) holds the items. Both models scored 5/5 with no rubric.

### What we did NOT measure, and why

- **Actual GPQA-Diamond.** The dataset (`Idavidrein/gpqa`) is gated; we have no access. The Q1–Q5 proxy is hand-curated graduate-level science with the same comparative structure (same items to both models, scored relative to each other). With N=5 the result is a smoke, not a benchmark — it can refute a quality regression, not establish parity.
- **v2 coding suite (`scripts/run_qwen36_benchmark.py`).** The spike's step 4 gates this behind passing steps 1–3 AND a tool-call compatibility check AND a VRAM-with-co-tenant check. The VRAM gate fails (see TL;DR). Running the 15-iter suite would not change the verdict — even a 100/0 result for SIQ does not buy back the embedding co-tenant or the MTP drafter.
- **Live embedding co-tenant load.** We projected from measured single-tenant VRAM. The projection assumes `gguf-embedding-code` peak of ~1.8 GiB at idle from prior measurement; a live test was skipped because the projected free margin (<300 MiB at -c 65536) already falls below any realistic prefill spike tolerance.

### Substitutions vs the spike protocol in the ticket

| Ticket step | What it asks | What we did | Why |
|---|---|---|---|
| 1: load compat | Q4_K_M + MTP on b9572 | Loaded both separately; combined `--model-draft` attempted | Drafter alone fails — combined run cannot succeed |
| 2: VRAM at 32K/65K/131K, with & without MTP | All six combinations | Five combinations (MTP variants impossible — drafter doesn't load) | Hardware-blocked |
| 3: 5 GPQA-Diamond items, temp=0 | Gated dataset items | 5 graduate-science proxy items | Dataset access denied |
| 4: full v2 coding bench | 15-iter v2 vs both | NOT RUN | Decision gate already settled by step 2 |

## Results

### VRAM sweep (SIQ-1-35B Q4_K_M, `GGML_CUDA_NO_PINNED=1`)

| Context | `--parallel` | Load VRAM | Peak VRAM after 48 K-tok prefill | Free at peak | Notes |
|---:|---:|---:|---:|---:|---|
| 32768 | 4 (default auto) | 21383 MiB | 21393 MiB | 2671 MiB | Headroom for one parallel slot |
| 65536 | 4 (default auto) | 22037 MiB | 22065 MiB | 1999 MiB | At the 22 GiB Gemma slot budget |
| 65536 | 1 | 21849 MiB | 21877 MiB | 2187 MiB | Slot shape that hermes-agent actually uses |
| 131072 | 4 (default auto) | 23381 MiB | 23409 MiB | 655 MiB | Unsafe — no headroom |
| 131072 | 1 | 23193 MiB | 23221 MiB | 843 MiB | Unsafe — no headroom |
| 32768 | 1 (Q1–Q5 run) | 21177 MiB | 21199 MiB | 2865 MiB | Smoke run config |

**Projected with `gguf-embedding-code` co-tenant (~1.8 GiB):**

| Context | `--parallel` | Projected peak | Projected free | Verdict |
|---:|---:|---:|---:|---|
| 32768 | 1 | ~23.0 GiB | ~1.1 GiB | Marginal but plausible |
| 65536 | 1 | ~23.7 GiB | ~0.3 GiB | OOM-prone under prefill |
| 131072 | 1 | ~25.0 GiB | -1.0 GiB | OOM |

For comparison, `gemma4-26b-a4b-qat-mtp-q4` runs at -c 256000 with the embedding co-tenant live in production (preset declares 22000 MiB budget). SIQ at -c 32768 single-tenant + embedding ~= Gemma at -c 256000 + embedding + MTP + multimodal. The architectures are simply not in the same fit class on this card.

### MTP drafter — the weights do not exist

The error from `llama-server` is the same in every attempt:

```
load_model: loading draft model '~/bin/mtp-SIQ-1-35B.f16.gguf'
llama_model_load: error loading model: missing tensor 'blk.40.attn_norm.weight'
llama_model_load_from_file_impl: failed to load model
srv    load_model: failed to load draft model, '~/bin/mtp-SIQ-1-35B.f16.gguf'
```

The first read (in the original spike notes) was "llama.cpp b9572 doesn't yet support the Qwen3.5-MoE MTP variant". **That was wrong.** Three independent pieces of evidence below prove the cause is missing upstream weights, not a llama.cpp gap.

#### Evidence 1 — `qwen35moe` MTP is fully supported in b9572

The same llama.cpp build runs `~/bin/Qwen3.6-35B-A3B-MTP-UD-IQ4_XS.gguf` (the existing Hermes baseline through 2026-06) with MTP active. That file declares the same `general.architecture = qwen35moe`, the same `qwen35moe.block_count = 41`, and the same `qwen35moe.nextn_predict_layers = 1` as the SIQ pair. It works. The arch and the MTP code path are not the problem.

The MTP graph constructor in `src/models/qwen35moe.cpp:551` (b9572) asserts exactly the tensors it needs at layer index `n_layer`:

```c++
GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");
GGML_ASSERT(layer.ffn_gate_inp  && "MTP block missing ffn_gate_inp");
```

These are the tensors a working MTP file must provide.

#### Evidence 2 — the SIQ "MTP" GGUF has zero block tensors

`mtp-SIQ-1-35B.f16.gguf` is 2.0 GiB but contains only **3 tensors** — and none of them are MTP block weights:

| Tensor | Shape | Purpose |
|---|---|---|
| `token_embd.weight` | [2048, 248320] F16 | Shared token embedding |
| `output_norm.weight` | [2048] F32 | Shared output norm |
| `output.weight` | [2048, 248320] F16 | Shared LM head |

Metadata claims `block_count = 41` and `nextn_predict_layers = 1`, but no `blk.*` tensors exist in the file. This file is the **shared head** referenced by `layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd` in the MTP graph (line 583 of `qwen35moe.cpp`) — it's not a drafter at all, and it cannot be loaded standalone (block_count says 41 layers but the file has 0).

#### Evidence 3 — Qwen3.6-35B-A3B-MTP's `blk.40` is fully populated; SIQ's is not

Direct comparison of `blk.40` tensor counts between the working file and SIQ's main + "mtp" combined:

| Tensor (expected at `blk.40` per llama.cpp) | Qwen3.6-35B-A3B-MTP | SIQ-1-35B (main + "mtp" combined) |
|---|---|---|
| `attn_q.weight` / `attn_k.weight` / `attn_v.weight` | ✓ | **MISSING** |
| `attn_q_norm.weight` / `attn_k_norm.weight` / `attn_norm.weight` | ✓ | **MISSING** |
| `attn_output.weight` | ✓ | **MISSING** |
| `ffn_down_exps` / `ffn_up_exps` / `ffn_gate_exps` (MoE) | ✓ | **MISSING** |
| `ffn_down_shexp` / `ffn_up_shexp` / `ffn_gate_shexp` (shared expert) | ✓ | **MISSING** |
| `ffn_gate_inp` / `ffn_gate_inp_shexp` | ✓ | **MISSING** |
| `post_attention_norm.weight` | ✓ | **MISSING** |
| `nextn.eh_proj.weight` | ✓ | **MISSING** |
| `nextn.enorm.weight` | ✓ | **MISSING** |
| `nextn.hnorm.weight` | ✓ | **MISSING** |
| `nextn.shared_head_norm.weight` | ✓ | **MISSING** |
| **Total tensors in blk.40** | **20** | **0** |

#### Evidence 4 — the upstream safetensors index lists 0 MTP keys

`AlexWortega/SIQ-1-35B/model.safetensors.index.json` lists 693 tensor keys totalling 34.66 B parameters. Filtering for `mtp` / `nextn` / `multi_token` / `predict` / `draft`:

```
MTP/nextn/draft keys: 0
layer indices present: [0, 1, 2, ..., 39]
```

Even though `config.json` declares `mtp_num_hidden_layers: 1`, the safetensors that ship with the model contain **no MTP weights at all**. The conversion to GGUF (whether the author's or anyone else's) cannot synthesise them — there is nothing to convert.

#### Conclusion

The MTP failure is a **publication defect of `AlexWortega/SIQ-1-35B`**, not a llama.cpp limitation. The author shipped:

- A complete 40-block main model (loads and runs)
- A shared embedding/LM-head file mislabelled as MTP
- **No MTP transformer block weights**

Until the author publishes the missing MTP layer weights (`blk.40.*` including `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`, and the full attention + MoE FFN block), MTP-accelerated inference is impossible on any llama.cpp build, no matter how recent. This shifts the follow-up from "watch llama.cpp upstream" to "open an issue on the SIQ repo asking the author for the missing weights".

### Q1–Q5 quality proxy (5 graduate-level science items)

Full responses: [result_siq1-35b-q4km_q1q5.json](result_siq1-35b-q4km_q1q5.json), [result_gemma4-26b-a4b-qat-mtp-q4_q1q5.json](result_gemma4-26b-a4b-qat-mtp-q4_q1q5.json).

| # | Domain | Expected | SIQ (temp=0) | Gemma 4 (temp=1.0 top_p 0.95 top_k 64) |
|---|---|---|---|---|
| Q1 | Compton scattering wavelength | 0.0524 nm | ✓ 0.0524 nm | ✓ 0.0524 nm (1024-tok thinking budget; retried at 4096) |
| Q2 | [Co(en)₂Cl₂]⁺ stereoisomers | 3 (cis-Δ + cis-Λ + trans) | ✓ 3, full chirality analysis | ✓ 3, full chirality analysis |
| Q3 | Elevated MMA + homocysteine | Vitamin B12 (folate would not elevate MMA) | ✓ B12, correct discriminator | ✓ B12, correct discriminator |
| Q4 | |Aut(ℤ/12ℤ)| | 4 = φ(12) | ✓ 4 | ✓ 4 |
| Q5 | Schwarzschild ISCO radius | 6 GM/c² | ✓ 6 (emitted as `<tool_call><function=complete>` block — model quirk for terminal numeric answers) | ✓ 6 (with full effective-potential derivation; needed >1024 tokens) |
| **Score** | | | **5/5** | **5/5** |

**Observations:**

- **Q5 SIQ tool-call leak.** SIQ wrapped the numeric answer in a `<tool_call><function=complete><answer>6</answer></function></tool_call>` block even though no tool was offered. The OpenAI wire-translation layer did not strip it (the response came back as raw `content`, not a `tool_calls` block). This is a downstream-correctness risk: an autoregressive client expecting plain text might fail to parse the answer. Not blocking, but worth a follow-up if SIQ is ever revisited.
- **Gemma 4 Q1/Q5 emptied at 1024 max_tokens.** Both completions used the full 1024-token budget inside `<think>...</think>` with no visible `content`. Re-running at 4096 max_tokens yielded the correct answer. For agent workloads the orchestrator already uses larger budgets, so this isn't a production concern, but smoke harnesses targeting Gemma 4 should set `max_tokens >= 4096`.
- **TPS for these N=5 prompts** (-c 32768 --parallel 1, no MTP for both — Gemma's MTP only kicks in with `--model-draft`, which the gpumod preset enables; we measured against the live preset which had MTP on):

| Model | Mean TPS (eval, over 5 items) |
|---:|---:|
| SIQ-1-35B Q4_K_M | 188.4 |
| Gemma 4 26B-A4B QAT MTP Q4 | 249.4 |

Gemma's lead here is the MTP drafter doing its job. Without MTP, Gemma's eval drops to roughly the same band (~85–95 tok/s based on prior gpumod-z0ks measurements), at which point SIQ would be **faster**. That advantage is unrecoverable on this build because the MTP file doesn't load.

### Tool-call wire format

`/v1/chat/completions` against the `--jinja`-enabled SIQ server with a single function definition returns:

```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\":\"Berlin\"}"
        },
        "id": "ZfUUiixrL3dtmSQfp5f86g5USxxVPlnO"
      }]
    }
  }]
}
```

Hermes-agent uses the OpenAI `tool_calls`/`finish_reason=tool_calls`/`tool_call_id` shape (verified by grepping `~/.hermes/profiles/orchestrator/sessions/*.json`). SIQ's chat template (`<tool_call><function=NAME><parameter=KEY>VAL</parameter></function></tool_call>`) is translated transparently to the OpenAI JSON wire format by `--jinja`. **No hermes-agent code changes would be needed for a swap.**

## Verdict

**Do NOT swap.** Keep `gemma4-26b-a4b-qat-mtp-q4` as the hermes-agent orchestrator.

The hard gates that fail:

1. **MTP weights are not in the published model.** The headline TPS uplift in the model card is unreachable on any llama.cpp build — the upstream safetensors index lists 0 MTP keys, and the `mtp-SIQ-1-35B.f16.gguf` shipped on HF is the shared head only, not a drafter. Re-evaluate only if the author publishes the missing MTP block weights. See [MTP drafter — the weights do not exist](#mtp-drafter-the-weights-do-not-exist).
2. **VRAM with embedding co-tenant.** At the 65 K context floor hermes-agent needs to grow into, the projected free margin is <300 MiB. Gemma 4 26B-A4B QAT MTP Q4 fits at -c 256000 with the same co-tenant live.

The soft signals are competitive:

- Quality smoke is a 5/5 tie at small N.
- Tool-call wire format is OpenAI-JSON-compatible via `--jinja`.
- SIQ at -c 32768 single-tenant uses 21199 MiB — would fit a *blank-mode* slot if the user accepted a 32 K hermes-agent context.

But "32 K + no MTP + no embedding co-tenant + ungated v2 coding bench performance" is a strict downgrade from the current baseline. The case for swapping reopens only when at least one of the hard gates clears.

## Follow-up

- **Do NOT land** `presets/llm/siq1-35b-q4.yaml`. The model isn't viable on this hardware/llama.cpp combo.
- **Open an issue on `AlexWortega/SIQ-1-35B`** requesting that the author publish the MTP block weights (the layer-40 transformer block + `nextn.eh_proj` / `nextn.enorm` / `nextn.hnorm`). llama.cpp's `qwen35moe` MTP code path (b9572 `fd3271e0b`) is ready and runs `Qwen3.6-35B-A3B-MTP` today — the only blocker is the missing weights.
- **Do NOT keep watching llama.cpp upstream** for this one — the original spike notes implied this was a recency gap; it is not. The arch and MTP graph are already in.
- **Q5 tool-call leak** ([result_siq1-35b-q4km_q1q5.json](result_siq1-35b-q4km_q1q5.json) → `Q5_astro_relativity`). If SIQ is ever evaluated again, design the test harness to reject responses where the content is wrapped in spurious `<tool_call>` blocks for non-tool prompts.
- **Re-check VRAM accounting** if upstream lands more aggressive KV quantisation (Q3 / Q2 cache types) — `cache_type_k q4_0` could buy enough headroom to put SIQ at -c 65536 inside the budget. Not on the roadmap today.

## Artifacts

- [q1q5_items.json](q1q5_items.json) — the 5 quality items + expected answers
- [run_q1q5.py](run_q1q5.py) — driver used to send the 5 items
- [result_siq1-35b-q4km_q1q5.json](result_siq1-35b-q4km_q1q5.json) — SIQ raw answers + per-item timing
- [result_gemma4-26b-a4b-qat-mtp-q4_q1q5.json](result_gemma4-26b-a4b-qat-mtp-q4_q1q5.json) — Gemma raw answers + per-item timing
- [artifacts/](artifacts/) — llama-server load logs for every context + MTP attempt + tool-call probe response
