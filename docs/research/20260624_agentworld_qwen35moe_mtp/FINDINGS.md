# Qwen-AgentWorld-35B-A3B — a hybrid Gated-DeltaNet + MTP MoE, and why its GGUF needs special handling

**Date:** 2026-06-24
**Tickets:** gpumod-qsgl (epic), gpumod-qsgl.1 (llama.cpp rebuild), gpumod-qsgl.2 (load-test spike), gpumod-qsgl.3 (benchmark — deferred)
**Status:** Loads + generates on llama.cpp b9784 via a **per-preset metadata override** (no file modification). Context ceiling (Q2c) and the comparison benchmark are still open.
**Preset:** [`presets/llm/agentworld-35b-a3b-q4.yaml`](https://github.com/jaigouk/gpumod/blob/main/presets/llm/agentworld-35b-a3b-q4.yaml)
**Cross-repo source brief:** local-llm-lab spike `qwen-jjx`; verdict written back to `local-llm-lab/kb/sources/qwen-agentworld.md`.

## TL;DR

`Qwen/Qwen-AgentWorld-35B-A3B` is **architecturally unlike every other model gpumod
serves today**. It is a *hybrid* MoE: most layers are **Gated-DeltaNet** (linear /
recurrent attention, SSM-style state) and only every 4th layer is full softmax
attention, plus it ships a **multi-token-prediction (MTP) layer** on top. llama.cpp
maps it to the **`qwen35moe`** runtime arch (distinct from `qwen3moe`).

The publicly available pre-quantized Q4_K_M GGUFs (`gaoqianshen`, `mitkox`,
`ivgranite` — all `gguf-my-repo` outputs) are **defective in a specific, recognisable
way**: the conversion **counted the MTP layer in `block_count` but never emitted its
tensors**. As shipped, the model aborts at load with
`missing tensor 'blk.40.attn_norm.weight'`.

The fix is **not** a file edit and **not** a runtime gap. It is two
`--override-kv` flags carried only in this model's preset `extra_args`:

```
--override-kv qwen35moe.block_count=int:40 --override-kv qwen35moe.nextn_predict_layers=int:0
```

This corrects the metadata **in memory at load time** — the GGUF on disk is never
touched, and no other preset (e.g. the hermes-agent/gemma4 daily driver) is affected.
With the override the **complete 40-layer model loads and generates at full quality**;
the only thing dropped is optional MTP speculative decoding (which llama.cpp does not
run for this arch anyway).

## Why this model is different

| Dimension | Most gpumod models (Qwen3.x dense/MoE, Gemma 4) | Qwen-AgentWorld-35B-A3B |
|---|---|---|
| HF architecture | `Qwen3MoeForCausalLM`, `Gemma...` | **`Qwen3_5MoeForConditionalGeneration`** |
| HF `model_type` | `qwen3moe`, etc. | **`qwen3_5_moe`** |
| llama.cpp runtime arch | `qwen3moe`, `gemma...` | **`qwen35moe`** (≠ `qwen3moe`) |
| Attention | uniform softmax attention | **hybrid**: `10 × (3 × Gated-DeltaNet → 1 × Gated-Attention)`; `full_attention_interval=4` |
| KV cache growth | every layer | **only the 10 full-attention layers** (DeltaNet layers carry fixed SSM state) |
| Speculative head | none | **1 MTP layer** (`mtp_num_hidden_layers=1`, `nextn_predict_layers=1`) |
| Experts | varies | 256 experts, **8 routed + 1 shared** (A3B active) |

**Naming-collision warning.** gpumod's `src/gpumod/compatibility.py` maps the string
`qwen35moe` from `Qwen3MoeForCausalLM`. That is misleading: per llama.cpp's own
converter, `Qwen3MoeForCausalLM → qwen3moe` and
**`Qwen3_5MoeForConditionalGeneration → qwen35moe`**. The `qwen35moe` *runtime* arch is
the **Qwen3.5** hybrid family (this model), not the Qwen3.6 MoE family gpumod's preset
names call `qwen36-*`. Re-align that mapping if/when the compat matrix is touched.

## Verified architecture (config.json + GGUF metadata)

Base model card / `config.json` (`Qwen/Qwen-AgentWorld-35B-A3B`):

- `architectures = ["Qwen3_5MoeForConditionalGeneration"]`, `model_type = qwen3_5_moe`
- `num_hidden_layers = 40`, **`mtp_num_hidden_layers = 1`**, `mtp_use_dedicated_embeddings = false`
- `num_experts = 256`, `num_experts_per_tok = 8`, `full_attention_interval = 4`
- `layer_types`: 40 entries alternating linear / full attention (full every 4th)
- `context_length = 262144`; total 35B / ~3B activated

GGUF metadata (`gaoqianshen/...-Q4_K_M`, via `llama-gguf` + `gguf_dump.py`):

- `general.architecture = "qwen35moe"`
- `qwen35moe.block_count = 41`  ← **40 transformer + 1 MTP**
- `qwen35moe.nextn_predict_layers = 1`  ← the MTP layer count
- `embedding_length = 2048`, `attention.head_count = 16`, `attention.head_count_kv = 2`
- `attention.key_length = 256`, `attention.value_length = 256`
- `expert_count = 256`, `expert_used_count = 8`, `expert_feed_forward_length = 512`, `expert_shared_feed_forward_length = 512`
- DeltaNet/SSM: `ssm.conv_kernel = 4`, `ssm.state_size = 128`, `ssm.group_count = 16`, `ssm.time_step_rank = 32`, `ssm.inner_size = 4096`
- `rope.freq_base = 1e7`, `rope.dimension_sections = [11,11,10,0]`, `rope.dimension_count = 64`
- File: **21,166,758,816 B (19.71 GiB)**, single file, no splits

Tensor inventory: **1467 tensors; blocks 0–39 fully present; block 40 has ZERO
tensors** (the MTP layer was dropped during conversion).

## The defect class: "MTP counted, MTP not emitted"

The chain that produces the failure:

1. `config.json` declares 40 transformer layers **+ 1 MTP layer**.
2. The `gguf-my-repo` automated converter writes `block_count = 41` and
   `nextn_predict_layers = 1` (i.e. "the last layer is MTP").
3. …but it **emits no tensors for that MTP layer** — `gguf-my-repo` does not pass the
   converter's `--mtp` handling, so the MTP weights are silently dropped.
4. The runtime reads `block_count = 41`, walks to block 40, and aborts on the first
   tensor it cannot find: `missing tensor 'blk.40.attn_norm.weight'`.

This is a **packaging defect in the artifact, not a runtime gap.** llama.cpp b9784 fully
supports `qwen35moe` (`LLM_ARCH_QWEN35MOE` + `src/models/qwen35moe.cpp` +
`src/models/delta-net-base.cpp`; in-tree since **b7990 / PR #19468, 2026-02-10**). The
host's *previous* binary (b9572) already supported it — the brief's premise that
"qwen3_5_moe runtime support is recent, must rebuild" was **stale** (the rebuild to
b9784 is good hygiene, not a prerequisite).

**All three published Q4_K_M repos share this defect** (verified by byte size, not
re-downloaded): `mitkox` is byte-identical to `gaoqianshen` (21,166,758,816 B);
`ivgranite` differs by 96 B of metadata only. None will load as-shipped.

### Why naïvely shifting `block_count` is not enough

`block_count = 40` alone fails differently: the runtime then treats block **39** as the
MTP layer and demands `blk.39.nextn.eh_proj.weight` (block 39 is a *real* full-attention
layer and has no nextn tensors). You must also set `nextn_predict_layers = 0` so the
runtime expects **40 regular layers and no MTP layer at all** — which is exactly what
the file contains.

## The fix — conditional, in-memory, per-preset

```yaml
# presets/llm/agentworld-35b-a3b-q4.yaml  (extra_args)
--override-kv qwen35moe.block_count=int:40 --override-kv qwen35moe.nextn_predict_layers=int:0
```

Properties that matter for an operator who runs a different daily driver:

- **Non-permanent.** `--override-kv` patches metadata in RAM at load. The 19.71 GiB GGUF
  on disk is never modified; there is nothing to revert.
- **Conditional / scoped.** The flags live only in this preset's `extra_args`. Starting
  any other service (hermes-agent / gemma4, `code` mode, etc.) uses a different preset
  with no overrides — **zero blast radius** on the daily driver.
- **Lossless for inference.** All 40 generative layers are present and load intact. Only
  optional MTP speculative decoding is forgone (not supported for this arch yet).

## Verification

- **Q2a — loads:** ✅ on llama.cpp **b9784 (`8be759e6f`)**, `GGML_CUDA_NO_PINNED=1`, GPU
  blank, `-c 4096`, no missing-tensor error with both overrides.
- **Q2b — generates:** ✅ coherent, on-task output. Prompt "Write a Python function that
  returns the nth Fibonacci number, with a docstring" produced a structured reasoning
  trace ("Thinking Process: 1. Understand the Goal… 2. Define the Fibonacci Sequence…").
- **Preset:** renders via `uv run gpumod template install-all --yes` (exit 0); both
  `--override-kv` flags confirmed in the rendered systemd unit; `GGML_CUDA_NO_PINNED=1`
  and the preflight `ExecStartPre` present.

## Setup

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 4090, 24 GB (24564 MB) |
| Host RAM | ~30 GiB usable (32 GB DDR4) |
| llama.cpp | **version 9784 (`8be759e6f`)**, `git describe` b9782-2-g8be759e6f, built 2026-06-24 |
| Model | `gaoqianshen/Qwen-AgentWorld-35B-A3B-Q4_K_M-GGUF` → `qwen-agentworld-35b-a3b-q4_k_m.gguf` (21,166,758,816 B) |
| Stability defaults | `GGML_CUDA_NO_PINNED=1` (template default); preflight RAM/VRAM checks |

(Driver / CUDA / OS per the host baseline in
[`docs/benchmarks/20260606_gemma4_26b_qat_vs_imatrix/README.md`](../../benchmarks/20260606_gemma4_26b_qat_vs_imatrix/README.md) — same machine; not re-verified this session.)

## Resolved (gpumod-qsgl.4, 2026-06-24)

- **Service-path validation — DONE.** `gpumod service start agentworld-35b-a3b-q4` (the
  preset applies the `--override-kv` fix through systemd) → `/health` OK → one
  `/v1/chat/completions` returned a coherent reasoning trace → clean stop. Confirms the
  full gpumod path, not just raw `llama-cli`.
- **Q2c — context ceiling: 65536 verified.** At `context_size=65536`, steady-state VRAM
  is **21361 MiB used / 2721 MiB free** on the 24 GB 4090 (~2.7 GB headroom). Locked at
  65536 rather than pushing to the edge: it is 2× the coding suite's `max_tokens=32768`
  (no truncation) and preserves headroom for a multi-hour run. Higher contexts are
  likely possible (KV is cheap, ~10 KB/token) but unnecessary and riskier for a long bench.
- **Runner wired.** `MODELS["agentworld-35b-a3b-q4"]` (`architecture="qwen35moe-hybrid-35B-A3B"`,
  `sampler=THINKING_CODING`) + `--model` choice added to `scripts/run_qwen36_benchmark.py`,
  with a guard test ([tests/unit/test_agentworld_benchmark_entry.py](https://github.com/jaigouk/gpumod/blob/main/tests/unit/test_agentworld_benchmark_entry.py)).
  **The bench must start the arm via the preset** (`gpumod service start`), never a raw
  `llama-server --base-url` (which would omit the override and fail to load).

## Open items

- **Benchmark (gpumod-qsgl.3).** Coding-suite comparison vs `qwen36-35b-a3b`,
  `qwen36-35b-a3b-mtp-iq4xs`, `gemma4-26b-a4b-qat-q4`. Now unblocked (runner wired,
  service path verified, Q2c locked). Heavy multi-hour run — schedule deliberately.
  Caveat for the report: AgentWorld is a world-model for environment simulation, not a
  coding model, so a low coding score is a data point, not a defect.

## Lessons / how to recognise this class in future models

1. **A successful HF `gguf-my-repo` conversion does not imply a loadable GGUF.** The
   converter can write metadata for layers whose tensors it never emits — especially
   **MTP / nextn / draft** layers that need an opt-in flag (`--mtp`).
2. **Check `block_count` vs the highest `blk.N` tensor index.** If `block_count` exceeds
   the highest emitted block, suspect a counted-but-unemitted MTP/draft layer. Confirm
   against `config.json` (`mtp_num_hidden_layers`, `num_nextn_predict_layers`).
3. **`--override-kv` is the non-destructive lever** for metadata defects on a model you
   serve under a dedicated preset — prefer it over `gguf_set_metadata.py` file edits when
   the same host runs other models.
4. **Hybrid (DeltaNet + sparse-full-attention) models have unusually cheap KV.** Don't
   size `context_size` from a uniform-attention assumption; only the periodic
   full-attention layers grow KV.
5. **Mind the `qwen35moe` ≠ `qwen3moe` naming collision** when reasoning about arch
   support and gpumod's compatibility matrix.
