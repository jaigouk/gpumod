# gpumod-scie — Embedding driver swap: vLLM → llama.cpp `--embedding`

**Date:** 2026-06-06
**Status:** Implemented. `gguf-embedding-code` preset shipped; `hermes-agent`,
`code`, and `finetuning` modes swapped. vLLM preset retained as fallback.
**Affected modes:** `hermes-agent`, `code`, `finetuning`
**Unaffected:** `rag`, `hacker` (still reference `vllm-embedding-code`)

## TL;DR

The same model, Qwen3-Embedding-0.6B, now runs under llama.cpp's
`llama-server --embedding --pooling last` instead of vLLM's pooling runner.
On this host the swap reclaims ~1.7 GB host RAM and ~0.7 GB VRAM per
embedding-loaded mode, shrinks cold start from ~60-90 s to ~2.5 s, and is
behaviourally indistinguishable for the workloads that actually call the
endpoint today (`/v1/embeddings` returns the same 1024-dim L2-normalised
vector). Throughput under high concurrency is lower than vLLM would deliver,
but no caller drives the concurrency where that gap matters.

The decision is driven entirely by the **30 GiB host RAM constraint** on
the benchmark-host box after the b4t multi-agent collapse (`k3s-setup-b4t`).
vLLM's ~2.3 GB fixed CUDA/Python overhead per process is wasted on a 0.6 B
one-shot pooling workload; llama.cpp is a single C++ process and pays
none of it.

## The model: Qwen3-Embedding-0.6B

| Property | Value |
|---|---|
| HF repo (vLLM) | `Qwen/Qwen3-Embedding-0.6B` |
| HF repo (GGUF) | `Qwen/Qwen3-Embedding-0.6B-GGUF` |
| Local GGUF file | `~/bin/Qwen3-Embedding-0.6B-Q8_0.gguf` (639 MB on disk) |
| Parameters | 0.6 B |
| Native embedding dim | 1024 |
| Training context | 32,768 tokens |
| Pooling | last-token (requires `--pooling last`) |
| Matryoshka-trained | yes — supports truncating + L2-renormalising to any prefix |
| Quantisation chosen | Q8_0 — quality loss below noise for an embedder; ~640 MB weights |
| L2-normalisation | server returns already-L2-normalised vectors (`‖v‖₂ = 1.0`) |

**Why this model.** Qwen3-Embedding-0.6B was already the deployed embedder
under vLLM (`presets/embedding/vllm-embedding-code.yaml`). It is the
smallest member of the Qwen3-Embedding family; matches the embedding
space used by everything indexed against it; and at Q8_0 the quality loss
versus fp16 is well below what we can measure on the workloads that use
it. There was no reason to change the model — only the driver.

**Why last-token pooling.** Qwen3-Embedding is trained with last-token
pooling. Using mean or CLS pooling instead would silently degrade
retrieval quality. `--pooling last` is therefore mandatory in the preset;
encoded in the preset filename family (`gguf-embedding-code`) to discourage
copy-pasting it for a different model later.

## The driver decision

### The constraint that forced this

The 30 GiB host on b4t cannot afford fixed-cost overhead on auxiliary
services. vLLM ≥ 0.6 splits the API server from the engine worker into
two processes, each carrying the full CUDA driver context + Python
interpreter + `transformers` / `vllm` imports. Empirically that bill is
~2.3 GB RSS regardless of model size:

| Process | RSS |
|---|---|
| `vllm` API server | ~842 MB |
| `VLLM::EngineCore` worker | ~1.47 GB |
| **Total** | **~2.31 GB** |

This is fixed cost per CUDA process. A 0.6 B model contributes only a
small fraction of it; switching to a hypothetically smaller embedder
would not reduce it. vLLM's reason for existing — PagedAttention +
continuous batching across many concurrent decode streams — buys
nothing for a one-shot forward pass through a pooling model.

### What llama.cpp gives instead

`llama-server` is a single C++ process. No Python, no API/engine split,
no `transformers` import. Loading a 0.6 B GGUF onto CUDA pays the CUDA
driver context (~1.7 GB VRAM) and almost nothing else.

| Metric | vLLM (prior) | llama.cpp (measured 2026-06-06) | Δ |
|---|---|---|---|
| RSS (steady-state) | ~2.31 GB | **421 MB** | **−1.89 GB (5.5× lower)** |
| VRAM (steady-state) | ~2.55 GB | **1.81 GB** | **−0.74 GB** |
| Cold start (model loaded) | 60-90 s | **2.5 s** | **24-36× faster** |
| Processes | 2 | 1 | −1 |
| Output dim | 1024 | 1024 | identical |
| L2-normalisation | server-side | server-side | identical |
| OpenAI `/v1/embeddings` schema | yes | yes | drop-in |

Measurements taken on the benchmark-host host with no other GPU-resident
services loaded (mode = `blank`, baseline 15 MiB residual). See the
"Measurement procedure" section.

### What we give up

| Capability | vLLM | llama.cpp `--embedding` | Why we accept the gap |
|---|---|---|---|
| PagedAttention KV reuse | yes | no | No KV-cache reuse in pooling forward passes anyway |
| Continuous batching across decode streams | yes | n/a | Embedding is a single forward pass — no decode |
| Server-side matryoshka slicing via `hf_overrides` | yes | no | Native 1024-dim works for every current caller |
| `dimensions` parameter on `/v1/embeddings` | accepted (via `hf_overrides`) | ignored | Only GrepAI ever sent it; GrepAI deprecated 2026-04-19 |
| Concurrent throughput at >10 RPS | higher | lower | No caller drives that load on this host |

## Caller audit (why no caller breaks)

Performed 2026-06-06 across `~/AI`, `~/Jaigouk`, `~/.hermes/hermes-agent`,
`~/Alty`. Hits filtered to source + config (no chat transcripts, no journal
files):

| Caller | Status | Reason |
|---|---|---|
| Open-WebUI (k8s, `apps/base/open-webui`) | unaffected | Uses default OpenAI client, no `dimensions` parameter, accepts whatever dim the server returns |
| Honcho | unaffected | `src/embedding_client.py` only supports openai / openrouter / gemini; never called `:8210`. `EMBED_MESSAGES=false` on this stack |
| GrepAI | unaffected | Deprecated 2026-04-19, no longer deployed |
| Hermes Agent (`~/.hermes/hermes-agent`) | unaffected | No direct embedding HTTP calls in source; proxy adapters only forward `/embeddings` path, they don't originate calls |
| qmd | unaffected | Uses its own GGUF stack via `node-llama-cpp`, independent of `:8210` |

**Matryoshka note for the future.** llama.cpp returns the full 1024-dim
vector; vLLM with `hf_overrides: '{"matryoshka_dimensions": [1024]}'` also
returned 1024 (the native dim). The two are observationally identical for
callers that don't pass `dimensions`. A future caller that wants <1024-dim
embeddings must slice + L2-renormalise client-side; it cannot ask the
server to do it.

**Numerical drift note.** Q8_0 quantisation produces vectors that are
slightly different from vLLM's fp16 output for the same input. Cosine
similarity is robust to this, and existing indexed vectors keep retrieving
sensibly, but a full reindex would eliminate the drift entirely. Practical
impact judged invisible; not done.

## Preset

`presets/embedding/gguf-embedding-code.yaml`:

```yaml
id: gguf-embedding-code
driver: llamacpp
port: 8210                       # same port — drop-in for callers
vram_mb: 800                     # down from 2500 under vLLM
context_size: 4096
model_path: $HOME/bin/Qwen3-Embedding-0.6B-Q8_0.gguf
unit_vars:
  n_gpu_layers: -1
  flash_attn: true
  extra_args: "--embedding --pooling last --parallel 4 --threads 8"
```

The rendered systemd unit inherits `Environment="GGML_CUDA_NO_PINNED=1"`
from the llamacpp template — eliminates the `cudaHostAlloc` driver-hang
class (see `20260525_oom_protection_findings/FINDINGS.md`).

**Two log signals to know about**, both benign:

1. `embeddings enabled with n_batch (2048) > n_ubatch (512), setting
   n_batch = n_ubatch = 512` — llama.cpp auto-clamps batch size for
   embedding mode. No fix needed.
2. `n_ctx_seq (1024) < n_ctx_train (32768)` — `--ctx-size 4096` divided
   by `--parallel 4` gives 1024 tokens of input per slot. Open-WebUI's
   default RAG chunk is ~400 tokens, well under this ceiling. If a
   future caller needs to embed longer passages, lower `--parallel` to
   2 (→ 2048 tokens per slot) or raise `--ctx-size`.

## Measurement procedure

Reproducible against this preset:

1. Switch to `blank` mode: `gpumod mode switch blank`.
2. Confirm baseline: `nvidia-smi --query-gpu=memory.used,memory.free
   --format=csv,noheader` → expect `15 MiB, ~24067 MiB`.
3. Start the service: `gpumod service start gguf-embedding-code`.
4. Poll `curl -sf http://127.0.0.1:8210/health` until 200.
5. Sample RSS: `ps -eo rss,comm --sort=-rss | grep llama-server`.
6. Sample VRAM: `nvidia-smi --query-gpu=memory.used,memory.free
   --format=csv,noheader`.
7. Functional check:
   ```bash
   curl -s -X POST http://127.0.0.1:8210/v1/embeddings \
     -H 'Content-Type: application/json' \
     -d '{"model":"qwen3-embedding-0.6b","input":"hello"}'
   ```
   Verify the returned `data[0].embedding` has length 1024 and an L2
   norm of 1.0.
8. Cosine sanity (three inputs, two related + one unrelated): the
   related-pair cosine should be visibly higher than either related/
   unrelated cosine. (Observed: 0.564 vs 0.348 / 0.357.)
9. Stop: `gpumod service stop gguf-embedding-code`.

## What stays under vLLM

The `vllm-embedding-code` preset is intentionally kept for two reasons:

1. **Fallback** — if a llama.cpp regression is found, modes can be reverted
   by changing one line per file.
2. **Active use** — the `rag` and `hacker` modes still reference it. Those
   modes pair the embedder with `vllm-embedding` (Qwen3-VL-Embedding-2B,
   port 8200) and `qwen3-coder` respectively; the consistency of running
   one driver per mode mattered more than the RAM saving. Revisit if
   either mode comes under RAM pressure too.

The Qwen3-VL-Embedding-2B service (`vllm-embedding`, port 8200) is **not**
migrated. It is a multimodal embedder; llama.cpp embedding mode does not
support the image-tower path of the Qwen3-VL family.

## Related decisions

- `20260525_oom_protection_findings/FINDINGS.md` — why
  `GGML_CUDA_NO_PINNED=1` is the default for all llamacpp services
  (eliminates the freeze class; 0.28% TPS cost).
- `20260604_multi_agent_hermes_capacity/` — the b4t RAM constraint that
  forced this swap (multi-slot Gemma collapsed to single-slot; freed VRAM
  was reclaimed by other consumers, leaving no slack for vLLM overhead).
