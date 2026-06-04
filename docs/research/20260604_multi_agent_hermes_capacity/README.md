# Multi-Agent Capacity Spike for hermes-agent (gpumod-8xaq)

**Date:** 2026-06-04
**Ticket:** gpumod-8xaq
**Host:** RTX 4090 (24 GB), gemma4-26b-a4b-q4 + vllm-embedding-code co-tenant, llama.cpp b9500

## Question

How many concurrent agent slots can hermes-agent serve from a single shared `gemma4-26b-a4b-q4` model on one 4090, with `vllm-embedding-code` (Qwen3-Embedding-0.6B, 2.5 GB) co-running? What is the right `--parallel N` × `context_size` knee?

## Phase 1 — VRAM ceiling (complete)

### Method

Stopped `gemma4-26b-a4b-q4` gpumod service. Left `vllm-embedding-code` running (the realistic Hermes co-tenant). Launched raw `llama-server` directly with `--parallel N --cont-batching --ctx-size (N × per_slot_ctx)`, waited for `/health`, captured `nvidia-smi --query-gpu=memory.used,memory.free`, then killed the process. 5-second driver-reclaim quiesce between configs.

llama-server flags matched the production preset: q8_0 KV cache, mmproj-BF16 vision encoder loaded, flash_attn on, `--chat-template-kwargs '{"enable_thinking":true}'`, threads=16.

Raw data: [`phase1_results.tsv`](phase1_results.tsv).

### Results

| Config | N | per-slot ctx | total -c | GPU used after load | GPU free | Boot |
|---|---:|---:|---:|---:|---:|---:|
| N3 ctx128K | 3 | 131,072 | 393,216 | 22,980 MiB | **1,102 MiB** | 20s |
| N3 ctx64K  | 3 |  65,536 | 196,608 | 21,110 MiB | **2,972 MiB** | 10s |
| N5 ctx64K  | 5 |  65,536 | 327,680 | 23,038 MiB | **1,044 MiB** | 15s |
| N5 ctx32K  | 5 |  32,768 | 163,840 | 20,928 MiB | **3,154 MiB** | 10s |
| N7 ctx32K  | 7 |  32,768 | 229,376 | 22,184 MiB | **1,898 MiB** | 6s |

**All 5 configs booted successfully — no OOM at idle.** Headroom is what differs.

### Per-slot KV cost (empirical)

GPU breakdown for each config:

- Baseline (always-on): vllm-embedding-code 2.5 GB + node IDE compositor 0.4 GB + CUDA scratch ~0.5 GB = **~3.4 GB shared overhead before gemma loads**
- gemma backbone (loaded once regardless of N): model 12.97 GB + mmproj 1.14 GB = **~14.1 GB shared**
- The remainder is per-slot KV cache.

Solving `per_slot_KV ≈ (GPU_used − 3.4 − 14.1 − ~0.5 fragmentation) / N`:

| Per-slot ctx | Theoretical (q8_0: 16 MB / 1K ctx) | Empirical (this run) | Per-slot overhead beyond raw KV |
|---|---:|---:|---:|
| 128K | 2048 MB | ~1,800 MB | ~150 MB shared inefficiency captured in scratch |
| 64K  | 1024 MB | ~1,100 MB | ~75 MB per slot (slot bookkeeping, attention scratch) |
| 32K  |  512 MB | ~660 MB  | ~150 MB per slot (fixed per-slot overhead dominates at small ctx) |

Takeaway: at small per-slot context, the fixed per-slot overhead (~100-150 MB for slot state) is a non-trivial fraction of total. Doesn't matter at 64K+, matters at 32K-16K.

### Headroom interpretation

The Phase 1 result is "boots cleanly at idle". Real-world Hermes use needs to also absorb:

| Demand | Size |
|---|---:|
| One image input (e.g. 1000×562 PNG) | ~200 MiB vision-token KV |
| KV cache growth per turn (q8_0) | ~16 MiB / 1K generated tokens (already counted in per-slot ctx allocation) |
| CUDA fragmentation, mmproj per-image scratch | ~200-500 MiB margin |

So **practical free-VRAM floor is ≥ 1.5 GiB** during a vision-enabled session. By that bar:

| Config | Free at idle | Free after one image + 5K tokens response | Verdict |
|---|---:|---:|---|
| N3 ctx128K | 1102 MiB | ~700 MiB | **Risky** — single image acceptable, two concurrent image inputs likely OOM |
| N3 ctx64K  | 2972 MiB | ~2400 MiB | **Comfortable** — vision + multi-turn growth all within budget |
| N5 ctx64K  | 1044 MiB | ~600 MiB | **Risky** — same problem as N3 ctx128K, worse since 5 slots mean more image events |
| N5 ctx32K  | 3154 MiB | ~2500 MiB | **Comfortable** — but 32K is short for tool-heavy turns |
| N7 ctx32K  | 1898 MiB | ~1300 MiB | **Marginal** — single-image OK, two concurrent risky |

### Recommendation from Phase 1 alone

Two production-viable configs for a 3+ agent setup:

1. **N=3, per-slot ctx=64K** — comfortable 3 GB headroom, 64K per slot is enough for most tool-heavy coding turns IF the agent compacts older turns aggressively. Best balance for **Workflow A** (TL + Dev + QA, asymmetric load).

2. **N=5, per-slot ctx=32K** — comfortable 3.2 GB headroom but 32K is tight for tool-heavy turns. Best for chat-style multi-agent where each conversation is short (less than ~5 tool-heavy turns).

**Notable: the original spike question of "N=3 at 128K" boots, but headroom is only 1.1 GB.** That's not safe for vision-enabled multi-turn use — recommend dropping the 128K hope and settling on 64K per slot. This changes the tool-overhead analysis in the spike doc: at 64K per slot, a Dev turn with 25-30K tokens of tool overhead occupies ~half the context after one turn. Aggressive compaction (drop oldest tool-result turns) becomes mandatory, not optional.

### Open questions for Phase 2

Phase 1 establishes which configs FIT. Phase 2 establishes whether they're FAST ENOUGH:

- At N=3 ctx=64K, what's the per-slot TPS when all 3 are generating? (Workflow B floor case)
- At N=5 ctx=32K, does cont-batching's idle-borrowing cover for asymmetric loads, or does the per-slot floor become ~25 TPS even in Workflow A?
- Does mmproj vision encoder become a contention point under concurrent image inputs across slots?
- Tool-storm: rapid context churn — does cont-batching slot eviction stay clean?

### Production state restored after Phase 1

`hermes-agent` mode restarted with the original `gemma4-26b-a4b-q4` (single-slot, 128K ctx) preset — no production state changed by Phase 1.

## Phase 2 — pending

Synthetic load runner not yet written. Estimated effort: ~2 hours runner + 4 × 10 min bench runs.
