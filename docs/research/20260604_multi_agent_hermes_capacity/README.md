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

## Phase 2 — synthetic workload bench (complete)

### Method

Built a stdlib-only Python workload runner ([`phase2_workload_runner.py`](phase2_workload_runner.py)) that spawns N concurrent threads, each representing one "agent slot" hitting `/v1/chat/completions` with a configurable role profile. Each turn prepends a synthetic `[TOOL RESULT]` blob of code-like filler text to the user message to exercise the same KV-cache pressure path as real tool use without requiring a tool executor.

Orchestrators ([`phase2_orchestrator.sh`](phase2_orchestrator.sh), [`phase2_followup_textonly.sh`](phase2_followup_textonly.sh), [`phase2_n3_200k.sh`](phase2_n3_200k.sh), [`phase2_n3_128k.sh`](phase2_n3_128k.sh)) boot raw `llama-server` at each config, wait for `/health`, capture VRAM, run the workload for 600 s, kill, quiesce 8 s, next config.

#### Safeguards (added after initial dry-run)

The original orchestrator launched `llama-server` directly, bypassing all gpumod preflight. With `MemAvailable` at 14 GiB (between the 12 GiB unrecoverable-hang and 18 GiB safe-floor thresholds from gpumod-x7rv), this re-exposed the cudaHostAlloc-freeze class. Retrofitted before the first config booted:

| Safeguard | Threshold | Action |
|---|---|---|
| `GGML_CUDA_NO_PINNED=1` env var | always | disables `cudaMallocHost`, eliminates the freeze class (gpumod-56md) |
| RAM preflight | `MemAvailable ≥ 13 GiB` | refuse boot if below |
| VRAM preflight | `free VRAM ≥ 400 MiB` after boot | skip workload if below |
| Side watchdog | `MemAvailable < 8 GiB` OR `free VRAM < 200 MiB` sustained 15 s | SIGTERM orchestrator |

No watchdog or preflight ever triggered during Phase 2. All 9 configs completed cleanly.

### Workload profiles

Defined in [`phase2_workload_runner.py`](phase2_workload_runner.py) under `PROFILES`:

| Role | Cadence | max_tokens | Synthetic tool overhead | Thinking |
|---|---|---:|---:|---|
| TL | 1 req / 120 s | 1500 | 400 tokens | on |
| Dev | continuous | 8000 | 15000 tokens | on |
| QA | 1 req / 60 s | 2500 | 5000 tokens | on |
| Research | continuous | 5000 | 8000 tokens | on |
| ToolStorm | continuous | 500 | 800 tokens | off |

`Workflow A` = `TL,Dev,QA` (asymmetric coding team, idle-borrowing case).
`Workflow B` = `3xResearch` (uniform continuous load, floor case).

### Results — all configs

Raw JSON: [`phase2_results/`](phase2_results/). Re-aggregate with [`phase2_aggregate.py`](phase2_aggregate.py).

| Config | Slots | Workload | Agg TPS | Per-call TPS | Dev p50 | Dev p95 | Free VRAM at boot |
|---|---:|---|---:|---:|---:|---:|---:|
| 01 N=1 ctx=128K | 1 | Dev (baseline) | 117.2 | 109.5 | 73 s | 73 s | 4,164 MiB |
| 02 N=3 ctx=64K | 3 | TL,Dev,QA | 142.3 | 69.7 | 93 s | 114 s | 2,972 MiB |
| 03 N=3 ctx=64K | 3 | 3×Research | **180.2** | 58.7 | — | — | 2,972 MiB |
| 04 N=5 ctx=32K | 5 | TL,2×Dev,2×QA | 181.2 | 42.9 | 158 s | 166 s | 3,595 MiB |
| 05 N=5 ctx=32K | 5 | 3×Research+Dev+TL | 169.3 | 38.8 | 173 s | 183 s | 3,595 MiB |
| 06 N=3 ctx=64K | 3 | 3×ToolStorm | 75.6 | 25.7 | 2 s p50 | 3 s p95 | 3,413 MiB |
| 07 N=3 ctx=256K | 3 | TL,Dev,QA | **10.5** | 10.1 | — (0 turns ok) | — | 1,077 MiB (text-only) |
| 08 N=3 ctx=200K | 3 | TL,Dev,QA | 78.9 | 29.2 | 293 s | 293 s | 971 MiB (text-only) |
| 09 N=3 ctx=128K | 3 | TL,Dev,QA | 132.1 | 68.6 | 96 s | 122 s | 1,823 MiB (text-only) |

### Headline finding 1 — the N=3 Workflow A scaling curve

| Per-slot ctx | Agg TPS | Per-call TPS | Dev p50 | Verdict |
|---|---:|---:|---:|---|
| 64K  | 142 | 70 | 93 s | Throughput max — narrow win |
| **128K** | **132** | **69** | **96 s** | ⭐ **Sweet spot** — 2× per-slot context for only 7% TPS cost |
| 200K | 79  | 29 | 293 s | -45% TPS, 3× Dev latency — degraded |
| 256K | 10  | — | — | **Collapsed** — Dev never finished a turn in 600 s |

### Headline finding 2 — the 256K throughput collapse (the surprise)

VRAM ceiling and TPS ceiling are **not the same wall**. At N=3 ctx=256K the model fits (1,077 MiB VRAM free), boots cleanly, exchanges HTTP requests — but aggregate throughput collapses by 13× vs ctx=64K, and the Dev role (8000-token turns) never finishes a single turn in the 600 s window.

Likely root cause: llama.cpp's attention compute scales with **allocated** slot context, not used context, so each generated token pays a per-token compute penalty proportional to slot size even when the conversation occupies a small fraction. Flash-attention with quantized KV may not optimize for sparse slots. The cliff is steep but continuous: 64K → 128K → 200K → 256K shows smooth degradation, then 256K falls off entirely.

Recommend filing a follow-up spike to confirm the mechanism (compare attention kernel performance at different `--ctx-size` values with otherwise-identical prompts) before deciding whether 256K is reachable with kernel-config changes or is fundamentally bandwidth-bound.

### Headline finding 3 — Workflow B (uniform continuous) is the highest-yield case

`3×Research` at N=3 ctx=64K hit **180 TPS aggregate** — the highest of any config. Per-slot rate drops from 109 (single-slot baseline) to 58 (~half), but total aggregate grows 54%. Symmetric continuous load is the cont-batching best case: no idle slots to optimize around, no role-mix scheduling overhead.

For research / multi-document-synthesis workloads, N=3 ctx=64K with three identical Research agents is the right config — even better than asymmetric Workflow A.

### Headline finding 4 — N=5 doesn't pay back

N=5 ctx=32K stretched Workflow A: 181 TPS aggregate (essentially tied with N=3 Workflow B). 5 slots adds capacity but **not throughput** — per-slot TPS drops to 43, Dev p95 ≈ 166 s (vs 114 s at N=3 ctx=64K). The 4th and 5th slots only make sense when concurrent-agent count itself is the bottleneck (not aggregate throughput) and Dev latency is acceptable.

Mixing `3×Research + Dev + TL` at N=5 ctx=32K gave **169 TPS** — *worse* than pure 3×Research at N=3. Mixing burst (TL) + steady (Research) + heavy (Dev) on cont-batching is worse than either uniform load alone.

### Headline finding 5 — Tool-storm churn is fine

`3×ToolStorm` at N=3 ctx=64K: 1052 turns in 600 s, p50 turn 1.5 s, p95 3.1 s, per-call 26 TPS. Cont-batching handles rapid small-turn churn cleanly — no slot-eviction errors, no latency spikes. Useful for tool-call-heavy agents that send many short responses.

### Recommended production configs

| Use case | Preset | Mode |
|---|---|---|
| **Default coding** (TL + Dev + QA team) | `gemma4-26b-a4b-q4-multi3-128k` (text-only, `--parallel 3 --cont-batching --ctx-size 393216`) | `hermes-agent-multi` |
| **Research / chat multi-agent** | Same preset, three Research-style agents | `hermes-research-multi` |
| **Heavy concurrent agent count** (5-agent chat, latency-tolerant) | `gemma4-26b-a4b-q4-multi5-32k` | `hermes-agent-multi5` |
| **Vision-enabled single-agent** (existing) | `gemma4-26b-a4b-q4` (vision, `--parallel 1 --ctx-size 131072`) | `hermes-agent` (unchanged) |

The multi-slot presets are **text-only** — they drop `--mmproj` to free ~1.1 GiB VRAM that would otherwise eat the 128K headroom. Vision-enabled multi-slot would need to fall back to ctx=64K per slot.

### Open questions / follow-up tickets

1. **256K throughput-collapse root cause** — file a spike to confirm attention-compute scaling and check whether `--no-context-shift`, different `--batch-size`, or alternative attention backends restore throughput at 256K.
2. **N=2 ctx=256K viability** — projected to fit at ~22.4 GiB; would test whether the throughput collapse is a per-slot or aggregate-bandwidth effect.
3. **Workflow B at ctx=128K** — measure aggregate TPS for `3×Research` at the recommended production ctx to confirm the 180 TPS finding holds at the larger per-slot context.

### Production state restored after Phase 2

`hermes-agent` mode restarted with the original `gemma4-26b-a4b-q4` (single-slot, 128K ctx, vision-enabled) preset. Multi-slot presets and modes proposed but **not yet created** — that's Phase 4 work.
