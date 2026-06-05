# Slot persistence for multi-agent Gemma 4 (gpumod-8viu)

**Date:** 2026-06-05
**Ticket:** gpumod-8viu
**Depends on:** [gpumod-8xaq](../20260604_multi_agent_hermes_capacity/README.md) (multi-slot capacity baseline)
**Host:** RTX 4090 (24 GB), llama.cpp b9500

## Question

`gpumod-8xaq` Phase 2 productionised `gemma4-26b-a4b-q4-multi` (N=3 cont-batching, 128K per slot, 132 TPS aggregate). It works fine for live concurrent agents thanks to the radix-tree prefix cache. But the "agent disconnects and reconnects later" pattern was unaddressed — KV is in-memory only, and another agent claiming the slot evicts the original. When the first agent returns, it pays a full re-prefill (seconds to tens of seconds at long conversations).

Does llama.cpp's [`--slot-save-path`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) feature solve this on Gemma 4 in our multi-slot config?

## Phase 2 v1 — the freeze incident

Initial smoke test included `--swa-full` based on the [persistent-KV tutorial](https://github.com/ggml-org/llama.cpp/discussions/20572) recommendation for SWA models. At ctx=393216 (3 × 128K) with Gemma 4's 30 layers, this forces every layer to allocate full-ctx KV instead of the 1024-token sliding window. VRAM blew past 24 GiB during cache allocation, triggering the cudaHostAlloc-class freeze (gpumod-x7rv pattern) despite `GGML_CUDA_NO_PINNED=1`. **Hard reboot required.** The kernel/journal showed no OOM signal — only the llama-server log captured the warning line:

```
W llama_kv_cache_iswa: using full-size SWA cache
              (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
```

**Lesson:** `--swa-full` is incompatible with Gemma 4 at any meaningful multi-slot ctx. Do NOT add to the preset.

## Phase 2 v2 — staged smoke test without `--swa-full`

Redesigned to fail-fast at minimal scale. Script: [`phase2_smoke_v2.sh`](phase2_smoke_v2.sh).

| Phase | Config | Result |
|---|---|---|
| A | `--parallel 1 --ctx-size 16384` (tiny KV) | ✅ PASS — saved 4.9 MB, restored, model recalled the magic word |
| B | `--parallel 3 --ctx-size 393216` (production scale) | ⚠ MISLEADING — saved **44 bytes** (empty slot); follow-up "passed" via prefix-cache rather than real restore |

Phase B's anomaly: the chat helper didn't pin to a slot. The conversation landed on slot 1 or 2 (whichever was idle), so saving slot 0 captured nothing. The follow-up call still recalled the magic word — but because the prefix cache had the original conversation in its actual slot, not because restore worked.

## Phase 2 v3 — slot-pinned multi-slot save/restore

Script: [`phase2_smoke_v3_slot_pinned.sh`](phase2_smoke_v3_slot_pinned.sh).

Added `"id_slot": 0` to every chat request to pin all turns to the same slot. The `id_slot` parameter is undocumented on `/v1/chat/completions` but [passes through](https://github.com/ggml-org/llama.cpp/discussions/22354) from the native `/completion` endpoint.

| Step | Result |
|---|---|
| Conv A → slot 0 | `"I have memorised the word gargleblast."` |
| Save slot 0 | `n_saved=41`, `n_written=4,908,808 bytes`, `save_ms=3.8` |
| Conv B → slot 0 (evict) | done |
| Restore slot 0 | `n_restored=41`, `n_read=4,908,808`, `restore_ms=2.3` |
| Follow-up to A → slot 0 | `"The magic word was gargleblast."` ✅ |

Server-side timing on the follow-up: **1,223 tok/s prompt-eval rate** vs **387 tok/s** on the cold prefill — 3× speedup confirms KV reuse from the restored state (not prefix-cache fallback).

## Findings

| Question | Answer |
|---|---|
| Does `--slot-save-path` work on b9500 with `gemma4-26b-a4b-q4-multi`? | ✅ Yes |
| Does it require `--swa-full`? | ❌ No — the tutorial overstated. Works without it on Gemma 4. |
| Is `--swa-full` safe on Gemma 4 multi-slot? | ❌ No — triggers host freeze. NEVER add. |
| Does `id_slot` pin work on `/v1/chat/completions` (b9500)? | ✅ Yes, undocumented passthrough |
| VRAM cost of `--slot-save-path`? | None — measured 1,823 MiB free identical to no-flag baseline |
| Disk cost per saved token? | ~120 KB (full-precision bf16 dump of K + V across 30 layers) |
| Save / restore latency? | 3.8 ms save, 2.3 ms restore for 41 tokens (sub-frame for any realistic conversation size) |
| Does restore actually accelerate follow-ups? | ✅ Confirmed — 3× faster prompt-eval rate on the post-restore turn |

## Disk-cost ceiling per agent

At ~120 KB/token: a 10K-token conversation = 1.2 GB on disk. A 50K-token tool-heavy Dev conversation = 6 GB. For 3 long-lived persistent agents, plan for **~5-20 GB of slot-cache disk** depending on conversation length. Cleanup policy: delete `.bin` files older than N days, or per-session expiry.

## Productionization (Phase 4)

Applied to `presets/llm/gemma4-26b-a4b-q4-multi.yaml`:

```yaml
extra_args: "--parallel 3 --cont-batching ... --slot-save-path $HOME/.cache/llama-slots ..."
```

**Host setup requirement:** `mkdir -p ~/.cache/llama-slots` (one-time). Documented in the preset header; the systemd unit will fail loudly if missing.

Wrapper pattern documented in [`modes/hermes-agent.yaml`](https://github.com/jaigouk/gpumod/blob/main/modes/hermes-agent.yaml) comment block:

```bash
# Before reconnect:
curl -X POST http://localhost:7109/slots/0?action=restore \
  -H 'Content-Type: application/json' \
  -d '{"filename":"agent_A.bin"}'

# Chat (with id_slot pin):
curl -X POST http://localhost:7109/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"id_slot": 0, "messages": [...], "max_tokens": 1500}'

# After response:
curl -X POST http://localhost:7109/slots/0?action=save \
  -H 'Content-Type: application/json' \
  -d '{"filename":"agent_A.bin"}'
```

Same pattern is now in the Hermes setup doc in the k3s-setup repo so cron jobs and Subagents can integrate it.

## Open follow-ups

1. **Stress test concurrent save/restore** — what happens if 3 agents simultaneously call `/slots/{id}?action=save` on different slots? Not tested here.
2. **Disk-cache cleanup policy** — no automated expiry; ops responsibility for now.
3. **Hermes skill wrapper** — file a small ticket to write a Hermes pre/post chat hook that does the restore + save round-trip automatically using a per-conversation filename derived from session id. Out of scope for gpumod-8viu (lives in `~/.hermes/`, not this repo).
4. **Cross-restart slot persistence** — does the slot file survive a llama-server restart and still restore correctly? Presumably yes (it's just bytes on disk) but not explicitly tested.

## Sources

- [llama.cpp server README — slot save/restore endpoints + cache_prompt](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Discussion #20572 — Persistent KV cache per session with llama-server hooks](https://github.com/ggml-org/llama.cpp/discussions/20572) (the tutorial that mis-recommended `--swa-full` for our case)
- [Discussion #22354 — KV cache reuse for OpenCode agent switching](https://github.com/ggml-org/llama.cpp/discussions/22354) (the `id_slot` passthrough confirmation)
- [PR #13194 — full-size SWA cache flag context](https://github.com/ggml-org/llama.cpp/pull/13194)
- [gpumod-x7rv research — cudaHostAlloc freeze root cause](https://github.com/jaigouk/gpumod/blob/main/docs/research/20260525_oom_protection_findings/FINDINGS.md)
