---
title: Multi-Agent with Continuous Batching
description: Serve multiple concurrent AI agents from one model with --parallel slots, cont-batching, slot pinning, and persistent per-agent KV cache.
---

# Multi-Agent with Continuous Batching

*Learn how to serve 3 concurrent agents from a single 24 GB GPU using the
multi-slot preset — including slot pinning and saving an agent's KV cache to
disk so it survives disconnects.*

This guide uses `gemma4-26b-a4b-q4-multi`
([preset](https://github.com/jaigouk/gpumod/blob/main/presets/llm/gemma4-26b-a4b-q4-multi.yaml)),
the production multi-slot configuration chosen by the
[multi-agent capacity spike](../research/20260604_multi_agent_hermes_capacity/README.md)
(gpumod-8xaq).

## Slots are not agents

`--parallel 3` makes llama-server allocate **three independent KV-cache
slots** — it does not create three agents. An "agent" is just any HTTP
client sending requests to the same endpoint:

- Run 3 separate client processes (3 Claude Code tabs, 3 Hermes sessions,
  3 aider invocations…), each pointed at the **same** base URL
  `http://localhost:7109/v1`.
- The cont-batching scheduler routes each incoming request to a free slot,
  first-come-first-served. A 4th concurrent request queues (milliseconds to
  seconds) until a slot frees.
- The server is stateless across requests — every `/v1/chat/completions`
  POST carries the full `messages` history. Multi-turn conversations stay
  fast anyway because the radix-tree **prefix cache** matches the new
  request against a slot's existing KV.

## When to use the multi-slot preset

| Situation | Preset |
|---|---|
| One interactive session, may send images | `gemma4-26b-a4b-q4` (single slot, vision) |
| 2–3 concurrent agents (agent team, cron + chat + subagent) | `gemma4-26b-a4b-q4-multi` (N=3, text-only) |
| Image inputs **and** multiple agents | Not on a 24 GB card — the multi preset drops `--mmproj` to free ~1.1 GiB; fall back to single-slot |

Key numbers from the capacity spike (see the
[research README](../research/20260604_multi_agent_hermes_capacity/README.md)
for the full data):

- **128K per slot is the sweet spot** — 132 TPS aggregate with 3 active
  agents, only 7% below the 64K-per-slot maximum, at 2× the context.
- **Do not raise per-slot context above 128K.** Throughput degrades at 200K
  (79 TPS) and collapses at 256K (10 TPS — a heavy agent never finishes a
  turn), even though the model still *fits* in VRAM.
- **N=5 doesn't pay back** — it adds slots but not aggregate throughput,
  and forces 32K per slot.

!!! danger "Never add `--swa-full` to a Gemma 4 multi-slot preset"
    `--swa-full` forces every layer to allocate full-context KV instead of
    Gemma 4's 1024-token sliding window. At multi-slot context sizes this
    blows past the 24 GiB VRAM ceiling **during cache allocation** and
    triggers a silent host freeze that requires a hard reboot — no OOM log,
    no recoverable error (see [Host stability](host-stability.md)).
    Slot save/restore works correctly **without** it
    ([gpumod-8viu incident report](../research/20260605_slot_persistence/README.md)).

## Prerequisites

- gpumod set up and a service started before —
  [Run your first service](../getting-started/first-service.md)
- ~22 GB of free VRAM (the multi preset + its embedding co-tenant)
- One-time host setup for slot persistence:

```bash
mkdir -p ~/.cache/llama-slots
```

(The systemd unit fails loudly at start if this directory is missing.)

## Steps

### 1. Switch to the multi-agent mode

The `hermes-agent` mode bundles `gemma4-26b-a4b-q4-multi` with a small
embedding service:

```bash
gpumod mode switch hermes-agent
```

Verify:

```bash
gpumod status | head -3
```

Expected output:

```
GPU: NVIDIA GeForce RTX 4090  VRAM: 24564 MB
Mode: hermes-agent
```

### 2. Confirm the slots exist

```bash
curl -s http://localhost:7109/props | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print('slots:', d['total_slots'], '| ctx per slot:', d['default_generation_settings']['n_ctx'])"
```

Expected output:

```
slots: 3 | ctx per slot: 131072
```

Three slots, 128K context each (the preset passes `--ctx-size 393216`;
llama-server divides it by `--parallel`).

### 3. Send a slot-pinned chat request

Adding `"id_slot": 0` pins the request to slot 0. This is an undocumented
but functional passthrough on `/v1/chat/completions` — you need it whenever
an agent must land on a **specific** slot (e.g. to pair with save/restore
below):

```bash
curl -s http://localhost:7109/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "id_slot": 0,
    "messages": [{"role": "user", "content": "Remember the magic word: tutorialdemo. Reply with one short sentence confirming you memorised it."}],
    "max_tokens": 1500
  }'
```

Expected output (abbreviated — the response also contains a
`reasoning_content` field because the preset enables thinking):

```json
{
  "choices": [{"message": {"content": "I have memorized the magic word: tutorialdemo."}, "finish_reason": "stop"}],
  "usage": {"completion_tokens": 212, "prompt_tokens": 35, "total_tokens": 247}
}
```

Note `completion_tokens: 212` for a one-sentence answer — the thinking
tokens count against `max_tokens`. Always send `max_tokens >= 1500` on this
preset (see [OpenAI-compatible clients](openai-clients.md)).

Without `id_slot`, requests are still served fine — they just land on
whichever slot is free, which is what you want for stateless concurrent
clients.

### 4. Save the slot's KV cache to disk

The preset enables `--slot-save-path`, which exposes per-slot save/restore
endpoints. Persist slot 0's state:

```bash
curl -s -X POST 'http://localhost:7109/slots/0?action=save' \
  -H 'Content-Type: application/json' \
  -d '{"filename": "agent_A.bin"}'
```

Expected output:

```json
{"id_slot":0,"filename":"agent_A.bin","n_saved":246,"n_written":29448948,"timings":{"save_ms":12.457}}
```

246 cached tokens → ~29 MB on disk (~120 KB/token, full-precision KV dump).
Budget **~5–20 GB of disk for 3 long-lived agents** and add a cleanup
policy — a 50K-token tool-heavy conversation is ~6 GB.

### 5. Restore and resume

After the agent disconnects (and possibly another client evicted slot 0),
restore before reconnecting:

```bash
curl -s -X POST 'http://localhost:7109/slots/0?action=restore' \
  -H 'Content-Type: application/json' \
  -d '{"filename": "agent_A.bin"}'
```

Expected output:

```json
{"id_slot":0,"filename":"agent_A.bin","n_restored":246,"n_read":29448948,"timings":{"restore_ms":6.216}}
```

Then continue the conversation (same `id_slot: 0`, full message history):

```bash
curl -s http://localhost:7109/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "id_slot": 0,
    "messages": [
      {"role": "user", "content": "Remember the magic word: tutorialdemo. Reply with one short sentence confirming you memorised it."},
      {"role": "assistant", "content": "I have memorized the magic word: tutorialdemo."},
      {"role": "user", "content": "What was the magic word? Answer in one short sentence."}
    ],
    "max_tokens": 1500
  }'
```

Expected output (abbreviated):

```json
{"choices": [{"message": {"content": "The magic word was tutorialdemo."}}]}
```

The restored turn skips re-prefilling the saved tokens — the validation run
measured a **3× faster prompt-eval rate** after restore versus a cold
prefill ([gpumod-8viu](../research/20260605_slot_persistence/README.md)).

### 6. Wrap it into an agent lifecycle

The production pattern (also documented in the
[`hermes-agent` mode file](https://github.com/jaigouk/gpumod/blob/main/modes/hermes-agent.yaml)):

```bash
# Before reconnect (404 on the very first session is fine):
curl -sf -X POST 'http://localhost:7109/slots/0?action=restore' \
  -H 'Content-Type: application/json' -d '{"filename":"agent_A.bin"}' || true

# Every chat turn — keep the id_slot pin:
curl -X POST http://localhost:7109/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"id_slot": 0, "messages": [...], "max_tokens": 1500}'

# After each response:
curl -sf -X POST 'http://localhost:7109/slots/0?action=save' \
  -H 'Content-Type: application/json' -d '{"filename":"agent_A.bin"}'
```

Use one filename (and one slot id) per persistent agent: `agent_A.bin` on
slot 0, `agent_B.bin` on slot 1, `agent_C.bin` on slot 2.

## What happened

One model, loaded once (~14 GB), serves three independent conversations:
each slot holds its own 128K-token KV cache (~1.8 GB per slot at q8_0), and
the continuous-batching scheduler packs all active generations into shared
forward passes — which is why 3 concurrent agents reach 132 TPS aggregate
instead of one agent's 109 TPS. Slot save/restore dumps a slot's KV tensors
to disk so a returning agent pays a ~6 ms restore instead of a multi-second
re-prefill, and `id_slot` pinning guarantees the restore and the follow-up
turn land on the same slot.

## Next steps

- [OpenAI-compatible clients](openai-clients.md) — point Hermes, Claude
  Code, aider, or Continue.dev at this endpoint
- [Modes deep dive](modes-deep-dive.md) — what `mode switch` does under the
  hood
- [Capacity spike research](../research/20260604_multi_agent_hermes_capacity/README.md)
  — the full N×ctx measurement matrix
- [Slot persistence research](../research/20260605_slot_persistence/README.md)
  — validation details and the `--swa-full` incident
