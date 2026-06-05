---
title: Pick a Mode for Your Workload
description: Decision guide for the bundled gpumod modes — by use case and by how much VRAM your GPU has.
---

# Pick a Mode for Your Workload

*Learn which bundled mode fits your use case and your GPU, before you
switch into it.*

A mode is a named bundle of services that gpumod starts and stops together
(see [Modes deep dive](../user-guide/modes-deep-dive.md) for how switching
works). The bundled modes were sized for a 24 GB card; this page helps you
pick one — or decide to build your own.

## Prerequisites

- gpumod initialized (`gpumod init`) — see [Getting Started](index.md)
- Know your VRAM: `gpumod status` prints it on the first line

## Decision tree

Start with your **primary task**:

- **Coding agent(s)** (Claude Code, aider, Hermes driving tools)
    - Need 2–3 *concurrent* agents → **`hermes-agent`** or **`code`** —
      both run Gemma 4 26B-A4B multi-slot (N=3 × 128K ctx, text-only) plus a
      code-embedding service. See
      [Multi-agent with cont-batching](../user-guide/multi-agent.md).
    - Need 5 cheap parallel sessions, smaller model → **`multi-agent-gpt-oss`**
      (GPT-OSS 20B, 14 GB).
- **Research / long-form reasoning** → **`hermes-agent`** (3 symmetric
  agents on cont-batching is the highest-aggregate-throughput configuration
  measured) or **`nemotron`** for a single reasoning model.
- **RAG / semantic search only** (no local chat LLM) → **`rag`** — two
  embedding models, 7.5 GB total, leaves the rest of the GPU free.
- **Voice** (ASR + TTS + chat) → **`speak`**.
- **Vision (image inputs)** → no bundled mode; the multi-slot Gemma preset
  is text-only. Start the single-slot `gemma4-26b-a4b-q4` service directly,
  or [create a mode](cli.md#mode-create) around it.
- **Benchmarking / measuring a model in isolation** → **`blank`** first
  (stops everything, 0 MB), then start only the service under test.
- **Finetuning on the same GPU** → **`finetuning`** — keeps only the small
  embedding service (2.5 GB) so the training job gets the rest.

## VRAM-fit table

Total VRAM per bundled mode (from `gpumod mode list`) against common GPU
sizes. ✓ = fits with headroom, ⚠ = fits but tight, ✗ = does not fit.

| Mode | Total VRAM | 12 GB | 16 GB | 24 GB |
|---|---:|:-:|:-:|:-:|
| `blank` | 0 MB | ✓ | ✓ | ✓ |
| `finetuning` | 2,500 MB | ✓ | ✓ | ✓ |
| `rag` | 7,500 MB | ✓ | ✓ | ✓ |
| `multi-agent-gpt-oss` | 14,000 MB | ✗ | ⚠ | ✓ |
| `multi-agent-qwen3-coder` | 20,000 MB | ✗ | ✗ | ✓ |
| `speak` | 22,000 MB | ✗ | ✗ | ⚠ |
| `multi-agent-qwen3-next` | 22,000 MB | ✗ | ✗ | ⚠ |
| `code` / `hermes-agent` | 22,500 MB | ✗ | ✗ | ⚠ |
| `nemotron` | 22,500 MB | ✗ | ✗ | ⚠ |
| `hacker` | 22,500 MB | ✗ | ✗ | ⚠ |

"Tight" means under ~2 GB of headroom on a 24 GB card — fine for the
designed workload, but don't co-run anything else on the GPU. Always
confirm with a simulation, which accounts for *your* GPU and any context
overrides:

```bash
gpumod simulate mode hermes-agent
```

Expected output:

```
Fits: 22500 / 24564 MB (headroom: 2064 MB)
```

## Which model does a mode load?

Three places to look, in increasing detail:

1. `gpumod mode status` (after switching) or `gpumod mode list` — names and
   VRAM totals.
2. The mode YAML in
   [`modes/`](https://github.com/jaigouk/gpumod/tree/main/modes) — the
   service ID list, often with extensive design-rationale comments
   (e.g. `hermes-agent.yaml` documents the whole slots-vs-agents model).
3. The service preset in
   [`presets/`](https://github.com/jaigouk/gpumod/tree/main/presets) — the
   exact model file, quantization, context size, and launch flags.

## What happened

Mode VRAM totals are additive: each member preset declares an empirically
measured `vram_mb`, and gpumod refuses switches whose sum exceeds your GPU.
The bundled catalog therefore encodes the host's tested configurations —
picking a mode is picking a *validated* VRAM budget, not just a model.

## Next steps

- [Run your first service](first-service.md) — if you haven't started
  anything yet
- [Modes deep dive](../user-guide/modes-deep-dive.md) — lifecycle, sleep,
  and drift recovery
- [OpenAI-compatible clients](../user-guide/openai-clients.md) — connect a
  client once your mode is up
- [CLI Reference](cli.md#gpumod-mode) — `mode create` for your own bundles
