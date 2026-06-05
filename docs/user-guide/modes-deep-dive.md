---
title: Modes Deep Dive
description: How mode switching works under the hood — lifecycle ordering, VRAM release, sleep states, and drift recovery.
---

# Modes Deep Dive

*Learn what `gpumod mode switch` actually does — and how it recovers when
the database and the real world disagree.*

A **mode** is a named bundle of services defined in a YAML file under
`modes/` (e.g. `hermes-agent.yaml` lists `gemma4-26b-a4b-q4-multi` +
`vllm-embedding-code`). Switching modes is the primary day-to-day operation:
gpumod computes the difference between what's running and what the target
mode needs, then stops, starts, or wakes services accordingly.

## Prerequisites

- gpumod initialized with at least the bundled modes
  ([Getting Started](../getting-started/index.md))
- Familiarity with starting a single service —
  [Run your first service](../getting-started/first-service.md)

## Steps

### 1. See what modes exist

```bash
gpumod mode list
```

Expected output (abbreviated):

```
                                 Service Modes
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ ID           ┃ Name              ┃ Description              ┃ Total VRAM (MB) ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ blank        │ Blank Mode        │ Empty mode (benchmarks)  │               0 │
│ hermes-agent │ Hermes Agent Mode │ Gemma 4 26B-A4B multi…   │           22500 │
│ rag          │ RAG Mode          │ RAG mode with dual emb…  │            7500 │
│ ...          │                   │                          │                 │
└──────────────┴───────────────────┴──────────────────────────┴─────────────────┘
```

The VRAM column is the **sum of member services' `vram_mb`** — gpumod uses
it to refuse switches that can't fit. See
[Pick a mode](../getting-started/pick-a-mode.md) for choosing one.

### 2. Simulate before switching

```bash
gpumod simulate mode rag
```

Expected output:

```
Fits: 7500 / 24564 MB (headroom: 17064 MB)
                              Services
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
┃ ID                  ┃ Name                  ┃ Driver ┃ VRAM (MB) ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
│ vllm-embedding-code │ Qwen3-Embedding-0.6B  │ vllm   │      2500 │
│ vllm-embedding      │ Qwen3-VL-Embedding-2B │ vllm   │      5000 │
└─────────────────────┴───────────────────────┴────────┴───────────┘
```

Simulation never starts or stops anything — it's safe at any time.

### 3. Switch

```bash
gpumod mode switch rag
```

What happens, in order:

1. **Validate** — target services exist, total VRAM fits the GPU.
2. **Partition** the current state: services to stop (running but not in
   target), services to keep, services to start (in target but not
   actively running), and sleeping target services to wake.
3. **Stop outgoing services** (dependents before dependencies).
4. **Wait for VRAM release** — gpumod polls `nvidia-smi` until the driver
   has actually reclaimed the memory, then enforces a short quiesce period.
   Skipping this is how you OOM a model load on a half-released GPU.
5. **Start incoming services** in dependency order, each gated by its own
   [preflight checks](host-stability.md) and health endpoint.

Switching back (`gpumod mode switch hermes-agent`) is the same operation in
reverse — the big model reloads from page cache in tens of seconds if RAM
allowed it to stay cached.

### 4. Verify

```bash
gpumod mode status
```

Expected output (abbreviated):

```
╭───────────────────── Active Mode: rag ──────────────────────╮
│ Name:        RAG Mode                                       │
│ Description: RAG mode with dual embedding models            │
│ Services:    vllm-embedding-code, vllm-embedding            │
│ Total VRAM:  7500 MB                                        │
╰──────────────────────────────────────────────────────────────╯
```

## Sleep-capable services vs. full stops

Not every transition needs a cold stop/start. Services whose preset sets
`supports_sleep: true` can be put to **sleep** instead — the process stays
alive (no reload cost) while VRAM is reduced:

| Sleep level | Driver | Mechanism | Wake time |
|---|---|---|---|
| L1 | vLLM | offload via sleep API | <1 s |
| L2 | vLLM | deeper offload via sleep API | <2 s |
| Router | llama.cpp | model unload/load | seconds |
| none | any | service must be fully stopped to free VRAM | full restart |

During a switch, a *sleeping* target service is **woken**, not re-launched —
that's faster and preserves the process. Check a preset's `supports_sleep`
field to know which behavior you'll get: many bundled presets support it
(e.g. the Router-sleep llama.cpp coder presets), while the large Gemma 4
presets are `supports_sleep: false` — for them a mode switch means a real
stop and a real reload. The state machine is documented in the
[architecture overview](../architecture/index.md).

## Drift recovery

The database records the current mode, but reality can drift: a host
reboot, a manual `systemctl --user stop`, or a service that crashed after
a previously successful switch. Naively, `mode switch hermes-agent` while
the DB already says "hermes-agent" would be a no-op — reporting success
while nothing runs.

Since gpumod-hrgg, `mode switch` reconciles against the **actual running
set**, not the DB record:

- Target services that are *not actively running* are partitioned into the
  start list and re-launched, even when the DB says the mode is already
  current.
- Target services found *sleeping* are routed to wake instead of being
  re-launched.
- The `Started:` block in the output lists exactly which services were
  (re)launched, so a drift recovery is visible, not silent.

## Troubleshooting

**"Switch reported success but my service isn't answering."**
Run the switch again — drift recovery will re-launch whatever isn't
actually running. Then check `gpumod service status <id>` and the journal:
`journalctl --user -u <id>.service -n 50`.

**"Switch targets a stale service set."**
If you edited a mode YAML (or re-rendered units with
`template install-all`), the DB may not know yet. Sync it:

```bash
gpumod mode sync
```

Expected output:

```
Mode sync: 0 inserted, 1 updated, 10 unchanged, 0 deleted.
```

**"Start refused: quiesce period active."**
A heavy service was stopped seconds ago; the driver is still reclaiming
VRAM. Wait the indicated seconds (or pass `--no-quiesce` if you know the
GPU is clean).

**"Preflight failed: not enough RAM."**
The incoming model needs more `MemAvailable` than the host currently has —
see [Host stability](host-stability.md) for why this check refuses rather
than risks a freeze.

**When in doubt:** `gpumod mode switch blank` stops everything and releases
all VRAM. It always works and gives you a clean slate.

## What happened

Modes turn "stop these three units, watch nvidia-smi, then start those two
in the right order" into one idempotent command. The key design choices:
VRAM release is *verified* (polled), not assumed; incoming services are
gated by preflight + health checks; and the switch reconciles against
observed systemd state rather than trusting its own database — so running
the same switch twice converges instead of lying.

## Next steps

- [Pick a mode](../getting-started/pick-a-mode.md) — decision guide for the
  bundled modes
- [Multi-agent with cont-batching](multi-agent.md) — what runs inside the
  `hermes-agent` mode
- [MCP Workflows](mcp-workflows.md) — daily mode switching from an AI
  assistant
- [Architecture](../architecture/index.md) — ServiceManager, partition
  contract, and the service state machine
