---
title: Run Your First Service
description: Boot a small embedding service end-to-end in five minutes — install the unit, start it, hit the health endpoint, and send a first request.
---

# Run Your First Service

*Learn the full gpumod service lifecycle — preset → systemd unit → running
endpoint — in about five minutes, using a small model that fits on any GPU.*

We use `vllm-embedding-code` (Qwen3-Embedding-0.6B, ~2.5 GB VRAM, ~60 s boot)
because it's the smallest bundled preset. Every step works the same for the
large chat presets — only the VRAM and boot time change.

## Prerequisites

- gpumod installed and initialized — see [Getting Started](index.md)
  (`uv sync`, `gpumod init`)
- User-level systemd lingering enabled:
  `sudo loginctl enable-linger $USER`
- A free GPU with at least 3 GB of VRAM available (`gpumod status`)

## Steps

### 1. Check the GPU and find the preset

```bash
gpumod status
```

Expected output (abbreviated):

```
GPU: NVIDIA GeForce RTX 4090  VRAM: 24564 MB
Mode: blank
                                    Services
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Name                                       ┃ State    ┃ VRAM (MB) ┃ Driver   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ Qwen3-Embedding-0.6B                       │ stopped  │      2500 │ vllm     │
│ ...                                        │          │           │          │
└────────────────────────────────────────────┴──────────┴───────────┴──────────┘
```

The service we want is registered as `vllm-embedding-code` (from
[`presets/embedding/vllm-embedding-code.yaml`](https://github.com/jaigouk/gpumod/blob/main/presets/embedding/vllm-embedding-code.yaml)).

### 2. Install the systemd unit

gpumod renders a systemd user unit from the preset — you never write unit
files by hand:

```bash
gpumod template install vllm-embedding-code --yes
systemctl --user daemon-reload
```

Expected output:

```
Installed unit file:
~/.config/systemd/user/vllm-embedding-code.service
```

### 3. Start the service

```bash
gpumod service start vllm-embedding-code
```

Expected output (after the model loads, ~60 s):

```
Started service vllm-embedding-code successfully.
```

!!! note "Quiesce period"
    If you just stopped another GPU service, the start may be refused for a
    few seconds:

    ```
    Error: Lifecycle error for 'vllm-embedding-code' during start: Quiesce period
    active: 6s remaining (configured: 10s). A heavy GPU service was stopped 4s ago.
    Wait for the GPU driver to fully reclaim memory, or use --no-quiesce to bypass.
    ```

    This is intentional — the NVIDIA driver needs a moment to reclaim VRAM
    after a service stops. Wait the indicated seconds and retry.

### 4. Verify health

```bash
gpumod service status vllm-embedding-code
```

Expected output:

```
╭──────────────────────── Service: vllm-embedding-code ────────────────────────╮
│ Name:    Qwen3-Embedding-0.6B                                                │
│ Driver:  vllm                                                                │
│ Port:    8210                                                                │
│ VRAM:    2500 MB                                                             │
│ State:   running                                                             │
│ Health:  OK                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Or hit the health endpoint directly:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8210/health
```

Expected output:

```
200
```

### 5. Send your first request

This is an embedding model, so the first request is an embedding (chat
models use `/v1/chat/completions` instead — see
[OpenAI-compatible clients](../user-guide/openai-clients.md)):

```bash
curl -s http://localhost:8210/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-Embedding-0.6B", "input": "gpumod manages GPU services"}'
```

Expected output (embedding vector truncated):

```json
{
  "object": "list",
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.0026, -0.0789, -0.009, -0.0673, "..."]}],
  "usage": {"prompt_tokens": 7, "total_tokens": 7}
}
```

A 1024-dimension vector comes back — the service is fully operational.

### 6. Stop the service (optional)

```bash
gpumod service stop vllm-embedding-code
```

Expected output:

```
Stopped service vllm-embedding-code successfully.
```

## What happened

gpumod read the preset YAML, rendered a sandboxed Jinja2 template into a
systemd **user** unit (`~/.config/systemd/user/`, no sudo needed), and
delegated the lifecycle to `systemctl --user`. Before launch, the
[preflight checks](../user-guide/host-stability.md) verified there was
enough RAM and VRAM and that the model files exist — a failing preflight
aborts the start instead of wedging the host. Health was confirmed by
polling the preset's `health_endpoint` until it returned 200.

Managing services one at a time works, but the real workflow is **modes** —
named bundles of services that gpumod starts and stops together, with VRAM
accounting.

## Next steps

- [Pick a mode for your workload](pick-a-mode.md) — switch service bundles
  instead of individual services
- [OpenAI-compatible clients](../user-guide/openai-clients.md) — wire chat
  clients to a running LLM service
- [CLI Reference](cli.md) — all `gpumod service`, `mode`, and `template`
  commands
- [Configuration](configuration.md) — environment variables and preflight
  thresholds
