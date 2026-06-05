---
title: OpenAI-Compatible Clients
description: Wire Hermes, Claude Code, aider, and Continue.dev to a gpumod-managed local LLM endpoint, including the reasoning-model max_tokens gotcha.
---

# OpenAI-Compatible Clients

*Learn how to point any OpenAI-compatible client at a gpumod-managed LLM
service — and how to avoid the empty-response trap with reasoning models.*

All examples target the `hermes-agent` mode endpoint at
`http://localhost:7109/v1` (Gemma 4 26B-A4B multi-slot — see
[Multi-agent with cont-batching](multi-agent.md)). Any llama.cpp or vLLM
service managed by gpumod exposes the same API shape on its own port —
check `gpumod service status <id>` for the port.

## Prerequisites

- A running LLM service —
  [Run your first service](../getting-started/first-service.md), then
  `gpumod mode switch hermes-agent` (or any chat-capable mode)
- Health returns 200: `curl -s http://localhost:7109/health`

## The endpoint shape

| Property | Value |
|---|---|
| Base URL | `http://localhost:7109/v1` |
| API key | Not required — any placeholder works (`not-needed`) |
| Chat endpoint | `POST /v1/chat/completions` |
| Anthropic-compatible endpoint | `POST /v1/messages` (llama.cpp also serves this — used by Claude Code) |
| Binding | `localhost` only by default (set `host: "0.0.0.0"` in the preset to expose) |

### 1. Find the model name

```bash
curl -s http://localhost:7109/v1/models | python3 -c \
  "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])"
```

Expected output:

```
gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
```

llama-server hosts a single model, so it serves whatever `model` string you
send — but clients usually require *some* value, and using the real name
keeps logs unambiguous.

### 2. Smoke-test with curl

```bash
curl -s http://localhost:7109/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-26B-A4B-it-UD-IQ4_XS.gguf",
    "messages": [{"role": "user", "content": "Say OK"}],
    "max_tokens": 1500
  }'
```

Expected output (abbreviated):

```json
{
  "choices": [{"message": {"content": "OK", "reasoning_content": "The user said \"Say OK\"..."},
               "finish_reason": "stop"}],
  "usage": {"completion_tokens": 29, "prompt_tokens": 11}
}
```

## The reasoning-model gotcha: `max_tokens >= 1500`

The bundled Gemma 4 presets enable thinking
(`--chat-template-kwargs '{"enable_thinking":true}'`). The model **thinks
before it answers**, and the thinking tokens count against `max_tokens`:

- A one-sentence answer can consume **200+ completion tokens** (measured:
  212 tokens for "I have memorized the magic word…").
- If `max_tokens` is too small (e.g. the OpenAI-client default of 256), the
  budget is exhausted mid-thought — you get `finish_reason: "length"` and
  an **empty or truncated `content`** while `reasoning_content` holds the
  cut-off thinking.

**Rule: always send `max_tokens >= 1500`** against thinking-enabled
presets. For vision or long-form answers, go higher (see the per-preset
notes in the [preset YAMLs](https://github.com/jaigouk/gpumod/tree/main/presets/llm)).

## Wire up your client

### Hermes

```bash
hermes model
# Select: OpenAI-compatible
# Base URL: http://localhost:7109/v1
# API key: not-needed
# Model: gemma-4-26B-A4B
```

Switch back to a cloud provider any time with `hermes model` — no config
files to edit.

### Claude Code

Claude Code speaks the Anthropic Messages API. llama.cpp's server also
exposes an Anthropic-compatible `/v1/messages` endpoint, so you can point
Claude Code at the local service with environment variables:

```bash
export ANTHROPIC_BASE_URL=http://localhost:7109
export ANTHROPIC_AUTH_TOKEN=not-needed
export ANTHROPIC_MODEL=gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
claude
```

Verify the endpoint responds to the Messages shape first:

```bash
curl -s -X POST http://localhost:7109/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model": "gemma-4", "max_tokens": 1500, "messages": [{"role": "user", "content": "Say OK"}]}'
```

Expected output (abbreviated):

```json
{"type": "message", "role": "assistant",
 "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "OK"}],
 "stop_reason": "end_turn"}
```

### aider

```bash
export OPENAI_API_BASE=http://localhost:7109/v1
export OPENAI_API_KEY=not-needed
aider --model openai/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
```

The `openai/` prefix tells aider to use the generic OpenAI-compatible
provider against your `OPENAI_API_BASE`.

### Continue.dev

Add the model to your Continue config (`~/.continue/config.yaml`):

```yaml
models:
  - name: Gemma 4 local
    provider: openai
    model: gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
    apiBase: http://localhost:7109/v1
    apiKey: not-needed
    defaultCompletionOptions:
      maxTokens: 4000
```

Set `maxTokens` explicitly — Continue's default is too small for
thinking-enabled models (see the gotcha above).

## Running several clients at once

On the multi-slot preset, all clients share the same base URL and are
served **concurrently** — up to 3 in parallel, additional requests queue
briefly. No per-client configuration differs. For persistent per-agent
state across disconnects, see
[slot pinning and save/restore](multi-agent.md#3-send-a-slot-pinned-chat-request).

## What happened

gpumod services expose standard inference APIs (llama.cpp's server
implements OpenAI chat-completions *and* Anthropic messages; vLLM
implements OpenAI). Because the API is the contract, any client that can
change its base URL works unmodified — the only local-specific concerns are
the placeholder API key, the single-model `model` field, and budgeting
`max_tokens` for thinking tokens.

## Next steps

- [Multi-agent with cont-batching](multi-agent.md) — serve 3 concurrent
  clients from one model
- [Pick a mode](../getting-started/pick-a-mode.md) — which model bundle
  fits your workload
- [MCP Integration](mcp.md) — manage gpumod itself from your AI assistant
