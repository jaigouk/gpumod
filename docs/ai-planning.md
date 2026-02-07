---
title: AI-Assisted VRAM Planning
description: Use gpumod plan suggest for LLM-powered VRAM allocation recommendations, or let AI assistants plan via MCP tools.
---

# AI-Assisted Planning

`gpumod plan suggest` asks an LLM to recommend which services to run on
your GPU. It sends your registered services and GPU capacity to the LLM,
and displays the response. **It never starts or stops anything** -- the
output is advisory only.

This is different from:

- `gpumod simulate` -- deterministic VRAM math, no LLM involved
- `gpumod service start` -- actually starts a service
- `gpumod mode switch` -- actually switches running services

## How It Works

1. Collects registered services (IDs + VRAM) and GPU capacity
2. Sends a prompt to the configured LLM backend
3. Validates the LLM response (strict ID/value checks)
4. Displays the suggestion with reasoning

## Setup

Configure an LLM backend via environment variables:

```bash
# OpenAI (default)
export GPUMOD_LLM_BACKEND=openai
export GPUMOD_LLM_API_KEY=sk-...
export GPUMOD_LLM_MODEL=gpt-4o-mini

# Anthropic
export GPUMOD_LLM_BACKEND=anthropic
export GPUMOD_LLM_API_KEY=sk-ant-...

# Ollama (local, no key needed)
export GPUMOD_LLM_BACKEND=ollama
export GPUMOD_LLM_BASE_URL=http://localhost:11434
export GPUMOD_LLM_MODEL=llama3.1
```

## Usage

```bash
# Ask the LLM for a plan
uv run gpumod plan suggest

# Plan for a specific mode
uv run gpumod plan suggest --mode chat-mode

# Set a VRAM budget (leave headroom for other processes)
uv run gpumod plan suggest --budget 20000

# Preview the prompt without calling the LLM
uv run gpumod plan suggest --dry-run
```

## Example Output

```
            AI-Suggested VRAM Plan
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Service ID     ┃ VRAM (MB) ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ llama-3-1-8b   │      8192 │
│ bge-large      │      1024 │
│ Total          │      9216 │
└────────────────┴───────────┘
Fits: 9216 / 24576 MB (headroom: 15360 MB)

Reasoning: Llama 3.1 8B is the primary chat service at 8 GB.
BGE-Large adds embedding retrieval at 1 GB. Together they fit
within the 24 GB RTX 4090 with 15 GB headroom for KV cache.
```

The output is text you read and decide whether to act on.
gpumod does not auto-execute any LLM suggestion.

## Planning via MCP

If you use gpumod through its MCP server (in Cursor, Claude Code, etc.),
you do not need `plan suggest`. The AI assistant **is** the planner -- it
can call MCP tools directly to gather the same information and reason
about it:

1. `gpu_status` -- see GPU capacity and current VRAM usage
2. `list_services` -- see registered services and their VRAM requirements
3. `simulate_mode` -- test whether a configuration fits before switching
4. `switch_mode` -- apply the plan (with your confirmation)

The `plan suggest` CLI command exists for when you are working in the
terminal without an AI assistant.
