---
title: Configuration
description: Configure gpumod via environment variables — database path, LLM backends, VRAM settings, and MCP server options.
---

# Configuration

All settings are configurable via environment variables with the `GPUMOD_`
prefix. Settings are managed by [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).

## Environment Variables

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `GPUMOD_DB_PATH` | `str` | `~/.config/gpumod/gpumod.db` | Path to the SQLite database file |
| `GPUMOD_PRESETS_DIR` | `str` | Auto-resolved | Path to the built-in presets directory |
| `GPUMOD_MODES_DIR` | `str` | Auto-resolved | Path to the built-in modes directory |
| `GPUMOD_LOG_LEVEL` | `str` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `GPUMOD_LLM_BACKEND` | `str` | `openai` | LLM provider backend (`openai`, `anthropic`, `ollama`) |
| `GPUMOD_LLM_API_KEY` | `str` | None | API key for the LLM backend (stored as `SecretStr`) |
| `GPUMOD_LLM_MODEL` | `str` | `gpt-4o-mini` | LLM model identifier |
| `GPUMOD_LLM_BASE_URL` | `str` | None | Custom base URL for the LLM API (e.g., for Ollama or proxy) |
| `GPUMOD_MCP_RATE_LIMIT` | `int` | `10` | Maximum MCP requests per minute (must be >= 1) |
| `GPUMOD_RAM_MIN_FREE_MB` | `int` | `1024` | Minimum free RAM (MB) — blocks service start below this |
| `GPUMOD_RAM_WARN_FREE_MB` | `int` | `4096` | Warn threshold (MB) — logs warning below this |
| `GPUMOD_VRAM_SAFETY_MARGIN_MB` | `int` | `512` | Extra VRAM buffer (MB) required beyond service allocation |
| `GPUMOD_QUIESCE_SECONDS` | `float` | `10.0` | Seconds to wait after heavy stop before allowing new heavy start (range 0-300). Helps the kernel consolidate freed pages so the next CUDA-pinned allocation doesn't hit a fragmentation hang. |

## Example: Using Ollama locally

```bash
export GPUMOD_LLM_BACKEND=ollama
export GPUMOD_LLM_BASE_URL=http://localhost:11434
export GPUMOD_LLM_MODEL=llama3.1
gpumod plan suggest
```

## Example: Custom database location

```bash
export GPUMOD_DB_PATH=/data/gpumod/services.db
gpumod init
```

## Example: Tuning preflight thresholds

On machines with high memory pressure (e.g., ZFS caches), the default RAM
thresholds may be too aggressive. Lower them via environment variables:

```bash
export GPUMOD_RAM_MIN_FREE_MB=512
export GPUMOD_RAM_WARN_FREE_MB=2048
gpumod service start my-model
```

To give large models extra VRAM headroom:

```bash
export GPUMOD_VRAM_SAFETY_MARGIN_MB=1024
```

A `.env.example` file is included in the repository root — copy it to `.env`
and uncomment the variables you want to override.

## AI Planning

gpumod integrates with LLM APIs to provide AI-assisted VRAM allocation
planning via `gpumod plan suggest`.

### How It Works

1. gpumod gathers your registered services, their VRAM requirements, and
   GPU capacity.
2. A carefully constructed prompt is sent to the configured LLM backend
   with only minimal data (service IDs, VRAM amounts, GPU capacity).
3. The LLM returns a structured JSON plan with service allocations and
   reasoning.
4. gpumod validates all IDs and values in the LLM response against strict
   regex patterns and VRAM limits.
5. The plan is simulated through the SimulationEngine to verify
   feasibility.
6. Results are displayed with advisory CLI commands you can choose to
   execute.

### Supported Backends

| Backend | Environment Variable | Notes |
|---------|---------------------|-------|
| OpenAI | `GPUMOD_LLM_API_KEY` | Default backend, uses `gpt-4o-mini` |
| Anthropic | `GPUMOD_LLM_API_KEY` | Set `GPUMOD_LLM_BACKEND=anthropic` |
| Ollama | (no key required) | Set `GPUMOD_LLM_BACKEND=ollama`, runs locally |

### Example

```bash
# Configure the LLM backend
export GPUMOD_LLM_BACKEND=openai
export GPUMOD_LLM_API_KEY=sk-...
export GPUMOD_LLM_MODEL=gpt-4o-mini

# Get a plan
gpumod plan suggest
```

Output:

```
Fits: 9216 / 24576 MB (headroom: 15360 MB)
            AI-Suggested VRAM Plan
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Service ID     ┃ VRAM (MB) ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ llama-3-1-8b   │      8192 │
│ bge-large      │      1024 │
│ Total          │      9216 │
└────────────────┴───────────┘

Reasoning: The Llama 3.1 8B model is the primary chat service
requiring 8GB VRAM. BGE-Large provides embedding retrieval at
only 1GB. Together they fit well within the 24GB RTX 4090 with
15GB headroom for KV cache growth.

Suggested commands (advisory only):
  gpumod simulate services llama-3-1-8b,bge-large
  gpumod service start llama-3-1-8b
  gpumod service start bge-large
```

### Dry Run

Preview the prompt that would be sent to the LLM without actually calling
the API:

```bash
gpumod plan suggest --dry-run
```

This is useful for reviewing what data is sent to the LLM, verifying
your configuration, and debugging prompt templates.
