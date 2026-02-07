# Configuration

All settings are configurable via environment variables with the `GPUMOD_`
prefix. Settings are managed by [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).

## Environment Variables

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `GPUMOD_DB_PATH` | `str` | `~/.config/gpumod/gpumod.db` | Path to the SQLite database file |
| `GPUMOD_PRESETS_DIR` | `str` | Auto-resolved | Path to the built-in presets directory |
| `GPUMOD_LOG_LEVEL` | `str` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `GPUMOD_LLM_BACKEND` | `str` | `openai` | LLM provider backend (`openai`, `anthropic`, `ollama`) |
| `GPUMOD_LLM_API_KEY` | `str` | None | API key for the LLM backend (stored as `SecretStr`) |
| `GPUMOD_LLM_MODEL` | `str` | `gpt-4o-mini` | LLM model identifier |
| `GPUMOD_LLM_BASE_URL` | `str` | None | Custom base URL for the LLM API (e.g., for Ollama or proxy) |
| `GPUMOD_MCP_RATE_LIMIT` | `int` | `10` | Maximum MCP requests per minute (must be >= 1) |

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
