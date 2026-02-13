---
title: MCP Workflows Guide
description: Common workflows for using gpumod MCP tools with AI assistants.
---

# MCP Workflows Guide

Practical workflows for managing GPU services via MCP tools in Claude Code,
Cursor, or other AI assistants.

## Quick Navigation

| Workflow | Use Case | Key Tools |
|----------|----------|-----------|
| [1. Find & Run a Model](#workflow-1-find-and-run-a-model) | Try a new model | `search_hf_models`, `list_gguf_files`, `switch_mode` |
| [2. Set Up RAG](#workflow-2-set-up-rag) | Embedding + chat | `simulate_mode`, `switch_mode` |
| [3. Run Large Models](#workflow-3-run-large-models) | 20GB+ models | `gpu_status`, `switch_mode("blank")` |
| [4. Create a Preset](#workflow-4-create-a-preset) | Save model config | `generate_preset`, `fetch_model_config` |
| [5. Create a Mode](#workflow-5-create-a-mode) | Group services | `simulate_mode`, `list_modes` |
| [6. Daily Mode Switching](#workflow-6-daily-mode-switching) | Work transitions | `switch_mode` |
| [7. Troubleshoot](#workflow-7-troubleshoot) | Debug failures | `gpu_status`, `list_services` |

---

## Workflow 1: Find and Run a Model

**Goal**: Discover a model on HuggingFace, check if it fits, and run it.

### Steps

**1. Check current state**

```
You: "What's my GPU status?"
```

```json
{"vram_free_mb": 24408, "current_mode": "blank", "running_services": []}
```

**2. Search for models**

```
You: "Find GLM-4 coding models"
```

Tool: `search_hf_models(query="GLM-4 code GGUF")`

**3. Check quantizations**

```
You: "What GGUF files for GLM-4.7-Flash-UD?"
```

Tool: `list_gguf_files(repo_id="THUDM/GLM-4.7-Flash-UD-GGUF")`

```json
{"files": [
  {"filename": "Q4_K_M.gguf", "vram_estimate_mb": 17500},
  {"filename": "Q4_K_XL.gguf", "vram_estimate_mb": 21000}
]}
```

**4. Simulate fit**

```
You: "Will Q4_K_XL fit on my 24GB GPU?"
```

Tool: `simulate_mode` → fits: true, margin: 3.5GB

**5. Switch and load**

```
You: "Switch to code mode"
```

Tool: `switch_mode("code")`

If the GGUF file doesn't exist, gpumod detects this and asks for confirmation:

```
Model file not found: /data/models/gguf/GLM-4.7-Flash-UD-Q4_K_XL.gguf
Download from HuggingFace? (21.5 GB) [y/N]:
```

Downloads require explicit user confirmation (default NO).

### Outcome

Model loads in 30-60 seconds. Service available at configured port.

---

## Workflow 2: Set Up RAG

**Goal**: Run embedding + chat models together within 24GB.

### Steps

**1. Plan VRAM budget**

```
You: "I need embeddings and chat for RAG. What fits in 24GB?"
```

Assistant calculates:
- Embedding (Qwen-2B): ~3GB
- Chat (Mistral-Small-24B Q4): ~15GB
- **Total: 18GB** ✓

**2. Verify with simulation**

```
You: "Simulate rag mode"
```

Tool: `simulate_mode("rag")`

```json
{"fits": true, "total_vram_mb": 18000, "margin_mb": 6564}
```

**3. Switch mode**

```
You: "Switch to rag mode"
```

Tool: `switch_mode("rag")`

```json
{"started": ["qwen-embedding", "mistral-chat"], "stopped": ["glm-code"]}
```

### Outcome

Both services running. Embedding on port 8200, chat on port 7070.

---

## Workflow 3: Run Large Models

**Goal**: Run a 20GB+ model that needs nearly all VRAM.

### Steps

**1. Check what's using VRAM**

```
You: "GPU status"
```

```json
{"vram_used_mb": 18000, "current_mode": "rag"}
```

**2. Clear VRAM first**

```
You: "Switch to blank mode"
```

Tool: `switch_mode("blank")`

Wait for VRAM release (~5-10s).

**3. Verify empty**

```
You: "GPU status"
```

```json
{"vram_used_mb": 156, "running_services": []}
```

**4. Start large model**

```
You: "Switch to nemotron mode"
```

Tool: `switch_mode("nemotron")`

### Outcome

23GB model loads. Takes 45-60 seconds for large models.

### Key Point

Large models (>20GB) need blank mode first. Don't try to load over existing services.

---

## Workflow 4: Create a Preset

**Goal**: Save a model configuration for reuse.

### Steps

**1. Research the model**

```
You: "Get config for Devstral-Small-2505"
```

Tool: `fetch_model_config(repo_id="mistralai/Devstral-Small-2505")`

```json
{"architecture": "MistralForCausalLM", "context_length": 131072}
```

**2. Generate preset**

```
You: "Generate a preset with 32K context on port 8000"
```

Tool: `generate_preset`

```yaml
id: devstral-small
driver: vllm
port: 8000
vram_mb: 16000
model_id: mistralai/Devstral-Small-2505
unit_vars:
  max_model_len: 32768
```

**3. Save and register**

```bash
# Save the YAML, then:
gpumod preset load presets/devstral-small.yaml
```

**4. Verify**

```
You: "List services"
```

Tool: `list_services` → shows new service (stopped).

### Outcome

Service registered and available for mode inclusion.

---

## Workflow 5: Create a Mode

**Goal**: Group services into a named configuration.

### Steps

**1. Plan and verify VRAM**

```
You: "I want 'research' mode with embedding + reasoner"
```

Check if it fits:

| Service | VRAM |
|---------|------|
| qwen-embedding | 3GB |
| mistral-reasoner | 15GB |
| **Total** | **18GB** ✓ |

**2. Create mode**

```bash
gpumod mode create research --services qwen-embedding,mistral-reasoner
```

**3. Verify**

```
You: "List modes"
```

Tool: `list_modes`

```json
{"modes": ["blank", "code", "rag", "research"]}
```

### Outcome

New mode ready for `switch_mode("research")`.

---

## Workflow 6: Daily Mode Switching

**Goal**: Transition between work contexts throughout the day.

### Morning: Code

```
You: "Start with code mode"
```

Tool: `switch_mode("code")` → GLM-4.7 loads (~30s)

### Midday: Research

```
You: "Switch to research mode"
```

Tool: `switch_mode("research")`

What happens:
1. Stops code mode services
2. Waits for VRAM release
3. Starts embedding + reasoner

### Evening: Shutdown

```
You: "Blank mode"
```

Tool: `switch_mode("blank")` → GPU idle

### Key Points

- Mode switches handle VRAM automatically
- No manual service management needed
- Same endpoint ports across modes

---

## Workflow 7: Troubleshoot

**Goal**: Diagnose why a model won't load.

### Check 1: VRAM conflict

```
You: "GPU status"
```

```json
{"vram_used_mb": 21000, "running_services": ["glm-code"]}
```

**Problem**: Another model using VRAM.

**Fix**: `switch_mode("blank")` first, or switch directly to target mode.

### Check 2: Service health

```
You: "List services"
```

```json
{"services": [
  {"id": "glm-code", "state": "unhealthy", "health": "timeout"}
]}
```

**Problem**: Service started but not responding.

**Fix**: Check logs with `journalctl --user -u gpumod-glm-code.service -n 50`

### Check 3: Model file missing

gpumod's preflight check detects missing GGUF files before starting:

```
Preflight failed: Model file not found
  /data/models/gguf/GLM-4.7-Flash-UD-Q4_K_XL.gguf

Download from HuggingFace? (21.5 GB) [y/N]:
```

**Options**:
- Type `y` to download (requires disk space + time for large models)
- Type `n` to abort and download manually:
  ```bash
  wget -c "https://huggingface.co/THUDM/GLM-4.7-Flash-UD-GGUF/resolve/main/Q4_K_XL.gguf" \
    -O /data/models/gguf/GLM-4.7-Flash-UD-Q4_K_XL.gguf
  ```

### Recovery

When in doubt:

```
You: "Switch to blank mode"
```

This stops everything and releases VRAM. Always works.

---

## Tool Reference

### Tier 1: Direct Operations

| Tool | Purpose | Example |
|------|---------|---------|
| `gpu_status` | Check GPU state | "What's my VRAM?" |
| `list_services` | Show all services | "What's running?" |
| `list_modes` | Show available modes | "What modes exist?" |
| `switch_mode` | Change mode | "Switch to rag" |
| `simulate_mode` | Test VRAM fit | "Will this fit?" |
| `start_service` | Start one service | "Start embedding" |
| `stop_service` | Stop one service | "Stop chat model" |

### Tier 2: Discovery

| Tool | Purpose | Example |
|------|---------|---------|
| `search_hf_models` | Find models | "Find Qwen GGUF" |
| `list_gguf_files` | Check quantizations | "What sizes available?" |
| `fetch_model_config` | Get architecture | "What's the context length?" |
| `generate_preset` | Create config | "Make a preset" |
| `fetch_driver_docs` | Get llama.cpp/vLLM docs | "What flags available?" |

### Tier 3: Complex Reasoning

| Tool | Purpose | Example |
|------|---------|---------|
| `consult` | Multi-step analysis | "Can I run Qwen-235B on 24GB?" |

---

## Best Practices

### Before Loading Large Models

1. Check VRAM: `gpu_status`
2. Switch to blank for 20GB+ models
3. Wait for load (30-60s for large models)

### For Mode Switching

- Trust automatic VRAM management
- Don't force on timeout — wait longer
- Use `switch_mode("blank")` for recovery

### For RAG Setups

- Plan total VRAM (add all services + 1GB margin)
- Start embedding first (smaller, verifies setup)
- Check health before sending requests

### Security

- Services bind to localhost (127.0.0.1) by default
- Override with `host: "0.0.0.0"` in preset for external access

---

## See Also

- [CLI Reference](../getting-started/cli.md) — Command-line interface
- [MCP Integration](mcp.md) — Setup for AI assistants

---

Sources:
- [GitBook: Documentation Structure Tips](https://gitbook.com/docs/guides/docs-best-practices/documentation-structure-tips)
- [MDN: Creating Effective Technical Documentation](https://developer.mozilla.org/en-US/blog/technical-writing/)
