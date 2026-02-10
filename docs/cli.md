---
title: gpumod CLI Reference
description: Complete command reference for gpumod — status, service, mode, simulate, model, template, plan, and tui commands for GPU service management.
---

# CLI Reference

gpumod uses a subcommand structure. Every command supports `--help` for
detailed usage.

```bash
gpumod --help
```

## gpumod status

Show system status including GPU info, VRAM usage, active mode, and
running services.

```bash
# Default table output
gpumod status

# Visual VRAM bar chart
gpumod status --visual

# JSON output for scripting
gpumod status --json
```

## gpumod init

Initialize the database and load preset configurations.

```bash
# Use default database path (~/.config/gpumod/gpumod.db)
gpumod init

# Specify a custom database path
gpumod init --db-path /path/to/custom.db

# Specify an additional preset directory
gpumod init --preset-dir /path/to/my/presets
```

## gpumod service

Manage individual GPU services.

### service list

List all registered services with their current state.

```bash
gpumod service list
gpumod service list --json
```

### service status

Show detailed status of a specific service.

```bash
gpumod service status llama-3-1-8b
gpumod service status llama-3-1-8b --json
```

Output includes service name, driver type, port, VRAM allocation, state,
uptime, and health check status.

### service start

Start a registered service.

```bash
gpumod service start llama-3-1-8b
```

### service stop

Stop a running service.

```bash
gpumod service stop llama-3-1-8b
```

## gpumod mode

Manage service modes -- named groups of services for specific use cases.

### mode list

List all defined modes.

```bash
gpumod mode list
gpumod mode list --json
```

### mode status

Show the currently active mode.

```bash
gpumod mode status
gpumod mode status --json
```

### mode switch

Switch to a different mode. This starts services in the target mode and
stops services not in the target mode.

```bash
gpumod mode switch chat-mode
gpumod mode switch chat-mode --json
```

### mode create

Create a new mode from existing services.

```bash
gpumod mode create "Coding Mode" \
  --services devstral-small-2,bge-large \
  --description "Code completion with embedding retrieval"
```

The mode ID is auto-generated from the name (e.g., "Coding Mode" becomes
`coding-mode`).

## gpumod template

Manage Jinja2 systemd unit file templates.

### template list

List available template files.

```bash
gpumod template list
gpumod template list --json
```

### template show

Show a rendered template with sample context.

```bash
gpumod template show vllm.service.j2
```

### template generate

Generate a systemd unit file for a registered service. The template is
selected automatically based on the service driver type.

```bash
# Preview the generated unit file
gpumod template generate llama-3-1-8b

# Write to a file
gpumod template generate llama-3-1-8b --output /tmp/llama.service

# JSON output
gpumod template generate llama-3-1-8b --json
```

### template install

Install a generated unit file to the systemd directory. Requires
confirmation via `--yes`.

```bash
# Preview first (no --yes)
gpumod template install llama-3-1-8b

# Install with confirmation
gpumod template install llama-3-1-8b --yes
```

The unit file is written to `~/.config/systemd/user/gpumod-{service_id}.service`.

## gpumod model

Manage the ML model registry for VRAM estimation.

### model list

List all registered models.

```bash
gpumod model list
gpumod model list --json
```

### model info

Show detailed model information and VRAM estimates.

```bash
# Default context size (4096 tokens)
gpumod model info meta-llama/Llama-3.1-8B-Instruct

# Custom context size
gpumod model info meta-llama/Llama-3.1-8B-Instruct --context-size 32768
```

### model register

Register a model in the registry. Metadata is automatically fetched from
HuggingFace Hub for `huggingface` source models.

```bash
# Register a HuggingFace model (metadata auto-fetched)
gpumod model register meta-llama/Llama-3.1-8B-Instruct

# Register a GGUF model with file path
gpumod model register my-gguf-model \
  --source gguf \
  --file-path ~/models/model.gguf

# Register a local model with manual metadata
gpumod model register my-local-model \
  --source local \
  --vram 8192 \
  --params 7.0 \
  --architecture llama
```

### model remove

Remove a model from the registry.

```bash
gpumod model remove meta-llama/Llama-3.1-8B-Instruct
```

## gpumod simulate

Simulate VRAM usage without starting or stopping services.

### simulate mode

Simulate VRAM usage for a defined mode, optionally adding or removing
services from the simulation.

```bash
# Simulate a mode as-is
gpumod simulate mode chat-mode

# Add a service to the simulation
gpumod simulate mode chat-mode --add mistral-7b

# Remove a service from the simulation
gpumod simulate mode chat-mode --remove bge-large

# Override context sizes for specific services
gpumod simulate mode chat-mode --context llama-3-1-8b=32768

# Visual VRAM bar comparison
gpumod simulate mode chat-mode --visual

# JSON output
gpumod simulate mode chat-mode --json
```

When services exceed GPU VRAM, gpumod suggests alternatives such as
dropping optional services or reducing context window sizes.

### simulate services

Simulate VRAM usage for an explicit list of services.

```bash
# Simulate specific services
gpumod simulate services llama-3-1-8b,bge-large,mistral-7b

# With context overrides
gpumod simulate services llama-3-1-8b,bge-large \
  --context llama-3-1-8b=16384

# Visual output
gpumod simulate services llama-3-1-8b,bge-large --visual
```

## gpumod plan

AI-assisted VRAM allocation planning.

### plan suggest

Get an AI-generated VRAM allocation plan. The LLM analyzes your
registered services and GPU capacity to suggest an optimal configuration.

```bash
# Get a plan for all registered services
gpumod plan suggest

# Plan for a specific mode
gpumod plan suggest --mode chat-mode

# Set a VRAM budget (e.g., leave headroom for other processes)
gpumod plan suggest --budget 20000

# Preview the prompt without calling the LLM
gpumod plan suggest --dry-run

# JSON output
gpumod plan suggest --json
```

The plan output includes:

- AI-suggested service allocations with VRAM amounts
- Simulation results showing whether the plan fits the GPU
- Advisory CLI commands you can copy-paste to implement the plan
- Reasoning from the LLM about its allocation decisions

Plans are **advisory only** -- gpumod never auto-executes LLM suggestions.

## gpumod tui

Launch an interactive terminal dashboard powered by
[Textual](https://textual.textualize.io/).

```bash
gpumod tui
```

The TUI displays a live GPU status bar, service list with state
indicators, a command input for `/status`, `/switch <mode>`,
`/simulate`, and `/quit`, and a footer with keyboard shortcuts.

Press `q` to quit or type `/help` for available commands.

## gpumod discover

Discover GGUF models from HuggingFace and generate ready-to-use presets.

```bash
# List unsloth models (default)
gpumod discover

# Search by name
gpumod discover --search deepseek

# Search in a specific organization
gpumod discover -s kimi -a moonshotai

# Combine author and search
gpumod discover --author bartowski --search llama

# Filter by task type
gpumod discover --task code
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--search` | `-s` | Search models by name (e.g., "deepseek", "kimi") |
| `--author` | `-a` | HuggingFace organization (default: unsloth) |
| `--task` | | Filter by task: code, chat, embed, reasoning |
| `--vram` | | VRAM budget in MB (default: detected available) |
| `--context` | | Context size (default: 8192) |
| `--dry-run` | | Preview without writing files |
| `--json` | | Output JSON, no interaction |
| `--no-cache` | | Bypass HuggingFace cache |
| `--verbose` | `-v` | Debug output |

### Workflow

1. **System detection** -- Detects GPU, VRAM, and running services
2. **Model search** -- Fetches GGUF models matching your criteria
3. **VRAM filtering** -- Shows only models that fit your VRAM budget
4. **Interactive selection** -- Pick a model and quantization
5. **Preset generation** -- Creates a ready-to-use YAML preset

### Examples

```bash
# Find coding models from unsloth that fit 24GB VRAM
gpumod discover --task code --vram 24576

# Search for DeepSeek models from any author, output as JSON
gpumod discover --search deepseek --author "" --json

# Preview a preset without writing it
gpumod discover --search qwen --dry-run
```

After generating a preset, follow the printed instructions to download
the model and register the service.

## gpumod watch

Watch preset and mode directories for YAML file changes and automatically
sync to the database. Useful for rapid iteration during development.

```bash
# Start watching with default settings (500ms debounce)
gpumod watch

# Custom debounce window
gpumod watch --debounce 200

# Skip initial sync before starting
gpumod watch --no-sync

# Timeout after N seconds (for testing)
gpumod watch --timeout 60
```

The watcher monitors `.yaml` and `.yml` files, ignoring editor temp files
(`.swp`, `.tmp`, `~` backups). Changes are debounced to coalesce rapid saves
from editors. Press `Ctrl+C` to stop.
