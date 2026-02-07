# gpumod

GPU Service Manager for ML workloads on Linux/NVIDIA systems.

gpumod manages vLLM, llama.cpp, FastAPI, and Docker-based inference services on
NVIDIA GPUs. It tracks VRAM allocation, supports mode-based service
switching, provides VRAM simulation before deployment, and exposes an
MCP server for AI assistant integration.

---

## Table of Contents

- [Motivation](#motivation)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
  - [gpumod status](#gpumod-status)
  - [gpumod init](#gpumod-init)
  - [gpumod service](#gpumod-service)
  - [gpumod mode](#gpumod-mode)
  - [gpumod template](#gpumod-template)
  - [gpumod model](#gpumod-model)
  - [gpumod simulate](#gpumod-simulate)
  - [gpumod plan](#gpumod-plan)
  - [gpumod tui](#gpumod-tui)
- [MCP Server Setup](#mcp-server-setup)
- [AI Planning](#ai-planning)
- [Configuration](#configuration)
- [Presets](#presets)
- [Security Model](#security-model)
- [Contributing](#contributing)
- [License](#license)

---

## Motivation

Running multiple ML inference services on a single GPU requires careful
VRAM management. Loading a 7B parameter model alongside an embedding
service and a code completion model means tracking exactly how much memory
each service needs, which combinations fit on your hardware, and how to
switch between workload profiles without manual intervention.

gpumod solves this by providing:

- A single source of truth for service VRAM requirements
- Mode-based switching between workload configurations
- VRAM simulation to validate configurations before deployment
- AI-assisted planning to suggest optimal allocations
- An MCP server so AI assistants can query and manage GPU services directly

---

## Features

- **Service Management** -- Register, start, stop, and monitor GPU services
  with support for vLLM, llama.cpp, FastAPI, and Docker drivers.
- **Mode Switching** -- Define named modes (e.g., "chat", "coding",
  "embedding") that bundle services together. Switch modes to start/stop
  the right combination of services automatically.
- **VRAM Simulation** -- Simulate VRAM usage for any mode or service
  combination before deployment. Receive alternative suggestions when
  services exceed GPU capacity.
- **Template Engine** -- Generate and install systemd unit files from
  Jinja2 templates, customized per driver type.
- **Model Registry** -- Track ML models with metadata fetched from
  HuggingFace Hub or parsed from GGUF files. Estimate VRAM based on
  parameter count and context window size.
- **YAML Presets** -- Define services as YAML files for repeatable
  deployments. Ship with built-in presets for common models.
- **AI Planning** -- Use LLM backends (OpenAI, Anthropic, Ollama) to
  suggest optimal VRAM allocation plans. Plans are advisory only and
  never auto-executed.
- **MCP Server** -- Expose GPU management as an MCP server for Claude
  Desktop and other MCP-compatible AI assistants. Includes 9 tools and
  8 browsable resources.
- **Health Monitoring** -- Continuous health checking for running services
  with configurable failure thresholds, jitter, exponential backoff,
  and per-service monitoring tasks.
- **Docker Support** -- Manage containerized services (Qdrant, Langfuse)
  via the Docker SDK alongside systemd-managed inference services.
- **Interactive TUI** -- Textual-based terminal dashboard with live GPU
  status, service list, and command input (`gpumod tui`).
- **Rich CLI** -- Beautiful terminal output with Rich tables, panels,
  VRAM bar charts, and JSON output mode for scripting.
- **Security First** -- Input validation at every boundary, error
  sanitization, rate limiting, and a documented threat model.

---

## Installation

### From PyPI

```bash
pip install gpumod
```

### From source (development)

```bash
git clone https://github.com/jaigouk/gpumod.git
cd gpumod
pip install -e ".[dev]"
```

Or using [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e ".[dev]"
```

### Requirements

- Python >= 3.11
- Linux with NVIDIA GPU and drivers installed
- `nvidia-smi` accessible in PATH (for VRAM detection)

---

## Quick Start

### 1. Initialize the database

```bash
gpumod init
```

This creates the SQLite database at `~/.config/gpumod/gpumod.db` and
loads any preset YAML files found in the built-in presets directory.

```
Initialized. Found 6 preset(s), loaded 6, skipped 0.
```

### 2. Check system status

```bash
gpumod status
```

View GPU information, current mode, and running services:

```
GPU: NVIDIA RTX 4090  VRAM: 24576 MB
Mode: none
No services registered.
```

Use `--visual` for a VRAM bar chart:

```bash
gpumod status --visual
```

Use `--json` for machine-readable output:

```bash
gpumod status --json
```

### 3. List registered services

```bash
gpumod service list
```

```
                    GPU Services
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
┃ ID               ┃ Name                 ┃ Driver  ┃ Port ┃ VRAM (MB) ┃ State   ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
│ llama-3-1-8b     │ Llama 3.1 8B Instru… │ vllm    │ 8000 │      8192 │ stopped │
│ mistral-7b       │ Mistral 7B Instruct… │ vllm    │ 8000 │      6144 │ stopped │
│ bge-large        │ BGE Large EN v1.5    │ fastapi │ 9200 │      1024 │ stopped │
└──────────────────┴──────────────────────┴─────────┴──────┴───────────┴─────────┘
```

### 4. Create a mode

```bash
gpumod mode create "Chat Mode" \
  --services llama-3-1-8b,bge-large \
  --description "LLM chat with embedding retrieval"
```

```
Created mode chat-mode (Chat Mode).
  Services: llama-3-1-8b, bge-large
```

### 5. Simulate before switching

```bash
gpumod simulate mode chat-mode
```

```
Fits: 9216 / 24576 MB (headroom: 15360 MB)
                  Services
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┓
┃ ID             ┃ Name               ┃ Driver  ┃ VRAM (MB) ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━┩
│ llama-3-1-8b   │ Llama 3.1 8B Inst… │ vllm    │      8192 │
│ bge-large      │ BGE Large EN v1.5  │ fastapi │      1024 │
└────────────────┴────────────────────┴─────────┴───────────┘
```

### 6. Switch modes

```bash
gpumod mode switch chat-mode
```

```
Switched to mode chat-mode successfully.
Started:
  + llama-3-1-8b
  + bge-large
```

---

## CLI Reference

gpumod uses a subcommand structure. Every command supports `--help` for
detailed usage.

```bash
gpumod --help
```

### gpumod status

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

### gpumod init

Initialize the database and load preset configurations.

```bash
# Use default database path (~/.config/gpumod/gpumod.db)
gpumod init

# Specify a custom database path
gpumod init --db-path /path/to/custom.db

# Specify an additional preset directory
gpumod init --preset-dir /path/to/my/presets
```

### gpumod service

Manage individual GPU services.

#### service list

List all registered services with their current state.

```bash
gpumod service list
gpumod service list --json
```

#### service status

Show detailed status of a specific service.

```bash
gpumod service status llama-3-1-8b
gpumod service status llama-3-1-8b --json
```

Output includes service name, driver type, port, VRAM allocation, state,
uptime, and health check status.

#### service start

Start a registered service.

```bash
gpumod service start llama-3-1-8b
```

#### service stop

Stop a running service.

```bash
gpumod service stop llama-3-1-8b
```

### gpumod mode

Manage service modes -- named groups of services for specific use cases.

#### mode list

List all defined modes.

```bash
gpumod mode list
gpumod mode list --json
```

#### mode status

Show the currently active mode.

```bash
gpumod mode status
gpumod mode status --json
```

#### mode switch

Switch to a different mode. This starts services in the target mode and
stops services not in the target mode.

```bash
gpumod mode switch chat-mode
gpumod mode switch chat-mode --json
```

#### mode create

Create a new mode from existing services.

```bash
gpumod mode create "Coding Mode" \
  --services devstral-small-2,bge-large \
  --description "Code completion with embedding retrieval"
```

The mode ID is auto-generated from the name (e.g., "Coding Mode" becomes
`coding-mode`).

### gpumod template

Manage Jinja2 systemd unit file templates.

#### template list

List available template files.

```bash
gpumod template list
gpumod template list --json
```

#### template show

Show a rendered template with sample context.

```bash
gpumod template show vllm.service.j2
```

#### template generate

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

#### template install

Install a generated unit file to the systemd directory. Requires
confirmation via `--yes`.

```bash
# Preview first (no --yes)
gpumod template install llama-3-1-8b

# Install with confirmation
gpumod template install llama-3-1-8b --yes
```

The unit file is written to `/etc/systemd/system/gpumod-{service_id}.service`.

### gpumod model

Manage the ML model registry for VRAM estimation.

#### model list

List all registered models.

```bash
gpumod model list
gpumod model list --json
```

#### model info

Show detailed model information and VRAM estimates.

```bash
# Default context size (4096 tokens)
gpumod model info meta-llama/Llama-3.1-8B-Instruct

# Custom context size
gpumod model info meta-llama/Llama-3.1-8B-Instruct --context-size 32768
```

#### model register

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

#### model remove

Remove a model from the registry.

```bash
gpumod model remove meta-llama/Llama-3.1-8B-Instruct
```

### gpumod simulate

Simulate VRAM usage without starting or stopping services.

#### simulate mode

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

#### simulate services

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

### gpumod plan

AI-assisted VRAM allocation planning.

#### plan suggest

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

### gpumod tui

Launch an interactive terminal dashboard powered by
[Textual](https://textual.textualize.io/).

```bash
gpumod tui
```

The TUI displays a live GPU status bar, service list with state
indicators, a command input for `/status`, `/switch <mode>`,
`/simulate`, and `/quit`, and a footer with keyboard shortcuts.

Press `q` to quit or type `/help` for available commands.

---

## MCP Server Setup

gpumod includes an MCP (Model Context Protocol) server that lets AI
assistants like Claude Desktop manage GPU services directly.

### Claude Desktop Configuration

Add the following to your Claude Desktop MCP configuration file
(`~/.config/claude/claude_desktop_config.json` on Linux):

```json
{
  "mcpServers": {
    "gpumod": {
      "command": "python",
      "args": ["-m", "gpumod.mcp_main"],
      "env": {
        "GPUMOD_DB_PATH": "~/.config/gpumod/gpumod.db",
        "GPUMOD_MCP_RATE_LIMIT": "10"
      }
    }
  }
}
```

### Running the MCP server manually

```bash
python -m gpumod.mcp_main
```

The server starts in stdio mode by default, which is the standard
transport for MCP clients.

### Available MCP Tools

The MCP server exposes 9 tools:

| Tool | Description | Type |
|------|-------------|------|
| `gpu_status` | Get current GPU status, VRAM usage, running services | Read-only |
| `list_services` | List all registered services with driver type and VRAM | Read-only |
| `list_modes` | List all available GPU modes | Read-only |
| `service_info` | Get detailed info for a specific service | Read-only |
| `model_info` | Get model metadata and VRAM estimates | Read-only |
| `simulate_mode` | Simulate VRAM for a mode with optional changes | Read-only |
| `switch_mode` | Switch to a different GPU mode (starts/stops services) | Mutating |
| `start_service` | Start a specific service | Mutating |
| `stop_service` | Stop a specific service | Mutating |

Mutating tools are clearly marked in their descriptions and should
trigger confirmation prompts in MCP clients.

### Available MCP Resources

The MCP server provides 8 browsable resources:

| URI | Description |
|-----|-------------|
| `gpumod://help` | Overview of gpumod capabilities |
| `gpumod://config` | Current configuration and settings |
| `gpumod://modes` | List all defined modes |
| `gpumod://modes/{mode_id}` | Detail view of a specific mode |
| `gpumod://services` | List all registered services |
| `gpumod://services/{service_id}` | Detail view of a specific service |
| `gpumod://models` | List all registered models |
| `gpumod://models/{model_id}` | Detail view of a specific model |

---

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

---

## Configuration

All settings are configurable via environment variables with the `GPUMOD_`
prefix. Settings are managed by [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).

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

### Example: Using Ollama locally

```bash
export GPUMOD_LLM_BACKEND=ollama
export GPUMOD_LLM_BASE_URL=http://localhost:11434
export GPUMOD_LLM_MODEL=llama3.1
gpumod plan suggest
```

### Example: Custom database location

```bash
export GPUMOD_DB_PATH=/data/gpumod/services.db
gpumod init
```

---

## Presets

Presets are YAML files that define service configurations for repeatable
deployments. They are loaded during `gpumod init` and converted into
registered services.

### Preset directory structure

```
presets/
  llm/
    llama-3.1-8b.yaml
    mistral-7b.yaml
    qwen-2.5-72b-gguf.yaml
    devstral-small-2.yaml
  embedding/
    bge-large.yaml
    nomic-embed.yaml
```

### Preset YAML schema

Each preset file must conform to the `PresetConfig` schema:

```yaml
# Required fields
id: llama-3-1-8b              # Unique service identifier
name: Llama 3.1 8B Instruct   # Human-readable name
driver: vllm                   # Driver type: vllm, llamacpp, or fastapi
vram_mb: 8192                  # VRAM allocation in megabytes

# Optional fields
port: 8000                     # Service port number
context_size: 8192             # Context window size in tokens
kv_cache_per_1k: 32           # KV cache memory per 1000 tokens (MB)
model_id: meta-llama/Llama-3.1-8B-Instruct  # HuggingFace model ID
model_path: $HOME/models/model.gguf          # File path (env vars expanded)
health_endpoint: /health       # Health check endpoint (default: /health)
startup_timeout: 120           # Startup timeout in seconds (default: 60)
supports_sleep: true           # Whether the service supports sleep modes
sleep_mode: l1                 # Sleep mode: none, l1, l2, or router
unit_template: custom.j2       # Custom Jinja2 template name
unit_vars:                     # Variables passed to the systemd template
  gpu_mem_util: 0.9
  max_model_len: 8192
```

### Driver types

| Driver | Use Case | Template |
|--------|----------|----------|
| `vllm` | vLLM inference server | `vllm.service.j2` |
| `llamacpp` | llama.cpp server | `llamacpp.service.j2` |
| `fastapi` | Custom FastAPI server | `fastapi.service.j2` |
| `docker` | Docker container | N/A (uses Docker SDK) |

### Built-in presets

gpumod ships with example presets for common ML workloads:

| Preset | Driver | Model | VRAM |
|--------|--------|-------|------|
| `llama-3.1-8b` | vLLM | Llama 3.1 8B Instruct | 8 GB |
| `mistral-7b` | vLLM | Mistral 7B Instruct v0.3 | 6 GB |
| `devstral-small-2` | vLLM | Devstral Small 2505 | 15 GB |
| `qwen-2.5-72b-gguf` | llama.cpp | Qwen 2.5 72B Q4_K_M | 20 GB |
| `bge-large` | FastAPI | BGE Large EN v1.5 | 1 GB |
| `nomic-embed` | FastAPI | Nomic Embed Text v1.5 | 2 GB |

### Creating custom presets

1. Create a YAML file following the schema above.
2. Place it in a directory (e.g., `~/gpumod-presets/llm/my-model.yaml`).
3. Initialize with the custom directory:

```bash
gpumod init --preset-dir ~/gpumod-presets
```

Or set the environment variable:

```bash
export GPUMOD_PRESETS_DIR=~/gpumod-presets
gpumod init
```

### Example: vLLM preset

```yaml
id: llama-3-1-8b
name: Llama 3.1 8B Instruct
driver: vllm
port: 8000
vram_mb: 8192
context_size: 8192
kv_cache_per_1k: 32
model_id: meta-llama/Llama-3.1-8B-Instruct
health_endpoint: /health
startup_timeout: 120
supports_sleep: true
sleep_mode: l1
unit_vars:
  gpu_mem_util: 0.9
  max_model_len: 8192
```

### Example: llama.cpp GGUF preset

```yaml
id: qwen-2-5-72b-gguf
name: Qwen 2.5 72B GGUF Q4_K_M
driver: llamacpp
port: 8080
vram_mb: 20480
context_size: 32768
kv_cache_per_1k: 128
model_id: Qwen/Qwen2.5-72B-Instruct-GGUF
model_path: $HOME/models/qwen2.5-72b-instruct-q4_k_m.gguf
health_endpoint: /health
startup_timeout: 180
supports_sleep: true
sleep_mode: l2
unit_vars:
  n_gpu_layers: 80
  threads: 8
```

### Example: FastAPI embedding preset

```yaml
id: bge-large
name: BGE Large EN v1.5
driver: fastapi
port: 9200
vram_mb: 1024
model_id: BAAI/bge-large-en-v1.5
health_endpoint: /health
startup_timeout: 60
supports_sleep: false
sleep_mode: none
unit_vars:
  app_module: embedding_server:app
  working_dir: /opt/embedding
```

### Example: Docker container preset

```yaml
id: qdrant
name: Qdrant Vector DB
driver: docker
port: 6333
vram_mb: 0
health_endpoint: /healthz
startup_timeout: 30
extra_config:
  image: qdrant/qdrant:v1.11
  ports:
    - "6333:6333"
  environment:
    QDRANT__STORAGE__ON_DISK_PAYLOAD: "true"
```

Docker presets use `extra_config` for container settings (image, ports,
environment variables, volumes). The Docker driver enforces security
controls: no `--privileged`, no host network, no unsafe volume mounts,
and environment variable sanitization.

---

## Security Model

gpumod follows a defense-in-depth security approach. The full security
specification is in [`docs/SECURITY.md`](docs/SECURITY.md).

### Key principles

- **Input validation at every boundary** -- All service, mode, and model
  IDs are validated against strict regex patterns (`^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$`)
  before any business logic executes. This prevents shell injection, SQL
  injection, path traversal, and template injection attacks.

- **Error sanitization** -- Internal file paths, Python tracebacks, and
  module references are stripped from error messages before they reach MCP
  clients, preventing information disclosure.

- **Rate limiting** -- The MCP server enforces request rate limits
  (configurable via `GPUMOD_MCP_RATE_LIMIT`) to prevent resource
  exhaustion.

- **Tool classification** -- MCP tools are classified as read-only or
  mutating. Mutating tools (switch_mode, start_service, stop_service) are
  clearly marked so MCP clients can present confirmation prompts.

- **LLM security controls** -- API keys are stored as `SecretStr` and
  never logged. LLM output is advisory only and never auto-executed. Only
  minimal data (service IDs, VRAM numbers) is sent to LLM APIs.

- **Parameterized queries** -- All database operations use parameterized
  SQL queries via aiosqlite, preventing SQL injection as defense-in-depth.

- **Sandboxed templates** -- Jinja2 templates run in a
  `SandboxedEnvironment` with template name validation to prevent path
  traversal and code execution.

- **No shell=True** -- All subprocess calls use `create_subprocess_exec`
  with an allowlisted command set, never `shell=True`.

### Threat model summary

| Threat | Mitigation |
|--------|------------|
| Shell injection via IDs | Strict regex validation (SEC-V1) |
| SQL injection | Regex validation + parameterized queries (SEC-D2) |
| Path traversal | Regex validation + path validation (SEC-V1) |
| Template injection | Regex rejects `{` `}` in IDs + SandboxedEnvironment (SEC-D3) |
| Information disclosure | Error sanitization middleware (SEC-E1) |
| Unauthorized mutations | Tool classification + confirmation UI (SEC-A1) |
| Resource exhaustion | Rate limiting + alternatives cap (SEC-R1, SEC-R2) |
| LLM prompt injection | Prompt hardening + response validation (SEC-L1, SEC-L2) |
| API key leakage | SecretStr storage, never logged (SEC-L3) |
| LLM auto-execution | Advisory-only output (SEC-L4) |
| Data exfiltration to LLM | Data minimization (SEC-L5) |
| Docker privilege escalation | Privileged mode blocked (SEC-D7) |
| Container host network | Host/macvlan network blocked (SEC-D8) |
| Unsafe volume mounts | Critical host paths rejected (SEC-D9) |
| Env var injection | Env variable sanitization (SEC-D10) |

For the complete threat model, input validation spec, and implementation
checklist, see [`docs/SECURITY.md`](docs/SECURITY.md).

---

## Contributing

Contributions are welcome. Please follow these guidelines:

### Development setup

```bash
git clone https://github.com/jaigouk/gpumod.git
cd gpumod
pip install -e ".[dev]"
```

### Running tests

```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/gpumod --cov-fail-under=80

# Run only unit tests (skip integration and e2e)
pytest tests/ -v -m "not integration"

# Run E2E tests on GPU machines
pytest tests/e2e/ -v

# CPU-only CI (skip GPU/Docker tests)
pytest tests/ -v -m "not gpu_required and not docker_required"
```

### Code quality

gpumod enforces strict code quality via ruff, mypy, and pytest:

```bash
# Lint
ruff check src/ tests/

# Format check
ruff format --check src/ tests/

# Type check (strict mode)
mypy src/ --strict

# Full quality gate
ruff check src/ tests/ && \
  ruff format --check src/ tests/ && \
  mypy src/ --strict && \
  pytest tests/ -v --cov=src/gpumod --cov-fail-under=80
```

### Code style

- Python >= 3.11 with `from __future__ import annotations`
- Async-first: use `async`/`await` for I/O operations
- Pydantic v2 models with `ConfigDict(extra="forbid")`
- Type annotations on all functions (mypy strict mode)
- Docstrings in NumPy/Google style
- Import sorting via ruff (isort rules)
- Line length: 99 characters

### Pull requests

1. Create a feature branch from `main`.
2. Write tests for new functionality.
3. Ensure the full quality gate passes.
4. Submit a pull request with a clear description.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2024-2026 Jaigouk Kim
