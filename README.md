# gpumod

GPU Service Manager for ML workloads on Linux/NVIDIA systems.

gpumod manages vLLM, llama.cpp, FastAPI, and Docker-based inference services on
NVIDIA GPUs. It tracks VRAM allocation, supports mode-based service switching,
provides VRAM simulation before deployment, and exposes an MCP server for AI
assistant integration.

## Features

- **Service Management** -- Register, start, stop, and monitor GPU services
  with support for vLLM, llama.cpp, FastAPI, and Docker drivers
- **Mode Switching** -- Define named modes (e.g., "chat", "coding") that
  bundle services together and switch between them
- **VRAM Simulation** -- Simulate VRAM for any configuration before
  deployment, with alternative suggestions when capacity is exceeded
- **Model Registry** -- Track ML models with metadata from HuggingFace Hub
  or GGUF files, with automatic VRAM estimation
- **MCP Server** -- Expose GPU management as an MCP server for Claude Code,
  Cursor, Claude Desktop, and other MCP-compatible AI assistants
- **Template Engine** -- Generate and install systemd unit files from Jinja2
  templates, customized per driver type
- **AI Planning** -- LLM-assisted VRAM allocation suggestions (advisory only)
- **Interactive TUI** -- Terminal dashboard with live GPU status
- **Rich CLI** -- Beautiful output with tables, VRAM bar charts, and JSON mode
- **Host-Stability Doctor** -- Preflight checks (`gpumod doctor sysctl`,
  `gpumod doctor oom-protection`, `gpumod doctor venv`) that catch
  fragmentation-class freezes and operator-disconnect failures BEFORE
  they happen. Installable systemd drop-ins in
  [scripts/oom-protection/](scripts/oom-protection/) protect critical
  services (code-server, SSH) from being killed under memory pressure.

## Installation

Requires [uv](https://docs.astral.sh/uv/), Python >= 3.12, Linux with
NVIDIA GPU, and `nvidia-smi` in PATH.

```bash
git clone https://github.com/jaigouk/gpumod.git
cd gpumod
uv sync

# Install globally so `gpumod` is always on your PATH
uv tool install -e .
```

## Quick Start

```bash
# Initialize database and load presets
gpumod init

# Check GPU status
gpumod status

# List services
gpumod service list
```

## Deploying a Service

gpumod auto-generates systemd unit files from presets — no manual unit files needed.

```bash
# Enable user-level systemd lingering (one-time setup)
sudo loginctl enable-linger $USER

# Preview the generated unit file
gpumod template generate vllm-chat

# Install it to ~/.config/systemd/user/
gpumod template install vllm-chat --yes

# Start the service (uses systemctl --user, no sudo needed)
gpumod service start vllm-chat
```

See the [Getting Started](https://jaigouk.com/gpumod/getting-started/) guide
for full setup instructions.

## Mode Switching

Modes bundle services together and fit them within your VRAM budget.

```bash
# Simulate VRAM usage before switching
gpumod simulate mode coding-mode

# Switch modes (starts/stops services automatically)
gpumod mode switch coding-mode

# Launch interactive TUI
gpumod tui
```

## MCP Integration

gpumod exposes 16 tools and 8 resources via the
[Model Context Protocol](https://modelcontextprotocol.io/). Add it to your
IDE to let AI assistants query GPU status, simulate VRAM, switch modes,
discover models on HuggingFace, and consult an RLM-based reasoning engine
for complex questions like "Can I run Qwen3-235B on 24GB?".

```json
{
  "mcpServers": {
    "gpumod": {
      "command": "uv",
      "args": ["--directory", "/path/to/gpumod", "run", "python", "-m", "gpumod.mcp_main"],
      "env": {
        "OTEL_SDK_DISABLED": "true"
      }
    }
  }
}
```

> **Important:** gpumod depends on opentelemetry. Without `OTEL_SDK_DISABLED=true`,
> the SDK may print a startup message to stdout, which corrupts the JSON-RPC
> stream and causes MCP clients (Hermes, Claude Code, etc.) to fail with
> `Failed to parse JSONRPC message from server`.

See [MCP Integration](docs/user-guide/mcp.md) for setup instructions for Claude Code,
Cursor, Claude Desktop, and Antigravity.

## Configuration

All settings are configurable via environment variables with the `GPUMOD_`
prefix. A `.env.example` file is included in the repository root — copy it to
`.env` and uncomment the variables you want to override.

Key settings include preflight thresholds (RAM/VRAM), LLM backend
configuration, database path, and MCP rate limits. See
[Configuration](docs/getting-started/configuration.md) for the full list.

## Host Stability

On hosts where GPU services compete with desktop apps, browsers, and CI
runners, the dominant failure mode is `cudaHostAlloc` hanging the NVIDIA
driver when contiguous high-order pages are exhausted. gpumod ships three
layers of defense:

1. **Preflight RAMCheck** — refuses to start services when MemAvailable is
   below a safe floor (`model_size × 1.1 + 1024 MB`).
2. **`vm.min_free_kbytes=1 GiB`** — installer at
   [scripts/install-gpumod-sysctl.sh](scripts/install-gpumod-sysctl.sh)
   tells the kernel to keep more contiguous pages free at all times.
3. **`GGML_CUDA_NO_PINNED=1`** is set by default in the llamacpp systemd
   template — `cudaMallocHost` is bypassed, eliminating the freeze class
   entirely with ~0.3% TPS cost (measured 2026-05-26).
4. **Cgroup memory protection for code-server / SSH** — installer at
   [scripts/oom-protection/install.sh](scripts/oom-protection/install.sh)
   keeps the operator connected during heavy GPU loads.

After installation, run `gpumod doctor sysctl` and
`gpumod doctor oom-protection` to verify the protections are in place.

## Security

Input validation at every boundary, error sanitization, rate limiting,
parameterized queries, sandboxed templates, and no `shell=True`. See
[Security](docs/architecture/SECURITY.md) for the full threat model.

## Documentation

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/getting-started/cli.md) | All commands: status, service, mode, simulate, model, template, plan, tui |
| [MCP Integration](docs/user-guide/mcp.md) | MCP server setup for Claude Code, Cursor, Claude Desktop, Antigravity |
| [Configuration](docs/getting-started/configuration.md) | Environment variables, LLM backends, settings |
| [AI Planning](docs/user-guide/ai-planning.md) | LLM-assisted VRAM allocation planning |
| [Architecture](docs/architecture/index.md) | System design and component overview |
| [Security](docs/architecture/SECURITY.md) | Threat model, input validation, security controls |
| [Benchmarks](docs/benchmarks/README.md) | LLM benchmark framework and results |
| [Contributing](docs/contributing.md) | Development setup, tests, code quality, PR process |

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Jaigouk Kim
