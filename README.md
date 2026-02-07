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

## Installation

Requires [uv](https://docs.astral.sh/uv/), Python >= 3.12, Linux with
NVIDIA GPU, and `nvidia-smi` in PATH.

```bash
git clone https://github.com/jaigouk/gpumod.git
cd gpumod
uv sync
```

## Quick Start

```bash
# Initialize database and load presets
uv run gpumod init

# Check GPU status
uv run gpumod status

# List services
uv run gpumod service list

# Simulate VRAM usage before switching
uv run gpumod simulate mode coding-mode

# Switch modes
uv run gpumod mode switch coding-mode

# Launch interactive TUI
uv run gpumod tui
```

## MCP Integration

gpumod exposes 9 tools and 8 resources via the
[Model Context Protocol](https://modelcontextprotocol.io/). Add it to your
IDE to let AI assistants query GPU status, simulate VRAM, and switch modes.

```json
{
  "mcpServers": {
    "gpumod": {
      "command": "uv",
      "args": ["--directory", "/path/to/gpumod", "run", "python", "-m", "gpumod.mcp_main"]
    }
  }
}
```

See [docs/mcp.md](docs/mcp.md) for setup instructions for Claude Code,
Cursor, Claude Desktop, and Antigravity.

## Security

Input validation at every boundary, error sanitization, rate limiting,
parameterized queries, sandboxed templates, and no `shell=True`. See
[docs/SECURITY.md](docs/SECURITY.md) for the full threat model.

## Documentation

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/cli.md) | All commands: status, service, mode, simulate, model, template, plan, tui |
| [MCP Integration](docs/mcp.md) | MCP server setup for Claude Code, Cursor, Claude Desktop, Antigravity |
| [Configuration](docs/configuration.md) | Environment variables, LLM backends, settings |
| [Presets](docs/presets.md) | YAML preset schema, driver types, built-in examples |
| [AI Planning](docs/ai-planning.md) | LLM-assisted VRAM allocation planning |
| [Architecture](docs/ARCHITECTURE.md) | System design and component overview |
| [Security](docs/SECURITY.md) | Threat model, input validation, security controls |
| [Benchmarks](docs/benchmarks/README.md) | LLM benchmark framework and results |
| [Contributing](docs/contributing.md) | Development setup, tests, code quality, PR process |

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2024-2026 Jaigouk Kim
