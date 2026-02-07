---
title: gpumod - GPU Service Manager for ML Workloads
description: Manage vLLM, llama.cpp, FastAPI, and Docker inference services on NVIDIA GPUs with VRAM simulation, mode switching, and MCP integration for AI assistants.
---

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

## Quick Start

```bash
# Clone and install
git clone https://github.com/jaigouk/gpumod.git
cd gpumod
uv sync
uv tool install -e .  # makes `gpumod` available globally

# Initialize database and load presets
gpumod init

# Check GPU status
gpumod status

# Deploy a service (auto-generates systemd unit file)
gpumod template generate vllm-chat
sudo gpumod template install vllm-chat --yes
gpumod service start vllm-chat

# Simulate VRAM usage before switching modes
gpumod simulate mode coding-mode

# Switch modes (starts/stops services automatically)
gpumod mode switch coding-mode

# Launch interactive TUI
gpumod tui
```

See the [Getting Started](getting-started.md) guide for sudoers configuration
and full deployment instructions.

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

See [MCP Integration](mcp.md) for setup instructions for Claude Code,
Cursor, Claude Desktop, and Antigravity.

## Requirements

- [uv](https://docs.astral.sh/uv/) >= 0.4
- Python >= 3.12
- Linux with NVIDIA GPU
- `nvidia-smi` in PATH

## License

Apache License 2.0 -- Copyright 2026 Jaigouk Kim
