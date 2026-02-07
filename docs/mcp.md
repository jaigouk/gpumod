# MCP Server Integration

gpumod includes an MCP (Model Context Protocol) server that lets AI
assistants manage GPU services directly. The server exposes tools for
querying status, simulating VRAM, and switching modes.

All IDE configurations below assume you cloned gpumod and installed it
with `uv sync`. Adjust `command` paths if you used pip instead.

## Claude Code

Claude Code discovers MCP servers from `.mcp.json` in the project root.
Create this file in your project (or home directory for global access):

```json
{
  "mcpServers": {
    "gpumod": {
      "command": "uv",
      "args": ["--directory", "/path/to/gpumod", "run", "python", "-m", "gpumod.mcp_main"],
      "env": {
        "GPUMOD_DB_PATH": "~/.config/gpumod/gpumod.db"
      }
    }
  }
}
```

Or add it via the CLI:

```bash
claude mcp add gpumod \
  -- uv --directory /path/to/gpumod run python -m gpumod.mcp_main
```

## Cursor

Cursor reads MCP configuration from `.cursor/mcp.json` in the project
root. Create the file:

```json
{
  "mcpServers": {
    "gpumod": {
      "command": "uv",
      "args": ["--directory", "/path/to/gpumod", "run", "python", "-m", "gpumod.mcp_main"],
      "env": {
        "GPUMOD_DB_PATH": "~/.config/gpumod/gpumod.db"
      }
    }
  }
}
```

After saving, restart the Cursor agent or open Settings > MCP to verify
the server is connected.

## Claude Desktop

Add to `~/.config/claude/claude_desktop_config.json` (Linux) or
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "gpumod": {
      "command": "uv",
      "args": ["--directory", "/path/to/gpumod", "run", "python", "-m", "gpumod.mcp_main"],
      "env": {
        "GPUMOD_DB_PATH": "~/.config/gpumod/gpumod.db"
      }
    }
  }
}
```

Restart Claude Desktop after editing the config.

## Antigravity (Google)

Antigravity stores MCP config in `mcp_config.json`. To edit it:

1. Open the **...** dropdown at the top of the agent panel
2. Click **Manage MCP Servers**
3. Click **View raw config**
4. Add the gpumod entry:

```json
{
  "mcpServers": {
    "gpumod": {
      "command": "uv",
      "args": ["--directory", "/path/to/gpumod", "run", "python", "-m", "gpumod.mcp_main"],
      "env": {
        "GPUMOD_DB_PATH": "~/.config/gpumod/gpumod.db"
      }
    }
  }
}
```

Save and the server will connect automatically.

## Running the MCP server manually

For testing or debugging, run the server directly:

```bash
cd /path/to/gpumod
uv run python -m gpumod.mcp_main
```

The server starts in stdio mode, which is the standard transport for
MCP clients. Set `GPUMOD_LOG_LEVEL=DEBUG` for verbose output.

## Available MCP Tools

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

## Available MCP Resources

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
