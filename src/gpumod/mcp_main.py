"""Entry point for the gpumod MCP server.

Usage::

    python -m gpumod.mcp_main

Starts the server in stdio mode by default, suitable for integration with
MCP clients such as Claude Desktop.
"""

from __future__ import annotations

from gpumod.mcp_server import create_mcp_server

server = create_mcp_server()

if __name__ == "__main__":
    server.run()
