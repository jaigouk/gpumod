"""Entry point for the gpumod MCP server.

Usage::

    python -m gpumod.mcp_main

Starts the server in stdio mode by default, suitable for integration with
MCP clients such as Claude Desktop. Set GPUMOD_MCP_TRANSPORT to
"streamable-http" or "sse" for network transports (e.g. systemd service).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from gpumod.mcp_server import create_mcp_server

if TYPE_CHECKING:
    from fastmcp import FastMCP

_NETWORK_TRANSPORTS = {"streamable-http", "sse", "http"}


def get_transport_config() -> dict[str, Any]:
    """Read transport configuration from environment variables.

    Returns
    -------
    dict
        Keys: ``transport`` (always present), plus ``host`` and ``port``
        for network transports.

    Raises
    ------
    ValueError
        If GPUMOD_MCP_PORT is not a valid integer.
    """
    transport = os.environ.get("GPUMOD_MCP_TRANSPORT", "stdio").strip()
    if not transport:
        transport = "stdio"

    if transport in _NETWORK_TRANSPORTS:
        host = os.environ.get("GPUMOD_MCP_HOST", "127.0.0.1")
        port_str = os.environ.get("GPUMOD_MCP_PORT", "8808")
        try:
            port = int(port_str)
        except ValueError:
            msg = f"GPUMOD_MCP_PORT must be an integer, got: {port_str!r}"
            raise ValueError(msg) from None
        return {"transport": transport, "host": host, "port": port}

    return {"transport": transport}


def run_server(server: FastMCP[Any]) -> None:
    """Run the MCP server with transport from environment variables."""
    config = get_transport_config()
    server.run(**config)


server = create_mcp_server()

if __name__ == "__main__":
    run_server(server)
