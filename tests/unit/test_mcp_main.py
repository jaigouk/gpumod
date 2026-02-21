"""Tests for MCP server transport selection (gpumod-9z8).

Verifies that mcp_main.py reads GPUMOD_MCP_TRANSPORT, GPUMOD_MCP_HOST,
and GPUMOD_MCP_PORT env vars to dispatch between stdio and network transports.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestGetTransportConfig:
    """Test env-var-based transport configuration parsing."""

    def test_default_is_stdio(self) -> None:
        """No env vars → stdio transport, no host/port."""
        from gpumod.mcp_main import get_transport_config

        with patch.dict("os.environ", {}, clear=True):
            config = get_transport_config()

        assert config["transport"] == "stdio"
        assert "host" not in config
        assert "port" not in config

    def test_streamable_http_with_defaults(self) -> None:
        """GPUMOD_MCP_TRANSPORT=streamable-http → default host/port."""
        from gpumod.mcp_main import get_transport_config

        with patch.dict(
            "os.environ",
            {"GPUMOD_MCP_TRANSPORT": "streamable-http"},
            clear=True,
        ):
            config = get_transport_config()

        assert config["transport"] == "streamable-http"
        assert config["host"] == "127.0.0.1"
        assert config["port"] == 8808

    def test_sse_transport(self) -> None:
        """GPUMOD_MCP_TRANSPORT=sse → sse with default host/port."""
        from gpumod.mcp_main import get_transport_config

        with patch.dict("os.environ", {"GPUMOD_MCP_TRANSPORT": "sse"}, clear=True):
            config = get_transport_config()

        assert config["transport"] == "sse"
        assert config["host"] == "127.0.0.1"
        assert config["port"] == 8808

    def test_custom_host_and_port(self) -> None:
        """Custom GPUMOD_MCP_HOST and GPUMOD_MCP_PORT override defaults."""
        from gpumod.mcp_main import get_transport_config

        with patch.dict(
            "os.environ",
            {
                "GPUMOD_MCP_TRANSPORT": "streamable-http",
                "GPUMOD_MCP_HOST": "0.0.0.0",  # noqa: S104
                "GPUMOD_MCP_PORT": "9999",
            },
            clear=True,
        ):
            config = get_transport_config()

        assert config["transport"] == "streamable-http"
        assert config["host"] == "0.0.0.0"  # noqa: S104
        assert config["port"] == 9999

    def test_stdio_ignores_host_port(self) -> None:
        """When transport is stdio, host and port are not included."""
        from gpumod.mcp_main import get_transport_config

        with patch.dict(
            "os.environ",
            {
                "GPUMOD_MCP_TRANSPORT": "stdio",
                "GPUMOD_MCP_HOST": "0.0.0.0",  # noqa: S104
                "GPUMOD_MCP_PORT": "9999",
            },
            clear=True,
        ):
            config = get_transport_config()

        assert config["transport"] == "stdio"
        assert "host" not in config
        assert "port" not in config

    def test_empty_transport_defaults_to_stdio(self) -> None:
        """Empty GPUMOD_MCP_TRANSPORT string → treated as stdio."""
        from gpumod.mcp_main import get_transport_config

        with patch.dict("os.environ", {"GPUMOD_MCP_TRANSPORT": ""}, clear=True):
            config = get_transport_config()

        assert config["transport"] == "stdio"

    def test_invalid_port_raises_value_error(self) -> None:
        """Non-numeric GPUMOD_MCP_PORT → ValueError."""
        import pytest

        from gpumod.mcp_main import get_transport_config

        with (
            patch.dict(
                "os.environ",
                {
                    "GPUMOD_MCP_TRANSPORT": "streamable-http",
                    "GPUMOD_MCP_PORT": "not-a-number",
                },
                clear=True,
            ),
            pytest.raises(ValueError, match="GPUMOD_MCP_PORT"),
        ):
            get_transport_config()

    def test_http_transport(self) -> None:
        """GPUMOD_MCP_TRANSPORT=http → http with default host/port."""
        from gpumod.mcp_main import get_transport_config

        with patch.dict("os.environ", {"GPUMOD_MCP_TRANSPORT": "http"}, clear=True):
            config = get_transport_config()

        assert config["transport"] == "http"
        assert config["host"] == "127.0.0.1"
        assert config["port"] == 8808


class TestRunServer:
    """Test that run_server dispatches correctly to server.run()."""

    def test_stdio_dispatch(self) -> None:
        """stdio config → server.run(transport='stdio')."""
        from gpumod.mcp_main import run_server

        mock_server = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            run_server(mock_server)

        mock_server.run.assert_called_once_with(transport="stdio")

    def test_streamable_http_dispatch(self) -> None:
        """streamable-http config → server.run with transport, host, port."""
        from gpumod.mcp_main import run_server

        mock_server = MagicMock()
        with patch.dict(
            "os.environ",
            {"GPUMOD_MCP_TRANSPORT": "streamable-http"},
            clear=True,
        ):
            run_server(mock_server)

        mock_server.run.assert_called_once_with(
            transport="streamable-http", host="127.0.0.1", port=8808
        )

    def test_sse_dispatch_custom_port(self) -> None:
        """sse + custom port → server.run with correct kwargs."""
        from gpumod.mcp_main import run_server

        mock_server = MagicMock()
        with patch.dict(
            "os.environ",
            {
                "GPUMOD_MCP_TRANSPORT": "sse",
                "GPUMOD_MCP_PORT": "7777",
            },
            clear=True,
        ):
            run_server(mock_server)

        mock_server.run.assert_called_once_with(transport="sse", host="127.0.0.1", port=7777)
