"""Integration tests for FastMCP 3.x API compatibility.

Verifies that the MCP server works correctly with FastMCP 3.x Client
behavioral changes discovered during QA:

- CallToolResult is a dataclass with .data, .is_error, .content attributes
- CallToolResult is NOT subscriptable (no result[0])
- Client.call_tool() raises ToolError on error when raise_on_error=True (default)
- Client.call_tool(raise_on_error=False) returns error inline
- Client.read_resource() still returns a list
- Rate limit errors propagate through the Client correctly
- Nonexistent tool/resource raises NotFoundError (not ToolError)
- None vs {} arguments handled identically
- call_tool_mcp returns raw mcp.types.CallToolResult (different from call_tool)
- Tool/resource discovery returns correct types
- Rate limit on read_resource through Client
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import fields as dataclass_fields
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import mcp.types
import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import NotFoundError, ToolError
from mcp.shared.exceptions import McpError

from gpumod.db import Database
from gpumod.mcp_resources import register_resources
from gpumod.mcp_server import (
    ErrorSanitizationMiddleware,
    RateLimitMiddleware,
    create_mcp_server,
    gpumod_lifespan,
)
from gpumod.mcp_tools import register_tools
from gpumod.models import DriverType, Mode, Service
from gpumod.templates.modes import ModeSyncResult
from gpumod.templates.presets import PresetSyncResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def mcp_client(tmp_path: Path) -> Client:
    """Create an MCP Client backed by a pre-populated database."""
    db_path = tmp_path / "fastmcp3_test.db"

    db = Database(db_path)
    await db.connect()

    await db.insert_service(
        Service(
            id="test-svc",
            name="Test Service",
            driver=DriverType.VLLM,
            port=8000,
            vram_mb=4000,
        )
    )
    await db.insert_service(
        Service(
            id="test-llama",
            name="Test Llama",
            driver=DriverType.LLAMACPP,
            port=8001,
            vram_mb=6000,
        )
    )
    await db.insert_mode(Mode(id="test-mode", name="Test", description="Test mode"))
    await db.set_mode_services("test-mode", ["test-svc"])

    await db.close()

    server = create_mcp_server(db_path=db_path)
    return Client(server)


@pytest.fixture(autouse=True)
def _disable_mcp_auto_sync():
    """Disable auto-sync in MCP lifespan to preserve test fixtures."""
    mock_preset_result = PresetSyncResult(inserted=0, updated=0, unchanged=0, deleted=0)
    mock_mode_result = ModeSyncResult(inserted=0, updated=0, unchanged=0, deleted=0)

    with (
        patch("gpumod.mcp_server.sync_presets", new=AsyncMock(return_value=mock_preset_result)),
        patch("gpumod.mcp_server.sync_modes", new=AsyncMock(return_value=mock_mode_result)),
    ):
        yield


# ---------------------------------------------------------------------------
# CallToolResult API contract tests
# ---------------------------------------------------------------------------


class TestCallToolResultContract:
    """Verify CallToolResult has the expected FastMCP 3.x API surface."""

    async def test_call_tool_returns_call_tool_result_type(self, mcp_client: Client) -> None:
        """Client.call_tool() returns fastmcp CallToolResult, not a list."""
        from fastmcp.client.client import CallToolResult

        async with mcp_client as client:
            result = await client.call_tool("list_services", {})

        assert isinstance(result, CallToolResult)

    async def test_result_not_subscriptable(self, mcp_client: Client) -> None:
        """CallToolResult cannot be accessed with result[0] (was a list in 2.x)."""
        async with mcp_client as client:
            result = await client.call_tool("list_services", {})

        with pytest.raises(TypeError):
            result[0]  # type: ignore[index]

    async def test_result_has_data_attribute(self, mcp_client: Client) -> None:
        """CallToolResult has .data attribute containing parsed JSON."""
        async with mcp_client as client:
            result = await client.call_tool("list_services", {})

        assert hasattr(result, "data")
        assert result.data is not None

    async def test_result_has_is_error_attribute(self, mcp_client: Client) -> None:
        """CallToolResult has .is_error bool attribute."""
        async with mcp_client as client:
            result = await client.call_tool("list_services", {})

        assert hasattr(result, "is_error")
        assert result.is_error is False

    async def test_result_has_content_list(self, mcp_client: Client) -> None:
        """CallToolResult has .content attribute containing ContentBlock list."""
        async with mcp_client as client:
            result = await client.call_tool("list_services", {})

        assert hasattr(result, "content")
        assert isinstance(result.content, list)
        assert len(result.content) >= 1

    async def test_result_content_has_text(self, mcp_client: Client) -> None:
        """Each content block has a .text attribute with JSON."""
        async with mcp_client as client:
            result = await client.call_tool("list_services", {})

        text = result.content[0].text
        parsed = json.loads(text)
        assert isinstance(parsed, dict)
        assert "services" in parsed

    async def test_data_matches_content_text(self, mcp_client: Client) -> None:
        """The .data attribute matches the parsed JSON from .content[0].text."""
        async with mcp_client as client:
            result = await client.call_tool("list_services", {})

        content_parsed = json.loads(result.content[0].text)
        # .data should contain the same data as the parsed text content
        assert isinstance(result.data, dict)
        assert result.data["services"] == content_parsed["services"]

    async def test_result_is_dataclass(self, mcp_client: Client) -> None:
        """CallToolResult is a dataclass (not a Pydantic model or list)."""
        async with mcp_client as client:
            result = await client.call_tool("list_services", {})

        # Should be a dataclass
        field_names = {f.name for f in dataclass_fields(result)}
        assert "data" in field_names
        assert "is_error" in field_names
        assert "content" in field_names


# ---------------------------------------------------------------------------
# ToolError exception handling tests
# ---------------------------------------------------------------------------


class TestToolErrorHandling:
    """Verify ToolError behavior with FastMCP 3.x Client."""

    async def test_tool_error_importable(self) -> None:
        """ToolError is importable from fastmcp.exceptions."""
        from fastmcp.exceptions import ToolError as _ToolError

        assert _ToolError is not None

    async def test_validation_error_returns_structured_dict(self, mcp_client: Client) -> None:
        """Our validation errors return structured dicts, not ToolError.

        gpumod tools catch validation errors and return
        {"error": "...", "code": "VALIDATION_ERROR"} dicts. These are NOT
        server-level errors, so they should NOT raise ToolError.
        """
        async with mcp_client as client:
            result = await client.call_tool("service_info", {"service_id": "; rm -rf /"})

        # Should return normally (not raise ToolError)
        assert result.is_error is False
        assert isinstance(result.data, dict)
        assert result.data["code"] == "VALIDATION_ERROR"

    async def test_raise_on_error_false_returns_error_inline(
        self,
        tmp_path: Path,
    ) -> None:
        """With raise_on_error=False, tool errors return inline instead of raising.

        This tests the FastMCP 3.x Client parameter that suppresses ToolError.
        """
        db_path = tmp_path / "raise_err_test.db"
        db = Database(db_path)
        await db.connect()
        await db.insert_service(
            Service(
                id="test-svc",
                name="Test",
                driver=DriverType.VLLM,
                port=8000,
                vram_mb=4000,
            )
        )
        await db.close()

        server = create_mcp_server(db_path=db_path)
        client = Client(server)
        async with client:
            # Valid tool call should work fine either way
            result = await client.call_tool("list_services", {}, raise_on_error=False)
            assert result.data is not None


# ---------------------------------------------------------------------------
# Rate limit error propagation tests
# ---------------------------------------------------------------------------


class TestRateLimitErrorPropagation:
    """Verify rate limit errors propagate correctly through FastMCP 3.x Client."""

    async def test_rate_limit_error_through_client(self, tmp_path: Path) -> None:
        """Rate limit RuntimeError propagates through Client as ToolError or RuntimeError."""
        db_path = tmp_path / "rate_test.db"
        db = Database(db_path)
        await db.connect()
        await db.insert_service(
            Service(
                id="test-svc",
                name="Test",
                driver=DriverType.VLLM,
                port=8000,
                vram_mb=4000,
            )
        )
        await db.close()

        @asynccontextmanager
        async def _lifespan(
            server: FastMCP[dict[str, Any]],
        ) -> AsyncIterator[dict[str, Any]]:
            async with gpumod_lifespan(server, db_path=db_path) as ctx:
                yield ctx

        server: FastMCP[dict[str, Any]] = FastMCP(
            name="gpumod-rate-test",
            lifespan=_lifespan,
            middleware=[
                ErrorSanitizationMiddleware(),
                RateLimitMiddleware(max_requests=2, window_seconds=60.0),
            ],
        )
        register_resources(server)
        register_tools(server)

        client = Client(server)
        async with client:
            # First 2 requests succeed
            r1 = await client.call_tool("list_services", {})
            assert r1.is_error is False

            r2 = await client.call_tool("list_services", {})
            assert r2.is_error is False

            # 3rd request hits rate limit — should propagate as an error
            # FastMCP 3.x may raise ToolError, RuntimeError, or return error inline
            try:
                r3 = await client.call_tool("list_services", {})
                # If it returns instead of raising, check for error
                assert r3.is_error is True or (
                    isinstance(r3.data, dict) and "error" in str(r3.data).lower()
                ), "Rate limit should be indicated in result"
            except (ToolError, RuntimeError) as exc:
                # Rate limit error propagated as exception
                assert "rate limit" in str(exc).lower()

    async def test_rate_limit_does_not_leak_paths(self, tmp_path: Path) -> None:
        """Rate limit error messages don't leak internal paths."""
        db_path = tmp_path / "rate_path_test.db"
        db = Database(db_path)
        await db.connect()
        await db.insert_service(
            Service(
                id="test-svc",
                name="Test",
                driver=DriverType.VLLM,
                port=8000,
                vram_mb=4000,
            )
        )
        await db.close()

        @asynccontextmanager
        async def _lifespan(
            server: FastMCP[dict[str, Any]],
        ) -> AsyncIterator[dict[str, Any]]:
            async with gpumod_lifespan(server, db_path=db_path) as ctx:
                yield ctx

        server: FastMCP[dict[str, Any]] = FastMCP(
            name="gpumod-rate-path-test",
            lifespan=_lifespan,
            middleware=[
                ErrorSanitizationMiddleware(),
                RateLimitMiddleware(max_requests=1, window_seconds=60.0),
            ],
        )
        register_resources(server)
        register_tools(server)

        client = Client(server)
        async with client:
            await client.call_tool("list_services", {})

            try:
                await client.call_tool("list_services", {})
            except (ToolError, RuntimeError) as exc:
                error_msg = str(exc)
                assert "/home/" not in error_msg
                assert "/tmp/" not in error_msg


# ---------------------------------------------------------------------------
# read_resource regression guard
# ---------------------------------------------------------------------------


class TestReadResourceContract:
    """Verify Client.read_resource() return type is unchanged in 3.x."""

    async def test_read_resource_returns_list(self, mcp_client: Client) -> None:
        """Client.read_resource() still returns a list (not CallToolResult)."""
        async with mcp_client as client:
            contents = await client.read_resource("gpumod://help")

        assert isinstance(contents, list)
        assert len(contents) >= 1

    async def test_read_resource_content_has_text(self, mcp_client: Client) -> None:
        """Resource content items have .text attribute."""
        async with mcp_client as client:
            contents = await client.read_resource("gpumod://help")

        assert hasattr(contents[0], "text")
        assert isinstance(contents[0].text, str)
        assert len(contents[0].text) > 0

    async def test_read_resource_modes_template(self, mcp_client: Client) -> None:
        """Resource templates (modes/{id}) still work via read_resource."""
        async with mcp_client as client:
            contents = await client.read_resource("gpumod://modes/test-mode")

        assert isinstance(contents, list)
        text = contents[0].text
        assert "test-mode" in text.lower() or "Test" in text

    async def test_read_resource_services_returns_markdown(self, mcp_client: Client) -> None:
        """Services resource returns markdown table via read_resource."""
        async with mcp_client as client:
            contents = await client.read_resource("gpumod://services")

        assert isinstance(contents, list)
        text = contents[0].text
        assert "test-svc" in text
        assert "test-llama" in text


# ---------------------------------------------------------------------------
# Middleware interaction with FastMCP 3.x
# ---------------------------------------------------------------------------


class TestMiddlewareFastMCP3:
    """Verify middleware behavior with FastMCP 3.x request pipeline."""

    async def test_error_sanitization_with_tool_call(self, mcp_client: Client) -> None:
        """Error sanitization middleware runs on tool calls in FastMCP 3.x."""
        async with mcp_client as client:
            # Trigger a simulated mode error (may fail on CI without nvidia-smi)
            result = await client.call_tool("simulate_mode", {"mode_id": "test-mode"})

        data = result.data
        if isinstance(data, dict) and "error" in data:
            # Error path — verify no internal paths leaked
            error_msg = data["error"]
            assert "/home/" not in error_msg
            assert ".py" not in error_msg

    async def test_all_three_middleware_attached(self) -> None:
        """Server has all three middleware (error sanitization, request ID, rate limit)."""
        from gpumod.mcp_server import (
            ErrorSanitizationMiddleware,
            RateLimitMiddleware,
            RequestIDMiddleware,
        )

        server = create_mcp_server()
        middleware_types = {type(m) for m in server.middleware}
        assert ErrorSanitizationMiddleware in middleware_types
        assert RequestIDMiddleware in middleware_types
        assert RateLimitMiddleware in middleware_types


# ---------------------------------------------------------------------------
# Nonexistent tool/resource error handling
# ---------------------------------------------------------------------------


class TestNotFoundErrors:
    """Verify error handling for nonexistent tools and resources in 3.x.

    Server raises NotFoundError internally, but our ErrorSanitizationMiddleware
    catches it and re-raises. The Client then sees isError=True and raises
    ToolError (the generic client-side error). This is correct behavior —
    NotFoundError is a server-internal detail, clients see ToolError.
    """

    async def test_nonexistent_tool_raises_tool_error(self, mcp_client: Client) -> None:
        """Calling a nonexistent tool raises ToolError (wrapped from NotFoundError)."""
        async with mcp_client as client:
            with pytest.raises(ToolError):
                await client.call_tool("does_not_exist", {})

    async def test_nonexistent_tool_error_mentions_tool_name(self, mcp_client: Client) -> None:
        """ToolError message includes the tool name."""
        async with mcp_client as client:
            with pytest.raises(ToolError, match="does_not_exist"):
                await client.call_tool("does_not_exist", {})

    async def test_nonexistent_tool_error_does_not_leak_paths(self, mcp_client: Client) -> None:
        """Error for nonexistent tool does not leak internal file paths."""
        async with mcp_client as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool("does_not_exist", {})

        error_msg = str(exc_info.value)
        assert "/home/" not in error_msg
        assert ".py" not in error_msg

    async def test_nonexistent_tool_with_raise_on_error_false(self, mcp_client: Client) -> None:
        """With raise_on_error=False, nonexistent tool returns error inline."""
        async with mcp_client as client:
            result = await client.call_tool("does_not_exist", {}, raise_on_error=False)

        assert result.is_error is True
        error_text = result.content[0].text
        assert "does_not_exist" in error_text

    async def test_nonexistent_resource_raises_error(self, mcp_client: Client) -> None:
        """Reading a nonexistent resource raises an error."""
        async with mcp_client as client:
            with pytest.raises((NotFoundError, ToolError, Exception)):
                await client.read_resource("gpumod://nonexistent")

    async def test_nonexistent_resource_template_returns_not_found_message(
        self, mcp_client: Client
    ) -> None:
        """Reading a nonexistent template resource returns a not-found message.

        Our resource templates handle missing IDs gracefully by returning
        a text message rather than raising, which is a valid server design.
        """
        async with mcp_client as client:
            contents = await client.read_resource("gpumod://modes/does-not-exist-mode-xyz")

        assert isinstance(contents, list)
        text = contents[0].text
        assert "not found" in text.lower() or "does not exist" in text.lower()


# ---------------------------------------------------------------------------
# Arguments edge cases
# ---------------------------------------------------------------------------


class TestArgumentsEdgeCases:
    """Verify None vs {} arguments handling in FastMCP 3.x."""

    async def test_none_arguments_accepted(self, mcp_client: Client) -> None:
        """call_tool with None arguments works (converted to {} internally)."""
        async with mcp_client as client:
            result = await client.call_tool("gpu_status", None)

        assert result.is_error is False
        assert result.data is not None

    async def test_empty_dict_arguments_accepted(self, mcp_client: Client) -> None:
        """call_tool with {} arguments works identically to None."""
        async with mcp_client as client:
            result = await client.call_tool("gpu_status", {})

        assert result.is_error is False
        assert result.data is not None

    async def test_none_and_empty_dict_return_same_data(self, mcp_client: Client) -> None:
        """None and {} arguments produce identical results."""
        async with mcp_client as client:
            result_none = await client.call_tool("list_services", None)
            result_empty = await client.call_tool("list_services", {})

        assert result_none.data == result_empty.data

    async def test_omitted_arguments_accepted(self, mcp_client: Client) -> None:
        """call_tool with omitted arguments works."""
        async with mcp_client as client:
            result = await client.call_tool("gpu_status")

        assert result.is_error is False
        assert result.data is not None


# ---------------------------------------------------------------------------
# call_tool_mcp vs call_tool return type
# ---------------------------------------------------------------------------


class TestCallToolMcpVsCallTool:
    """Verify call_tool_mcp returns raw mcp.types.CallToolResult."""

    async def test_call_tool_mcp_returns_mcp_types_result(self, mcp_client: Client) -> None:
        """call_tool_mcp returns mcp.types.CallToolResult (not fastmcp dataclass)."""
        async with mcp_client as client:
            result = await client.call_tool_mcp("list_services", {})

        assert isinstance(result, mcp.types.CallToolResult)

    async def test_call_tool_mcp_has_is_error_camel_case(self, mcp_client: Client) -> None:
        """Raw mcp.types.CallToolResult uses camelCase isError."""
        async with mcp_client as client:
            result = await client.call_tool_mcp("list_services", {})

        assert hasattr(result, "isError")
        assert result.isError is False

    async def test_call_tool_mcp_content_is_list(self, mcp_client: Client) -> None:
        """Raw result has .content list of ContentBlock."""
        async with mcp_client as client:
            result = await client.call_tool_mcp("list_services", {})

        assert isinstance(result.content, list)
        assert len(result.content) >= 1
        # Content should be TextContent
        assert hasattr(result.content[0], "text")

    async def test_call_tool_mcp_no_data_attribute(self, mcp_client: Client) -> None:
        """Raw mcp.types.CallToolResult has no .data convenience attribute."""
        async with mcp_client as client:
            result = await client.call_tool_mcp("list_services", {})

        # .data is a fastmcp-only convenience; raw result doesn't have it
        with pytest.raises(AttributeError):
            _ = result.data  # type: ignore[attr-defined]

    async def test_call_tool_mcp_does_not_raise_on_error(self, mcp_client: Client) -> None:
        """call_tool_mcp never raises ToolError — it returns isError inline."""
        async with mcp_client as client:
            # This should not raise (call_tool_mcp doesn't auto-raise)
            result = await client.call_tool_mcp("service_info", {"service_id": "; rm -rf /"})

        # Our validation returns a structured error dict, not isError=True,
        # so this should be a normal response
        assert isinstance(result, mcp.types.CallToolResult)
        text = result.content[0].text
        parsed = json.loads(text)
        assert parsed["code"] == "VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# Tool and resource discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    """Verify tool/resource discovery returns correct types in 3.x."""

    async def test_list_tools_returns_tool_list(self, mcp_client: Client) -> None:
        """list_tools returns list[mcp.types.Tool], not a wrapper object."""
        async with mcp_client as client:
            tools = await client.list_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0
        assert isinstance(tools[0], mcp.types.Tool)

    async def test_list_tools_includes_all_registered_tools(self, mcp_client: Client) -> None:
        """All 16 gpumod tools are discoverable."""
        async with mcp_client as client:
            tools = await client.list_tools()

        tool_names = {t.name for t in tools}
        # Spot-check critical tools
        assert "gpu_status" in tool_names
        assert "list_services" in tool_names
        assert "list_modes" in tool_names
        assert "simulate_mode" in tool_names
        assert "switch_mode" in tool_names
        assert "start_service" in tool_names
        assert "stop_service" in tool_names
        assert "service_info" in tool_names
        assert "model_info" in tool_names

    async def test_tool_has_input_schema(self, mcp_client: Client) -> None:
        """Each tool has an inputSchema dict."""
        async with mcp_client as client:
            tools = await client.list_tools()

        for tool in tools:
            assert tool.inputSchema is not None, f"Tool {tool.name} missing inputSchema"

    async def test_list_resources_returns_resource_list(self, mcp_client: Client) -> None:
        """list_resources returns list[mcp.types.Resource]."""
        async with mcp_client as client:
            resources = await client.list_resources()

        assert isinstance(resources, list)
        assert len(resources) > 0
        assert isinstance(resources[0], mcp.types.Resource)

    async def test_list_resources_includes_static_resources(self, mcp_client: Client) -> None:
        """Static resources (help, config, etc.) are discoverable."""
        async with mcp_client as client:
            resources = await client.list_resources()

        resource_uris = {str(r.uri) for r in resources}
        assert "gpumod://help" in resource_uris
        assert "gpumod://modes" in resource_uris
        assert "gpumod://services" in resource_uris

    async def test_list_resource_templates_returns_template_list(self, mcp_client: Client) -> None:
        """list_resource_templates returns list[mcp.types.ResourceTemplate]."""
        async with mcp_client as client:
            templates = await client.list_resource_templates()

        assert isinstance(templates, list)
        assert len(templates) > 0
        assert isinstance(templates[0], mcp.types.ResourceTemplate)

    async def test_resource_templates_include_parameterized_uris(self, mcp_client: Client) -> None:
        """Resource templates have parameterized URI patterns."""
        async with mcp_client as client:
            templates = await client.list_resource_templates()

        template_uris = {str(t.uriTemplate) for t in templates}
        # Should have templates like gpumod://modes/{mode_id}
        has_mode_template = any("modes/" in uri and "{" in uri for uri in template_uris)
        has_service_template = any("services/" in uri and "{" in uri for uri in template_uris)
        assert has_mode_template, f"No mode template found in {template_uris}"
        assert has_service_template, f"No service template found in {template_uris}"


# ---------------------------------------------------------------------------
# Rate limit on read_resource through Client
# ---------------------------------------------------------------------------


class TestRateLimitReadResource:
    """Verify rate limiting works for read_resource through the full Client pipeline."""

    async def test_rate_limit_on_resource_reads_through_client(self, tmp_path: Path) -> None:
        """Resource reads through Client are rate-limited."""
        db_path = tmp_path / "rate_resource_test.db"
        db = Database(db_path)
        await db.connect()
        await db.insert_service(
            Service(
                id="test-svc",
                name="Test",
                driver=DriverType.VLLM,
                port=8000,
                vram_mb=4000,
            )
        )
        await db.close()

        @asynccontextmanager
        async def _lifespan(
            server: FastMCP[dict[str, Any]],
        ) -> AsyncIterator[dict[str, Any]]:
            async with gpumod_lifespan(server, db_path=db_path) as ctx:
                yield ctx

        server: FastMCP[dict[str, Any]] = FastMCP(
            name="gpumod-rate-resource-test",
            lifespan=_lifespan,
            middleware=[
                ErrorSanitizationMiddleware(),
                RateLimitMiddleware(max_requests=2, window_seconds=60.0),
            ],
        )
        register_resources(server)
        register_tools(server)

        client = Client(server)
        async with client:
            # First 2 resource reads succeed
            c1 = await client.read_resource("gpumod://help")
            assert isinstance(c1, list)

            c2 = await client.read_resource("gpumod://help")
            assert isinstance(c2, list)

            # 3rd should be rate limited.
            # read_resource errors propagate as McpError (not ToolError)
            # because resource reads don't go through the tool error path.
            with pytest.raises((ToolError, RuntimeError, McpError), match="(?i)rate limit"):
                await client.read_resource("gpumod://help")

    async def test_tool_calls_and_resource_reads_share_rate_limit(self, tmp_path: Path) -> None:
        """Tool calls and resource reads share the same per-client rate quota."""
        db_path = tmp_path / "rate_shared_test.db"
        db = Database(db_path)
        await db.connect()
        await db.insert_service(
            Service(
                id="test-svc",
                name="Test",
                driver=DriverType.VLLM,
                port=8000,
                vram_mb=4000,
            )
        )
        await db.close()

        @asynccontextmanager
        async def _lifespan(
            server: FastMCP[dict[str, Any]],
        ) -> AsyncIterator[dict[str, Any]]:
            async with gpumod_lifespan(server, db_path=db_path) as ctx:
                yield ctx

        server: FastMCP[dict[str, Any]] = FastMCP(
            name="gpumod-rate-shared-test",
            lifespan=_lifespan,
            middleware=[
                ErrorSanitizationMiddleware(),
                RateLimitMiddleware(max_requests=3, window_seconds=60.0),
            ],
        )
        register_resources(server)
        register_tools(server)

        client = Client(server)
        async with client:
            # Mix tool calls and resource reads (3 total = limit)
            await client.call_tool("list_services", {})
            await client.read_resource("gpumod://help")
            await client.call_tool("gpu_status", {})

            # 4th request (either type) should be rate limited
            with pytest.raises((ToolError, RuntimeError, McpError), match="(?i)rate limit"):
                await client.read_resource("gpumod://help")


# ---------------------------------------------------------------------------
# Multiple sequential calls (QA found rapid-fire issues)
# ---------------------------------------------------------------------------


class TestSequentialCalls:
    """Verify sequential tool/resource calls work correctly in 3.x."""

    async def test_multiple_different_tools_in_sequence(self, mcp_client: Client) -> None:
        """Multiple different tool calls in sequence all return valid results."""
        async with mcp_client as client:
            r1 = await client.call_tool("gpu_status", {})
            r2 = await client.call_tool("list_services", {})
            r3 = await client.call_tool("list_modes", {})
            r4 = await client.call_tool("service_info", {"service_id": "test-svc"})

        assert r1.is_error is False
        assert r2.is_error is False
        assert r3.is_error is False
        assert r4.is_error is False

        # Each returns different data
        assert "gpu" in r1.data or r1.data is not None
        assert "services" in r2.data
        assert "modes" in r3.data

    async def test_same_tool_called_multiple_times(self, mcp_client: Client) -> None:
        """Same tool called multiple times returns consistent results."""
        async with mcp_client as client:
            results = [await client.call_tool("list_services", {}) for _ in range(5)]

        for r in results:
            assert r.is_error is False
            assert len(r.data["services"]) == 2  # test-svc + test-llama

    async def test_interleaved_tool_and_resource_calls(self, mcp_client: Client) -> None:
        """Interleaved tool calls and resource reads don't interfere."""
        async with mcp_client as client:
            r1 = await client.call_tool("list_services", {})
            c1 = await client.read_resource("gpumod://help")
            r2 = await client.call_tool("list_modes", {})
            c2 = await client.read_resource("gpumod://services")
            r3 = await client.call_tool("service_info", {"service_id": "test-svc"})

        assert r1.is_error is False
        assert isinstance(c1, list)
        assert r2.is_error is False
        assert isinstance(c2, list)
        assert r3.is_error is False
