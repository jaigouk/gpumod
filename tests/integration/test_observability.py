"""Integration tests for observability middleware (SEC-A2).

Validates:
- RequestIDMiddleware generates valid UUIDs
- Request ID is reset after tool calls complete
- Different requests get different IDs
- Request ID set on resource reads
- Request ID reset on exception
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpumod.mcp_server import RequestIDMiddleware, request_id_var

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_valid_uuid4(value: str) -> bool:
    """Check if a string is a valid UUID4."""
    try:
        parsed = uuid.UUID(value, version=4)
        return str(parsed) == value
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Request ID Middleware Tests
# ---------------------------------------------------------------------------


class TestRequestIDMiddleware:
    """RequestIDMiddleware generates and manages request IDs (SEC-A2)."""

    async def test_tool_call_generates_uuid(self) -> None:
        """on_call_tool() sets request_id_var to a valid UUID."""
        middleware = RequestIDMiddleware()
        context = MagicMock()

        captured_id: str = ""

        async def capture_call_next(ctx: object) -> str:
            nonlocal captured_id
            captured_id = request_id_var.get()
            return "tool_result"

        result = await middleware.on_call_tool(context, capture_call_next)
        assert result == "tool_result"
        assert _is_valid_uuid4(captured_id)

    async def test_request_id_resets_after_tool_call(self) -> None:
        """Value is reset to empty after tool call completes."""
        middleware = RequestIDMiddleware()
        context = MagicMock()

        call_next = AsyncMock(return_value="result")
        await middleware.on_call_tool(context, call_next)

        # After the call, request_id_var should be back to default ("")
        assert request_id_var.get() == ""

    async def test_different_requests_get_different_ids(self) -> None:
        """Two sequential calls produce different UUIDs."""
        middleware = RequestIDMiddleware()
        context = MagicMock()

        captured_ids: list[str] = []

        async def capture_call_next(ctx: object) -> str:
            captured_ids.append(request_id_var.get())
            return "result"

        await middleware.on_call_tool(context, capture_call_next)
        await middleware.on_call_tool(context, capture_call_next)

        assert len(captured_ids) == 2
        assert captured_ids[0] != captured_ids[1]
        assert _is_valid_uuid4(captured_ids[0])
        assert _is_valid_uuid4(captured_ids[1])

    async def test_resource_read_sets_request_id(self) -> None:
        """on_read_resource() also sets a valid request ID."""
        middleware = RequestIDMiddleware()
        context = MagicMock()

        captured_id: str = ""

        async def capture_call_next(ctx: object) -> str:
            nonlocal captured_id
            captured_id = request_id_var.get()
            return "resource_data"

        result = await middleware.on_read_resource(context, capture_call_next)
        assert result == "resource_data"
        assert _is_valid_uuid4(captured_id)

    async def test_request_id_resets_after_resource_read(self) -> None:
        """Request ID is reset after resource read completes."""
        middleware = RequestIDMiddleware()
        context = MagicMock()

        call_next = AsyncMock(return_value="data")
        await middleware.on_read_resource(context, call_next)

        assert request_id_var.get() == ""

    async def test_request_id_reset_on_tool_exception(self) -> None:
        """Exception during tool call doesn't leave stale request ID."""
        middleware = RequestIDMiddleware()
        context = MagicMock()

        async def failing_call_next(ctx: object) -> str:
            # Verify ID is set during the call
            rid = request_id_var.get()
            assert _is_valid_uuid4(rid)
            raise ValueError("tool error")

        with pytest.raises(ValueError, match="tool error"):
            await middleware.on_call_tool(context, failing_call_next)

        # After exception, ID should be reset
        assert request_id_var.get() == ""

    async def test_request_id_reset_on_resource_exception(self) -> None:
        """Exception during resource read doesn't leave stale request ID."""
        middleware = RequestIDMiddleware()
        context = MagicMock()

        async def failing_call_next(ctx: object) -> str:
            rid = request_id_var.get()
            assert _is_valid_uuid4(rid)
            raise RuntimeError("resource error")

        with pytest.raises(RuntimeError, match="resource error"):
            await middleware.on_read_resource(context, failing_call_next)

        # After exception, ID should be reset
        assert request_id_var.get() == ""
