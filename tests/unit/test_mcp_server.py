"""Tests for MCP server foundation — server creation, lifecycle, validation, error sanitization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Return a temporary database path."""
    return tmp_path / "test.db"


# ---------------------------------------------------------------------------
# TestMCPServerCreation
# ---------------------------------------------------------------------------


class TestMCPServerCreation:
    """Tests for create_mcp_server factory."""

    def test_create_server_returns_fastmcp_instance(self) -> None:
        from fastmcp import FastMCP

        from gpumod.mcp_server import create_mcp_server

        server = create_mcp_server()
        assert isinstance(server, FastMCP)

    def test_server_name_is_gpumod(self) -> None:
        from gpumod.mcp_server import create_mcp_server

        server = create_mcp_server()
        assert server.name == "gpumod"

    def test_server_has_instructions(self) -> None:
        from gpumod.mcp_server import create_mcp_server

        server = create_mcp_server()
        assert server.instructions is not None
        assert len(server.instructions) > 0
        assert "GPU" in server.instructions

    def test_server_has_strict_input_validation(self) -> None:
        from gpumod.mcp_server import create_mcp_server

        server = create_mcp_server()
        # SEC-V2: strict_input_validation must be enabled
        assert server.strict_input_validation is True


# ---------------------------------------------------------------------------
# TestMCPLifecycle
# ---------------------------------------------------------------------------


class TestMCPLifecycle:
    """Tests for MCP server lifespan — DB connect on startup, close on shutdown."""

    async def test_lifecycle_creates_app_context(self, tmp_db_path: Path) -> None:
        from gpumod.mcp_server import create_mcp_server, gpumod_lifespan

        server = create_mcp_server()

        async with gpumod_lifespan(server, db_path=tmp_db_path) as ctx:
            # The lifespan should yield an AppContext-like dict with a 'db' key
            assert ctx is not None
            assert "db" in ctx
            assert "manager" in ctx

    async def test_lifecycle_closes_db_on_shutdown(self, tmp_db_path: Path) -> None:
        from gpumod.mcp_server import create_mcp_server, gpumod_lifespan

        server = create_mcp_server()

        async with gpumod_lifespan(server, db_path=tmp_db_path) as ctx:
            db = ctx["db"]
            # DB should be connected inside the context
            assert db._conn is not None

        # After exiting the context, DB should be closed
        assert db._conn is None

    async def test_lifecycle_context_available_to_tools(self, tmp_db_path: Path) -> None:
        from gpumod.mcp_server import create_mcp_server, gpumod_lifespan

        server = create_mcp_server()

        async with gpumod_lifespan(server, db_path=tmp_db_path) as ctx:
            # Verify the context contains all required service dependencies
            assert "db" in ctx
            assert "registry" in ctx
            assert "lifecycle" in ctx
            assert "vram" in ctx
            assert "sleep" in ctx
            assert "manager" in ctx
            assert "model_registry" in ctx
            assert "template_engine" in ctx
            assert "preset_loader" in ctx


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input validation via validation.py regexes."""

    def test_rejects_service_id_with_shell_chars(self) -> None:
        from gpumod.validation import validate_service_id

        with pytest.raises(ValueError, match="Invalid service_id"):
            validate_service_id("; rm -rf /")

    def test_rejects_service_id_with_sql_injection(self) -> None:
        from gpumod.validation import validate_service_id

        with pytest.raises(ValueError, match="Invalid service_id"):
            validate_service_id("'; DROP TABLE services--")

    def test_rejects_service_id_with_template_injection(self) -> None:
        from gpumod.validation import validate_service_id

        with pytest.raises(ValueError, match="Invalid service_id"):
            validate_service_id("{{7*7}}")

    def test_accepts_valid_service_id(self) -> None:
        from gpumod.validation import validate_service_id

        result = validate_service_id("vllm-chat-01")
        assert result == "vllm-chat-01"

    def test_rejects_empty_service_id(self) -> None:
        from gpumod.validation import validate_service_id

        with pytest.raises(ValueError, match="Invalid service_id"):
            validate_service_id("")

    def test_rejects_service_id_over_max_length(self) -> None:
        from gpumod.validation import validate_service_id

        with pytest.raises(ValueError, match="Invalid service_id"):
            validate_service_id("a" * 65)

    def test_accepts_valid_mode_id(self) -> None:
        from gpumod.validation import validate_mode_id

        result = validate_mode_id("chat-mode")
        assert result == "chat-mode"

    def test_rejects_invalid_mode_id(self) -> None:
        from gpumod.validation import validate_mode_id

        with pytest.raises(ValueError, match="Invalid mode_id"):
            validate_mode_id("../etc")

    def test_accepts_valid_model_id(self) -> None:
        from gpumod.validation import validate_model_id

        result = validate_model_id("meta-llama/Llama-3.1-8B")
        assert result == "meta-llama/Llama-3.1-8B"

    def test_rejects_invalid_model_id(self) -> None:
        from gpumod.validation import validate_model_id

        with pytest.raises(ValueError, match="Invalid model_id"):
            validate_model_id("../../passwd")

    def test_validates_context_override(self) -> None:
        from gpumod.validation import validate_context_override

        key, value = validate_context_override("vllm-svc", 4096)
        assert key == "vllm-svc"
        assert value == 4096

    def test_rejects_context_override_exceeding_max(self) -> None:
        from gpumod.validation import validate_context_override

        with pytest.raises(ValueError, match="Invalid context override"):
            validate_context_override("vllm-svc", 200000)

    def test_rejects_context_override_below_min(self) -> None:
        from gpumod.validation import validate_context_override

        with pytest.raises(ValueError, match="Invalid context override"):
            validate_context_override("vllm-svc", 0)

    def test_rejects_context_override_with_invalid_key(self) -> None:
        from gpumod.validation import validate_context_override

        with pytest.raises(ValueError, match="Invalid service_id"):
            validate_context_override("; rm -rf /", 4096)


# ---------------------------------------------------------------------------
# TestErrorSanitization
# ---------------------------------------------------------------------------


class TestErrorSanitization:
    """Tests for error sanitization — strip internal paths and tracebacks."""

    def test_strips_db_path_from_error(self) -> None:
        from gpumod.mcp_server import sanitize_error_message

        raw = "Database error: /home/user/.config/gpumod/gpumod.db: table not found"
        sanitized = sanitize_error_message(raw)
        assert "/home/" not in sanitized
        assert ".db" not in sanitized

    def test_strips_tmp_path_from_error(self) -> None:
        from gpumod.mcp_server import sanitize_error_message

        raw = "Failed to read /tmp/gpumod_cache/model.bin"
        sanitized = sanitize_error_message(raw)
        assert "/tmp/" not in sanitized

    def test_strips_traceback_details(self) -> None:
        from gpumod.mcp_server import sanitize_error_message

        raw = (
            "Traceback (most recent call last):\n"
            '  File "/home/user/gpumod/src/gpumod/db.py", line 42\n'
            "    raise RuntimeError('connection failed')"
        )
        sanitized = sanitize_error_message(raw)
        assert "Traceback" not in sanitized
        assert "/home/" not in sanitized
        assert "line 42" not in sanitized

    def test_preserves_user_facing_message(self) -> None:
        from gpumod.mcp_server import sanitize_error_message

        raw = "Mode not found: chat-mode"
        sanitized = sanitize_error_message(raw)
        assert sanitized == "Mode not found: chat-mode"

    def test_strips_var_path(self) -> None:
        from gpumod.mcp_server import sanitize_error_message

        raw = "Cannot access /var/log/gpumod/error.log"
        sanitized = sanitize_error_message(raw)
        assert "/var/" not in sanitized

    def test_strips_python_file_references(self) -> None:
        from gpumod.mcp_server import sanitize_error_message

        raw = "Error in /usr/lib/python3.11/asyncio/base_events.py:123"
        sanitized = sanitize_error_message(raw)
        assert ".py" not in sanitized


# ---------------------------------------------------------------------------
# TestErrorSanitizationMiddleware
# ---------------------------------------------------------------------------


class TestErrorSanitizationMiddleware:
    """Tests for the error sanitization middleware integration."""

    def test_middleware_is_attached_to_server(self) -> None:
        from gpumod.mcp_server import ErrorSanitizationMiddleware, create_mcp_server

        server = create_mcp_server()
        # Verify at least one middleware is our ErrorSanitizationMiddleware
        has_error_middleware = any(
            isinstance(m, ErrorSanitizationMiddleware) for m in server.middleware
        )
        assert has_error_middleware

    def test_middleware_sanitizes_tool_errors(self) -> None:
        from gpumod.mcp_server import sanitize_error_message

        # The middleware uses sanitize_error_message internally
        raw = "Failed: /home/user/.config/gpumod/gpumod.db locked"
        sanitized = sanitize_error_message(raw)
        assert "/home/" not in sanitized

    async def test_middleware_on_call_tool_sanitizes_exception(self) -> None:
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import ErrorSanitizationMiddleware

        middleware = ErrorSanitizationMiddleware()

        # Simulate a call_next that raises with an internal path
        async def failing_call_next(ctx: object) -> None:
            msg = "DB error: /home/user/.config/gpumod/gpumod.db locked"
            raise RuntimeError(msg)

        mock_context = AsyncMock()
        with pytest.raises(RuntimeError, match=r"^(?!.*/home/)"):
            await middleware.on_call_tool(mock_context, failing_call_next)

    async def test_middleware_on_call_tool_passes_through_on_success(self) -> None:
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import ErrorSanitizationMiddleware

        middleware = ErrorSanitizationMiddleware()

        async def ok_call_next(ctx: object) -> str:
            return "success"

        mock_context = AsyncMock()
        result = await middleware.on_call_tool(mock_context, ok_call_next)
        assert result == "success"

    async def test_middleware_on_read_resource_sanitizes_exception(self) -> None:
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import ErrorSanitizationMiddleware

        middleware = ErrorSanitizationMiddleware()

        async def failing_call_next(ctx: object) -> None:
            msg = "Cannot read /tmp/cache/model.bin"
            raise FileNotFoundError(msg)

        mock_context = AsyncMock()
        with pytest.raises(FileNotFoundError) as exc_info:
            await middleware.on_read_resource(mock_context, failing_call_next)
        assert "/tmp/" not in str(exc_info.value)

    async def test_middleware_on_read_resource_passes_through_on_success(self) -> None:
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import ErrorSanitizationMiddleware

        middleware = ErrorSanitizationMiddleware()

        async def ok_call_next(ctx: object) -> str:
            return "resource content"

        mock_context = AsyncMock()
        result = await middleware.on_read_resource(mock_context, ok_call_next)
        assert result == "resource content"

    def test_sanitize_message_only_path_returns_generic(self) -> None:
        """When the entire message is just a path, return a generic error."""
        from gpumod.mcp_server import sanitize_error_message

        raw = "/home/user/.config/gpumod/gpumod.db"
        sanitized = sanitize_error_message(raw)
        assert sanitized == "Internal error: please check server logs"


# ---------------------------------------------------------------------------
# TestMCPMainEntryPoint
# ---------------------------------------------------------------------------


class TestMCPMainEntryPoint:
    """Tests for mcp_main.py entry point."""

    def test_mcp_main_module_importable(self) -> None:
        import gpumod.mcp_main  # noqa: F401

    def test_mcp_main_has_server(self) -> None:
        from fastmcp import FastMCP

        from gpumod.mcp_main import server

        assert isinstance(server, FastMCP)


# ---------------------------------------------------------------------------
# TestRequestIDMiddleware (SEC-A2)
# ---------------------------------------------------------------------------


class TestRequestIDMiddleware:
    """Tests for request ID middleware (SEC-A2)."""

    def test_request_id_middleware_is_attached_to_server(self) -> None:
        from gpumod.mcp_server import RequestIDMiddleware, create_mcp_server

        server = create_mcp_server()
        has_rid_middleware = any(isinstance(m, RequestIDMiddleware) for m in server.middleware)
        assert has_rid_middleware

    async def test_request_id_set_in_contextvar_on_tool_call(self) -> None:
        """RequestIDMiddleware sets a UUID in request_id_var during on_call_tool."""
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RequestIDMiddleware, request_id_var

        middleware = RequestIDMiddleware()
        captured_id: str = ""

        async def capturing_call_next(ctx: object) -> str:
            nonlocal captured_id
            captured_id = request_id_var.get()
            return "ok"

        mock_context = AsyncMock()
        await middleware.on_call_tool(mock_context, capturing_call_next)
        assert captured_id != ""

    async def test_request_id_is_valid_uuid(self) -> None:
        """The request ID set by the middleware is a valid UUID4."""
        import uuid
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RequestIDMiddleware, request_id_var

        middleware = RequestIDMiddleware()
        captured_id: str = ""

        async def capturing_call_next(ctx: object) -> str:
            nonlocal captured_id
            captured_id = request_id_var.get()
            return "ok"

        mock_context = AsyncMock()
        await middleware.on_call_tool(mock_context, capturing_call_next)

        # Should parse as a valid UUID
        parsed = uuid.UUID(captured_id)
        assert str(parsed) == captured_id

    async def test_request_id_propagates_through_tool_call(self) -> None:
        """The request ID is available in the downstream call_next handler."""
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RequestIDMiddleware, request_id_var

        middleware = RequestIDMiddleware()
        ids_seen: list[str] = []

        async def capturing_call_next(ctx: object) -> str:
            ids_seen.append(request_id_var.get())
            return "ok"

        mock_context = AsyncMock()
        await middleware.on_call_tool(mock_context, capturing_call_next)

        assert len(ids_seen) == 1
        assert ids_seen[0] != ""

    async def test_different_requests_get_different_ids(self) -> None:
        """Each tool call gets a unique request ID."""
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RequestIDMiddleware, request_id_var

        middleware = RequestIDMiddleware()
        ids_seen: list[str] = []

        async def capturing_call_next(ctx: object) -> str:
            ids_seen.append(request_id_var.get())
            return "ok"

        mock_context = AsyncMock()
        await middleware.on_call_tool(mock_context, capturing_call_next)
        await middleware.on_call_tool(mock_context, capturing_call_next)
        await middleware.on_call_tool(mock_context, capturing_call_next)

        assert len(ids_seen) == 3
        assert len(set(ids_seen)) == 3  # All unique

    async def test_request_id_set_on_resource_read(self) -> None:
        """RequestIDMiddleware sets a UUID in request_id_var during on_read_resource."""
        import uuid
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RequestIDMiddleware, request_id_var

        middleware = RequestIDMiddleware()
        captured_id: str = ""

        async def capturing_call_next(ctx: object) -> str:
            nonlocal captured_id
            captured_id = request_id_var.get()
            return "data"

        mock_context = AsyncMock()
        await middleware.on_read_resource(mock_context, capturing_call_next)

        assert captured_id != ""
        # Should also be a valid UUID
        parsed = uuid.UUID(captured_id)
        assert str(parsed) == captured_id

    async def test_request_id_reset_after_tool_call(self) -> None:
        """After the tool call completes, the context var is reset."""
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RequestIDMiddleware, request_id_var

        middleware = RequestIDMiddleware()

        async def ok_call_next(ctx: object) -> str:
            return "ok"

        mock_context = AsyncMock()

        # Set a known value before the middleware call
        token = request_id_var.set("before")
        try:
            await middleware.on_call_tool(mock_context, ok_call_next)
            # After the middleware call, the var should be reset to "before"
            assert request_id_var.get() == "before"
        finally:
            request_id_var.reset(token)

    async def test_request_id_reset_after_resource_read(self) -> None:
        """After the resource read completes, the context var is reset."""
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RequestIDMiddleware, request_id_var

        middleware = RequestIDMiddleware()

        async def ok_call_next(ctx: object) -> str:
            return "data"

        mock_context = AsyncMock()

        token = request_id_var.set("before")
        try:
            await middleware.on_read_resource(mock_context, ok_call_next)
            assert request_id_var.get() == "before"
        finally:
            request_id_var.reset(token)

    async def test_request_id_reset_on_exception(self) -> None:
        """The context var is reset even if the downstream call raises."""
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RequestIDMiddleware, request_id_var

        middleware = RequestIDMiddleware()

        async def failing_call_next(ctx: object) -> str:
            msg = "boom"
            raise RuntimeError(msg)

        mock_context = AsyncMock()

        token = request_id_var.set("before")
        try:
            with pytest.raises(RuntimeError, match="boom"):
                await middleware.on_call_tool(mock_context, failing_call_next)
            assert request_id_var.get() == "before"
        finally:
            request_id_var.reset(token)


# ---------------------------------------------------------------------------
# TestRateLimitMiddleware (SEC-R2)
# ---------------------------------------------------------------------------


class TestRateLimitMiddleware:
    """Tests for rate limiting middleware (SEC-R2)."""

    async def test_allows_requests_under_limit(self) -> None:
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RateLimitMiddleware

        middleware = RateLimitMiddleware(max_requests=10, window_seconds=1.0)

        async def ok_call_next(ctx: object) -> str:
            return "success"

        mock_context = AsyncMock()

        # 10 requests should all succeed
        for _ in range(10):
            result = await middleware.on_call_tool(mock_context, ok_call_next)
            assert result == "success"

    async def test_rejects_over_limit(self) -> None:
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RateLimitMiddleware

        middleware = RateLimitMiddleware(max_requests=3, window_seconds=1.0)

        async def ok_call_next(ctx: object) -> str:
            return "success"

        mock_context = AsyncMock()

        # 3 requests should succeed
        for _ in range(3):
            await middleware.on_call_tool(mock_context, ok_call_next)

        # 4th request should be rejected
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            await middleware.on_call_tool(mock_context, ok_call_next)

    async def test_configurable_rate_limit(self) -> None:
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RateLimitMiddleware

        middleware = RateLimitMiddleware(max_requests=2, window_seconds=1.0)

        async def ok_call_next(ctx: object) -> str:
            return "success"

        mock_context = AsyncMock()

        # 2 requests should succeed
        await middleware.on_call_tool(mock_context, ok_call_next)
        await middleware.on_call_tool(mock_context, ok_call_next)

        # 3rd should fail
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            await middleware.on_call_tool(mock_context, ok_call_next)

    async def test_window_resets_after_expiry(self) -> None:
        import time
        from unittest.mock import AsyncMock, patch

        from gpumod.mcp_server import RateLimitMiddleware

        middleware = RateLimitMiddleware(max_requests=2, window_seconds=1.0)

        async def ok_call_next(ctx: object) -> str:
            return "success"

        mock_context = AsyncMock()

        # Use 2 requests
        now = time.monotonic()
        with patch("gpumod.mcp_server.time") as mock_time:
            mock_time.monotonic.return_value = now
            await middleware.on_call_tool(mock_context, ok_call_next)
            await middleware.on_call_tool(mock_context, ok_call_next)

            # Advance time past the window
            mock_time.monotonic.return_value = now + 1.1
            # Should succeed again after window reset
            result = await middleware.on_call_tool(mock_context, ok_call_next)
            assert result == "success"

    def test_rate_limit_middleware_is_attached_to_server(self) -> None:
        from gpumod.mcp_server import RateLimitMiddleware, create_mcp_server

        server = create_mcp_server()
        has_rate_middleware = any(isinstance(m, RateLimitMiddleware) for m in server.middleware)
        assert has_rate_middleware

    async def test_default_limit_is_10(self) -> None:
        from gpumod.mcp_server import RateLimitMiddleware

        middleware = RateLimitMiddleware()
        assert middleware._max_requests == 10
        assert middleware._window_seconds == 1.0
        assert middleware._client_requests == {}


# ---------------------------------------------------------------------------
# TestPerClientRateLimit (SEC-R3)
# ---------------------------------------------------------------------------


class TestPerClientRateLimit:
    """Tests for per-client rate limiting (SEC-R3)."""

    async def test_two_clients_get_independent_quotas(self) -> None:
        """Two distinct clients each get their full max_requests quota."""
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RateLimitMiddleware

        middleware = RateLimitMiddleware(max_requests=3, window_seconds=1.0)

        async def ok_call_next(ctx: object) -> str:
            return "success"

        # Client A context
        client_a_context = AsyncMock()
        client_a_context.client_id = "client-a"

        # Client B context
        client_b_context = AsyncMock()
        client_b_context.client_id = "client-b"

        # Both clients should be able to make 3 requests each
        for _ in range(3):
            result = await middleware.on_call_tool(client_a_context, ok_call_next)
            assert result == "success"

        for _ in range(3):
            result = await middleware.on_call_tool(client_b_context, ok_call_next)
            assert result == "success"

    async def test_exceeding_one_client_does_not_affect_another(self) -> None:
        """Hitting rate limit for one client leaves others unaffected."""
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RateLimitMiddleware

        middleware = RateLimitMiddleware(max_requests=2, window_seconds=1.0)

        async def ok_call_next(ctx: object) -> str:
            return "success"

        client_a_context = AsyncMock()
        client_a_context.client_id = "client-a"

        client_b_context = AsyncMock()
        client_b_context.client_id = "client-b"

        # Exhaust client A's quota
        await middleware.on_call_tool(client_a_context, ok_call_next)
        await middleware.on_call_tool(client_a_context, ok_call_next)

        # Client A should be rate limited
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            await middleware.on_call_tool(client_a_context, ok_call_next)

        # Client B should still work
        result = await middleware.on_call_tool(client_b_context, ok_call_next)
        assert result == "success"

    async def test_on_read_resource_enforces_rate_limit(self) -> None:
        """on_read_resource should enforce rate limits just like on_call_tool."""
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RateLimitMiddleware

        middleware = RateLimitMiddleware(max_requests=2, window_seconds=1.0)

        async def ok_call_next(ctx: object) -> str:
            return "resource content"

        mock_context = AsyncMock()

        # 2 resource reads should succeed
        result = await middleware.on_read_resource(mock_context, ok_call_next)
        assert result == "resource content"
        result = await middleware.on_read_resource(mock_context, ok_call_next)
        assert result == "resource content"

        # 3rd should be rate limited
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            await middleware.on_read_resource(mock_context, ok_call_next)

    async def test_on_read_resource_shares_quota_with_tool_calls(self) -> None:
        """Tool calls and resource reads share the same per-client quota."""
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RateLimitMiddleware

        middleware = RateLimitMiddleware(max_requests=3, window_seconds=1.0)

        async def ok_call_next(ctx: object) -> str:
            return "ok"

        mock_context = AsyncMock()

        # Mix tool calls and resource reads
        await middleware.on_call_tool(mock_context, ok_call_next)
        await middleware.on_read_resource(mock_context, ok_call_next)
        await middleware.on_call_tool(mock_context, ok_call_next)

        # 4th request (either type) should be rate limited
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            await middleware.on_read_resource(mock_context, ok_call_next)

    async def test_per_client_window_expiration(self) -> None:
        """After the window expires, a client's quota resets independently."""
        import time
        from unittest.mock import AsyncMock, patch

        from gpumod.mcp_server import RateLimitMiddleware

        middleware = RateLimitMiddleware(max_requests=2, window_seconds=1.0)

        async def ok_call_next(ctx: object) -> str:
            return "success"

        client_a_context = AsyncMock()
        client_a_context.client_id = "client-a"

        client_b_context = AsyncMock()
        client_b_context.client_id = "client-b"

        now = time.monotonic()
        with patch("gpumod.mcp_server.time") as mock_time:
            mock_time.monotonic.return_value = now

            # Exhaust client A's quota
            await middleware.on_call_tool(client_a_context, ok_call_next)
            await middleware.on_call_tool(client_a_context, ok_call_next)

            # Also use one of client B's
            await middleware.on_call_tool(client_b_context, ok_call_next)

            # Client A should be rate limited
            with pytest.raises(RuntimeError, match="Rate limit exceeded"):
                await middleware.on_call_tool(client_a_context, ok_call_next)

            # Advance time past the window
            mock_time.monotonic.return_value = now + 1.1

            # Client A should be able to make requests again
            result = await middleware.on_call_tool(client_a_context, ok_call_next)
            assert result == "success"

            # Client B should also work (window expired for it too)
            result = await middleware.on_call_tool(client_b_context, ok_call_next)
            assert result == "success"

    async def test_default_client_id_when_no_attribute(self) -> None:
        """When context has no client_id attribute, uses '__default__'."""
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RateLimitMiddleware

        middleware = RateLimitMiddleware(max_requests=2, window_seconds=1.0)

        async def ok_call_next(ctx: object) -> str:
            return "success"

        # AsyncMock without client_id attribute
        mock_context = AsyncMock(spec=[])

        await middleware.on_call_tool(mock_context, ok_call_next)
        assert "__default__" in middleware._client_requests

    async def test_get_client_id_extracts_from_context(self) -> None:
        """_get_client_id should extract client_id from context."""
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RateLimitMiddleware

        middleware = RateLimitMiddleware()

        # Context with string client_id
        context_with_id = AsyncMock()
        context_with_id.client_id = "test-client-42"
        assert middleware._get_client_id(context_with_id) == "test-client-42"

        # Context with int client_id (should be converted to string)
        context_with_int_id = AsyncMock()
        context_with_int_id.client_id = 123
        assert middleware._get_client_id(context_with_int_id) == "123"

        # Context without client_id
        context_without_id = AsyncMock(spec=[])
        assert middleware._get_client_id(context_without_id) == "__default__"

    async def test_on_read_resource_with_per_client_tracking(self) -> None:
        """Resource reads are tracked per client."""
        from unittest.mock import AsyncMock

        from gpumod.mcp_server import RateLimitMiddleware

        middleware = RateLimitMiddleware(max_requests=2, window_seconds=1.0)

        async def ok_call_next(ctx: object) -> str:
            return "data"

        client_a_context = AsyncMock()
        client_a_context.client_id = "reader-a"

        client_b_context = AsyncMock()
        client_b_context.client_id = "reader-b"

        # Exhaust client A's quota via resource reads
        await middleware.on_read_resource(client_a_context, ok_call_next)
        await middleware.on_read_resource(client_a_context, ok_call_next)

        # Client A should be rate limited on resource reads
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            await middleware.on_read_resource(client_a_context, ok_call_next)

        # Client B can still read resources
        result = await middleware.on_read_resource(client_b_context, ok_call_next)
        assert result == "data"
