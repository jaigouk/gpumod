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
