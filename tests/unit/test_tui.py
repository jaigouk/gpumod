"""Unit tests for the Interactive TUI (P7-T4).

Tests cover:
- App startup and GPU status display
- Service list rendering
- /status, /switch, /simulate, /quit commands
- Security: sanitized names (SEC-E3), ANSI stripping (SEC-T2)
- Graceful exit on /quit

Uses Textual's run_test() pilot for headless testing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.text import Text

from gpumod.models import (
    DriverType,
    GPUInfo,
    Service,
    ServiceInfo,
    ServiceState,
    ServiceStatus,
    SystemStatus,
    VRAMUsage,
)


def _get_widget_text(widget: object) -> str:
    """Extract plain text from a Textual Static widget's internal content."""
    content = getattr(widget, "_content", None)
    if content is None:
        return ""
    if isinstance(content, Text):
        return content.plain
    return str(content)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_system_status(
    *,
    gpu_name: str = "RTX 4090",
    vram_total: int = 24576,
    vram_used: int = 21700,
    current_mode: str | None = "code",
    services: list[ServiceInfo] | None = None,
) -> SystemStatus:
    """Build a SystemStatus for testing."""
    if services is None:
        services = [
            ServiceInfo(
                service=Service(
                    id="vllm-main",
                    name="VLLM Main",
                    driver=DriverType.VLLM,
                    port=8000,
                    vram_mb=19100,
                ),
                status=ServiceStatus(state=ServiceState.RUNNING, health_ok=True),
            ),
            ServiceInfo(
                service=Service(
                    id="embedding-svc",
                    name="Embedding",
                    driver=DriverType.VLLM,
                    port=8001,
                    vram_mb=2500,
                ),
                status=ServiceStatus(state=ServiceState.STOPPED),
            ),
        ]
    return SystemStatus(
        gpu=GPUInfo(name=gpu_name, vram_total_mb=vram_total),
        vram=VRAMUsage(total_mb=vram_total, used_mb=vram_used, free_mb=vram_total - vram_used),
        current_mode=current_mode,
        services=services,
    )


@pytest.fixture
def mock_ctx() -> MagicMock:
    """Create a mock AppContext for TUI testing."""
    ctx = MagicMock()
    ctx.manager = MagicMock()
    ctx.manager.get_status = AsyncMock(return_value=_make_system_status())
    ctx.manager.switch_mode = AsyncMock(
        return_value=MagicMock(success=True, mode_id="inference", started=["svc-1"], stopped=[])
    )
    ctx.db = MagicMock()
    ctx.db.close = AsyncMock()
    ctx.db.list_services = AsyncMock(return_value=[])
    ctx.db.list_modes = AsyncMock(return_value=[])
    ctx.vram = MagicMock()
    ctx.simulation = MagicMock()
    return ctx


# ---------------------------------------------------------------------------
# App startup and GPU status
# ---------------------------------------------------------------------------


class TestTUIAppStartup:
    """TUI app starts and displays GPU status bar."""

    async def test_app_starts_and_shows_gpu_name(self, mock_ctx: MagicMock) -> None:
        """App renders with GPU name visible."""
        from gpumod.tui import GpumodApp

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            text = _get_widget_text(app.query_one("#gpu-bar"))
            assert "RTX 4090" in text

    async def test_app_shows_vram_usage(self, mock_ctx: MagicMock) -> None:
        """App renders VRAM usage in the status bar."""
        from gpumod.tui import GpumodApp

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            text = _get_widget_text(app.query_one("#gpu-bar"))
            assert "24576" in text

    async def test_app_shows_current_mode(self, mock_ctx: MagicMock) -> None:
        """App renders the current mode name."""
        from gpumod.tui import GpumodApp

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            text = _get_widget_text(app.query_one("#gpu-bar"))
            assert "code" in text


# ---------------------------------------------------------------------------
# Service list
# ---------------------------------------------------------------------------


class TestTUIServiceList:
    """TUI service list displays services with state indicators."""

    async def test_service_list_shows_running_service(self, mock_ctx: MagicMock) -> None:
        """Running services appear in the service list."""
        from gpumod.tui import GpumodApp

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            text = _get_widget_text(app.query_one("#service-list"))
            assert "VLLM Main" in text

    async def test_service_list_shows_stopped_service(self, mock_ctx: MagicMock) -> None:
        """Stopped services appear in the service list."""
        from gpumod.tui import GpumodApp

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            text = _get_widget_text(app.query_one("#service-list"))
            assert "Embedding" in text

    async def test_service_list_shows_vram_amounts(self, mock_ctx: MagicMock) -> None:
        """Service VRAM amounts are displayed."""
        from gpumod.tui import GpumodApp

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            text = _get_widget_text(app.query_one("#service-list"))
            assert "19100" in text


# ---------------------------------------------------------------------------
# Security: name sanitization (SEC-E3, SEC-T1, SEC-T2)
# ---------------------------------------------------------------------------


class TestTUISanitization:
    """Service names are sanitized before display (SEC-E3)."""

    async def test_ansi_escapes_stripped(self, mock_ctx: MagicMock) -> None:
        """ANSI escape codes in service names are stripped before display."""
        from gpumod.tui import GpumodApp

        malicious_status = _make_system_status(
            services=[
                ServiceInfo(
                    service=Service(
                        id="bad-svc",
                        name="\x1b[31mEvil\x1b[0m Service",
                        driver=DriverType.VLLM,
                        port=8000,
                        vram_mb=4000,
                    ),
                    status=ServiceStatus(state=ServiceState.RUNNING),
                ),
            ],
        )
        mock_ctx.manager.get_status = AsyncMock(return_value=malicious_status)

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            text = _get_widget_text(app.query_one("#service-list"))
            assert "\x1b[31m" not in text
            assert "Evil" in text

    async def test_rich_markup_not_interpreted(self, mock_ctx: MagicMock) -> None:
        """Rich markup tags in service names display as plain text or stripped."""
        from gpumod.tui import GpumodApp

        markup_status = _make_system_status(
            services=[
                ServiceInfo(
                    service=Service(
                        id="markup-svc",
                        name="[bold red]Dangerous[/bold red]",
                        driver=DriverType.VLLM,
                        port=8000,
                        vram_mb=4000,
                    ),
                    status=ServiceStatus(state=ServiceState.RUNNING),
                ),
            ],
        )
        mock_ctx.manager.get_status = AsyncMock(return_value=markup_status)

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            text = _get_widget_text(app.query_one("#service-list"))
            assert "Dangerous" in text
            assert "[bold red]" not in text


# ---------------------------------------------------------------------------
# Commands: /quit
# ---------------------------------------------------------------------------


class TestTUIQuitCommand:
    """TUI exits cleanly on /quit command."""

    async def test_quit_command_exits(self, mock_ctx: MagicMock) -> None:
        """Sending /quit via handle_command exits the app."""
        from gpumod.tui import GpumodApp

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.handle_command("/quit")


# ---------------------------------------------------------------------------
# Commands: /status refresh
# ---------------------------------------------------------------------------


class TestTUIStatusCommand:
    """TUI /status command refreshes the display."""

    async def test_status_command_calls_get_status(self, mock_ctx: MagicMock) -> None:
        """Submitting /status triggers a refresh via get_status()."""
        from gpumod.tui import GpumodApp

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            initial_calls = mock_ctx.manager.get_status.call_count
            await app.handle_command("/status")
            assert mock_ctx.manager.get_status.call_count > initial_calls


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestTUICLICommand:
    """TUI is accessible via gpumod tui CLI command."""

    def test_tui_command_registered(self) -> None:
        """The 'tui' command is registered on the main CLI app."""
        from gpumod.cli import app

        callback_names = [
            cmd.callback.__name__ for cmd in app.registered_commands if cmd.callback is not None
        ]
        assert "tui" in callback_names


# ---------------------------------------------------------------------------
# Footer / help bar
# ---------------------------------------------------------------------------


class TestTUIFooter:
    """TUI footer shows available commands."""

    async def test_footer_visible(self, mock_ctx: MagicMock) -> None:
        """Footer bar is rendered with help text."""
        from gpumod.tui import GpumodApp

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            text = _get_widget_text(app.query_one("#help-bar"))
            assert "/status" in text or "quit" in text


# ---------------------------------------------------------------------------
# Commands: /switch
# ---------------------------------------------------------------------------


class TestTUISwitchCommand:
    """TUI /switch command dispatches mode switch."""

    async def test_switch_command_calls_switch_mode(self, mock_ctx: MagicMock) -> None:
        """Submitting /switch <mode> triggers manager.switch_mode()."""
        from gpumod.tui import GpumodApp

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.handle_command("/switch inference")
            mock_ctx.manager.switch_mode.assert_called_once_with("inference")

    async def test_switch_invalid_mode_shows_error(self, mock_ctx: MagicMock) -> None:
        """/switch with invalid mode ID shows error without calling switch_mode."""
        from gpumod.tui import GpumodApp

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.handle_command("/switch ; evil")
            mock_ctx.manager.switch_mode.assert_not_called()
            text = _get_widget_text(app.query_one("#output-panel"))
            assert "Invalid" in text

    async def test_switch_failed_shows_message(self, mock_ctx: MagicMock) -> None:
        """/switch that fails shows failure message."""
        from gpumod.tui import GpumodApp

        mock_ctx.manager.switch_mode = AsyncMock(
            return_value=MagicMock(success=False, message="Mode not found")
        )
        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.handle_command("/switch badmode")
            text = _get_widget_text(app.query_one("#output-panel"))
            assert "failed" in text.lower() or "not found" in text.lower()

    async def test_switch_exception_shows_error(self, mock_ctx: MagicMock) -> None:
        """/switch that raises an exception shows error."""
        from gpumod.tui import GpumodApp

        mock_ctx.manager.switch_mode = AsyncMock(side_effect=RuntimeError("boom"))
        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.handle_command("/switch inference")
            text = _get_widget_text(app.query_one("#output-panel"))
            assert "error" in text.lower()


# ---------------------------------------------------------------------------
# Commands: /simulate, /help, unknown
# ---------------------------------------------------------------------------


class TestTUIOtherCommands:
    """TUI handles /simulate, /help, and unknown commands."""

    async def test_simulate_command_shows_placeholder(self, mock_ctx: MagicMock) -> None:
        """/simulate shows placeholder message."""
        from gpumod.tui import GpumodApp

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.handle_command("/simulate")
            text = _get_widget_text(app.query_one("#output-panel"))
            assert "not yet implemented" in text.lower() or "simulate" in text.lower()

    async def test_help_command_shows_commands(self, mock_ctx: MagicMock) -> None:
        """/help shows available commands."""
        from gpumod.tui import GpumodApp

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.handle_command("/help")
            text = _get_widget_text(app.query_one("#output-panel"))
            assert "/status" in text
            assert "/quit" in text

    async def test_unknown_command_shows_message(self, mock_ctx: MagicMock) -> None:
        """Unknown command shows error message."""
        from gpumod.tui import GpumodApp

        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.handle_command("/foobar")
            text = _get_widget_text(app.query_one("#output-panel"))
            assert "Unknown" in text or "foobar" in text


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestTUIHelpers:
    """Test TUI helper functions directly."""

    def test_vram_bar_normal(self) -> None:
        """VRAM bar renders percentage correctly."""
        from gpumod.tui import _vram_bar

        bar = _vram_bar(12288, 24576)
        assert "50%" in bar

    def test_vram_bar_zero_total(self) -> None:
        """VRAM bar handles zero total gracefully."""
        from gpumod.tui import _vram_bar

        bar = _vram_bar(0, 0)
        assert "?" in bar

    def test_vram_bar_over_100(self) -> None:
        """VRAM bar caps at 100%."""
        from gpumod.tui import _vram_bar

        bar = _vram_bar(30000, 24576)
        assert "100%" in bar

    def test_build_gpu_bar_no_gpu(self) -> None:
        """GPU bar shows warning when no GPU."""
        from gpumod.tui import _build_gpu_bar

        status = SystemStatus(gpu=None)
        text = _build_gpu_bar(status)
        assert "No GPU" in text.plain

    def test_build_service_line_includes_name_and_vram(self) -> None:
        """Service line includes sanitized name and VRAM."""
        from gpumod.tui import _build_service_line

        si = ServiceInfo(
            service=Service(
                id="test-svc",
                name="Test Service",
                driver=DriverType.VLLM,
                port=8000,
                vram_mb=4000,
            ),
            status=ServiceStatus(state=ServiceState.RUNNING),
        )
        line = _build_service_line(si)
        assert "Test Service" in line.plain
        assert "4000" in line.plain

    def test_build_gpu_bar_no_vram_info(self) -> None:
        """GPU bar renders without VRAM info."""
        from gpumod.tui import _build_gpu_bar

        status = SystemStatus(
            gpu=GPUInfo(name="RTX 4090", vram_total_mb=24576),
            vram=None,
        )
        text = _build_gpu_bar(status)
        assert "RTX 4090" in text.plain
        assert "24576" in text.plain


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestTUIErrorHandling:
    """TUI handles errors gracefully."""

    async def test_status_fetch_error_shows_message(self, mock_ctx: MagicMock) -> None:
        """Error fetching status shows error in output panel."""
        from gpumod.tui import GpumodApp

        mock_ctx.manager.get_status = AsyncMock(side_effect=RuntimeError("GPU unavailable"))
        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            text = _get_widget_text(app.query_one("#output-panel"))
            assert "error" in text.lower() or "could not" in text.lower()

    async def test_no_services_shows_message(self, mock_ctx: MagicMock) -> None:
        """Empty service list shows appropriate message."""
        from gpumod.tui import GpumodApp

        empty_status = _make_system_status(services=[])
        mock_ctx.manager.get_status = AsyncMock(return_value=empty_status)
        app = GpumodApp(ctx=mock_ctx)
        async with app.run_test() as pilot:
            await pilot.pause()
            text = _get_widget_text(app.query_one("#service-list"))
            assert "No services" in text
