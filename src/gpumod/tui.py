"""Interactive TUI for gpumod — Textual-based terminal UI.

Provides a live dashboard showing GPU status, services, and an
interactive command input for /status, /switch, /simulate, and /quit.

Security: All displayed names pass through sanitize_name() (SEC-E3).
LLM output rendered as plain text via rich.text.Text() (SEC-T3).
Input validated through SEC-V1 validators before dispatch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Input, Static

from gpumod.validation import sanitize_name

if TYPE_CHECKING:
    from gpumod.cli import AppContext
    from gpumod.models import ServiceInfo, SystemStatus

logger = logging.getLogger(__name__)

# State → display indicator (plain text, no markup)
_STATE_INDICATORS: dict[str, tuple[str, str]] = {
    "running": ("●", "green"),
    "stopped": ("○", "dim"),
    "starting": ("◐", "yellow"),
    "sleeping": ("◑", "blue"),
    "unhealthy": ("●", "red"),
    "failed": ("✖", "red"),
    "unknown": ("?", "dim"),
    "stopping": ("◑", "yellow"),
}


def _vram_bar(used: int, total: int, width: int = 20) -> str:
    """Build an ASCII VRAM usage bar."""
    if total <= 0:
        return "[" + "?" * width + "]"
    ratio = min(used / total, 1.0)
    filled = int(ratio * width)
    empty = width - filled
    pct = int(ratio * 100)
    return f"[{'█' * filled}{'░' * empty}] {pct}%"


def _build_gpu_bar(status: SystemStatus) -> Text:
    """Build a Rich Text for the GPU status bar."""
    if status.gpu is None:
        return Text.from_markup("[yellow]No GPU detected[/yellow]")

    gpu_name = sanitize_name(status.gpu.name)
    total = status.gpu.vram_total_mb
    mode = sanitize_name(status.current_mode) if status.current_mode else "none"

    if status.vram is not None:
        used = status.vram.used_mb
        bar = _vram_bar(used, total)
        return Text(f"▐█▌ {gpu_name} ({total}MB) │ {mode} │ VRAM: {used}/{total}MB {bar}")

    return Text(f"▐█▌ {gpu_name} ({total}MB) │ {mode}")


def _build_service_line(si: ServiceInfo) -> Text:
    """Build a Rich Text for a single service line."""
    name = sanitize_name(si.service.name)
    state_val = si.status.state.value
    symbol, color = _STATE_INDICATORS.get(state_val, ("?", "dim"))
    vram = si.status.vram_mb if si.status.vram_mb is not None else si.service.vram_mb

    line = Text()
    line.append("  ")
    line.append(symbol, style=color)
    line.append(f" {name}  {vram}MB  {state_val}")
    return line


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class GPUBar(Static):
    """Top bar showing GPU name, mode, and VRAM usage."""

    DEFAULT_CSS = """
    GPUBar {
        dock: top;
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    """

    def update_status(self, status: SystemStatus) -> None:
        """Refresh the GPU bar content."""
        self.update(_build_gpu_bar(status))


class ServiceList(Static):
    """Scrollable list of services with state indicators."""

    DEFAULT_CSS = """
    ServiceList {
        height: 1fr;
        padding: 1 2;
    }
    """

    def update_services(self, status: SystemStatus) -> None:
        """Refresh the service list."""
        if not status.services:
            self.update(Text("No services registered.", style="dim"))
            return

        combined = Text()
        for i, si in enumerate(status.services):
            if i > 0:
                combined.append("\n")
            combined.append_text(_build_service_line(si))
        self.update(combined)


class OutputPanel(Static):
    """Panel for displaying command output and simulation results."""

    DEFAULT_CSS = """
    OutputPanel {
        height: auto;
        max-height: 12;
        padding: 0 2;
        color: $text-muted;
    }
    """

    def show_message(self, text: str) -> None:
        """Display a message as plain text (safe for untrusted content)."""
        self.update(Text(sanitize_name(text)))

    def show_markup(self, markup: str) -> None:
        """Display trusted markup text."""
        self.update(Text.from_markup(markup))


class HelpBar(Static):
    """Bottom help bar showing available commands."""

    DEFAULT_CSS = """
    HelpBar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def on_mount(self) -> None:
        """Set initial help text."""
        self.update(Text("?: help │ /status │ /switch <mode> │ /simulate │ q: quit"))


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


class GpumodApp(App[None]):
    """Interactive TUI for GPU service management."""

    TITLE = "gpumod"

    CSS = """
    Screen {
        layout: vertical;
    }
    #command-input {
        dock: bottom;
        margin: 0 1;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("q", "quit", "Quit", show=True, priority=True),
        Binding("question_mark", "help", "Help", show=False),
    ]

    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self._ctx = ctx
        self._status: SystemStatus | None = None

    def compose(self) -> ComposeResult:
        """Build the widget tree."""
        yield GPUBar(id="gpu-bar")
        with Vertical():
            yield ServiceList(id="service-list")
            yield OutputPanel(id="output-panel")
        yield Input(placeholder="> type a command...", id="command-input")
        yield HelpBar(id="help-bar")
        yield Footer()

    async def on_mount(self) -> None:
        """Load initial status on app startup."""
        await self._refresh_status()

    async def _refresh_status(self) -> None:
        """Fetch system status and update all widgets."""
        try:
            self._status = await self._ctx.manager.get_status()
        except Exception:
            logger.exception("Failed to fetch system status")
            self.query_one("#output-panel", OutputPanel).show_message(
                "Error: could not fetch system status"
            )
            return

        self.query_one("#gpu-bar", GPUBar).update_status(self._status)
        self.query_one("#service-list", ServiceList).update_services(self._status)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command input submission."""
        command = event.value.strip()
        event.input.value = ""
        if command:
            await self.handle_command(command)

    async def handle_command(self, command: str) -> None:
        """Parse and dispatch a slash command."""
        output = self.query_one("#output-panel", OutputPanel)

        if command in ("/quit", "/exit"):
            self.exit()
            return

        if command == "/status":
            await self._refresh_status()
            output.show_markup("[green]Status refreshed.[/green]")
            return

        if command in ("/help", "?"):
            output.show_markup("Commands: /status, /switch <mode>, /simulate, /quit")
            return

        if command.startswith("/switch "):
            mode_id = command[8:].strip()
            await self._do_switch(mode_id)
            return

        if command.startswith("/simulate"):
            output.show_message("Simulation not yet implemented in TUI.")
            return

        output.show_message(f"Unknown command: {command}")

    async def _do_switch(self, mode_id: str) -> None:
        """Execute a mode switch command."""
        from gpumod.validation import validate_mode_id

        output = self.query_one("#output-panel", OutputPanel)

        try:
            validate_mode_id(mode_id)
        except ValueError:
            output.show_markup(f"[red]Invalid mode ID:[/red] {sanitize_name(mode_id)}")
            return

        try:
            result = await self._ctx.manager.switch_mode(mode_id)
            if result.success:
                output.show_markup(f"[green]Switched to {sanitize_name(mode_id)}[/green]")
                await self._refresh_status()
            else:
                msg = result.message or "Switch failed"
                output.show_message(f"Switch failed: {sanitize_name(msg)}")
        except Exception:
            logger.exception("Mode switch failed")
            output.show_message("Error: mode switch failed")

    def action_help(self) -> None:
        """Show help text."""
        self.query_one("#output-panel", OutputPanel).show_markup(
            "Commands: /status, /switch <mode>, /simulate, /quit"
        )
