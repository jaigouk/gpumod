"""VRAM ASCII visualization module.

Provides Rich-based visualization of GPU VRAM usage with colored bar charts,
status panels, and comparison views for service management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.text import Text

from gpumod.validation import sanitize_name

if TYPE_CHECKING:
    from gpumod.models import SystemStatus

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FILLED_CHAR = "\u2588"  # █
_FREE_CHAR = "\u2591"  # ░

# Map service states to Rich color names
_STATE_COLORS: dict[str, str] = {
    "running": "green",
    "starting": "green",
    "sleeping": "yellow",
    "stopping": "yellow",
    "unhealthy": "red",
    "failed": "red",
    "stopped": "dim",
    "unknown": "dim",
}


def _mb_to_gb_label(mb: int) -> str:
    """Convert megabytes to a compact GB label.

    Parameters
    ----------
    mb:
        Value in megabytes.

    Returns
    -------
    str
        Compact GB string like "24GB" or "6.5GB".
    """
    gb = mb / 1024
    if gb == int(gb):
        return f"{int(gb)}GB"
    return f"{gb:.1f}GB"


# ---------------------------------------------------------------------------
# VRAMBar — renders a single VRAM usage bar
# ---------------------------------------------------------------------------


class VRAMBar:
    """Renders a single VRAM usage bar with service blocks and scale.

    Responsible only for bar rendering — does not compose panels or headers.
    """

    def state_color(self, state: str) -> str:
        """Return the Rich color name for a given service state.

        Parameters
        ----------
        state:
            A service state string (e.g. "running", "sleeping").

        Returns
        -------
        str
            A Rich color name.
        """
        return _STATE_COLORS.get(state, "dim")

    def render(
        self,
        gpu_total_mb: int,
        services: list[tuple[str, int, str]],
        width: int = 50,
    ) -> str:
        """Render a VRAM usage bar as a multi-line string.

        Parameters
        ----------
        gpu_total_mb:
            Total GPU VRAM in megabytes.
        services:
            List of (name, vram_mb, state) tuples for each service.
        width:
            Number of block characters in the bar (default 50).

        Returns
        -------
        str
            Multi-line string with bar, labels, and scale.
        """
        total_used = sum(vram for _, vram, _ in services)
        is_over = total_used > gpu_total_mb

        # Sanitize all service names
        sanitized_services = [(sanitize_name(name), vram, state) for name, vram, state in services]

        if is_over:
            return self._render_over_capacity(gpu_total_mb, total_used, sanitized_services, width)

        return self._render_normal(gpu_total_mb, sanitized_services, width)

    @staticmethod
    def _format_segment_label(label: str, num: int) -> str:
        """Format a label to fit within *num* characters with segment markers."""
        if num >= len(label) + 4:
            padded = f"|-- {label} "
            return padded + "-" * (num - len(padded)) + "|"
        if num >= len(label) + 2:
            return f"|{label}" + "-" * (num - len(label) - 2) + "|"
        if num >= 3:
            avail = num - 3  # |--|
            short = label[:avail] if avail > 0 else ""
            return f"|{short}" + "-" * (num - len(short) - 2) + "|"
        return "|" * num

    @staticmethod
    def _format_free_label(free_blocks: int) -> str:
        """Format the free-space segment label."""
        free_label = "free"
        if free_blocks >= len(free_label) + 4:
            padded = f"|  {free_label}"
            return padded + " " * (free_blocks - len(padded) - 1) + "|"
        if free_blocks >= 3:
            return "|" + " " * (free_blocks - 2) + "|"
        return " " * free_blocks

    def _render_normal(
        self,
        gpu_total_mb: int,
        services: list[tuple[str, int, str]],
        width: int,
    ) -> str:
        """Render a normal (within capacity) VRAM bar.

        Parameters
        ----------
        gpu_total_mb:
            Total GPU VRAM in megabytes.
        services:
            Sanitized list of (name, vram_mb, state) tuples.
        width:
            Bar width in characters.

        Returns
        -------
        str
            Multi-line string with bar, labels, and scale.
        """
        total_used = sum(vram for _, vram, _ in services)
        free_mb = gpu_total_mb - total_used

        # Calculate block counts for each service
        blocks: list[tuple[str, int, str, int]] = []  # name, vram, state, num_blocks
        assigned = 0
        for name, vram, state in services:
            num = round(vram / gpu_total_mb * width) if gpu_total_mb > 0 else 0
            blocks.append((name, vram, state, num))
            assigned += num

        free_blocks = width - assigned

        bar_chars = "".join(_FILLED_CHAR * num for _, _, _, num in blocks)
        bar_chars += _FREE_CHAR * max(free_blocks, 0)
        bar_line = f"[{bar_chars}]"

        label_parts = [
            self._format_segment_label(f"{name} ({vram}MB)", num) for name, vram, _, num in blocks
        ]
        if free_blocks > 0:
            label_parts.append(self._format_free_label(free_blocks))
        label_line = "".join(label_parts)

        detail_lines = [f"  {name}: {vram} MB ({state})" for name, vram, state, _ in blocks]
        if free_mb > 0:
            detail_lines.append(f"  free: {free_mb} MB")

        scale_line = self._build_scale(gpu_total_mb, width)

        parts = [bar_line, label_line, scale_line]
        if detail_lines:
            parts.extend(detail_lines)
        return "\n".join(parts)

    def _render_over_capacity(
        self,
        gpu_total_mb: int,
        total_used: int,
        services: list[tuple[str, int, str]],
        width: int,
    ) -> str:
        """Render an over-capacity warning bar.

        Parameters
        ----------
        gpu_total_mb:
            Total GPU VRAM in megabytes.
        total_used:
            Total VRAM required by all services.
        services:
            Sanitized list of (name, vram_mb, state) tuples.
        width:
            Bar width in characters.

        Returns
        -------
        str
            Multi-line string with warning, bar, and labels.
        """
        over = total_used - gpu_total_mb
        warning = (
            f"\u26a0 VRAM OVER CAPACITY: {total_used} MB required, "
            f"{gpu_total_mb} MB available ({over} MB over)"
        )

        # Fill entire bar
        bar_chars = _FILLED_CHAR * width
        bar_line = f"[{bar_chars}] OVER!"

        # Build label line
        label_parts: list[str] = []
        for name, vram, _, num_blocks in [
            (n, v, s, max(1, round(v / total_used * width))) for n, v, s in services
        ]:
            label = f"{name} ({vram}MB)"
            if num_blocks >= len(label) + 4:
                padded = f"|-- {label} "
                padded += "-" * (num_blocks - len(padded)) + "|"
            elif num_blocks >= 3:
                avail = num_blocks - 3
                short = label[:avail] if avail > 0 else ""
                padded = f"|{short}" + "-" * (num_blocks - len(short) - 2) + "|"
            else:
                padded = "|" * num_blocks
            label_parts.append(padded)
        label_line = "".join(label_parts)

        return f"{warning}\n{bar_line}\n{label_line}"

    def _build_scale(self, gpu_total_mb: int, width: int) -> str:
        """Build the scale line with tick marks.

        Parameters
        ----------
        gpu_total_mb:
            Total GPU VRAM in megabytes.
        width:
            Bar width in characters.

        Returns
        -------
        str
            A scale line with 0GB, quartile marks, and max GB.
        """
        # Positions: 0, 25%, 50%, 75%, 100%
        ticks = [
            (0, _mb_to_gb_label(0)),
            (gpu_total_mb // 4, _mb_to_gb_label(gpu_total_mb // 4)),
            (gpu_total_mb // 2, _mb_to_gb_label(gpu_total_mb // 2)),
            (gpu_total_mb * 3 // 4, _mb_to_gb_label(gpu_total_mb * 3 // 4)),
            (gpu_total_mb, _mb_to_gb_label(gpu_total_mb)),
        ]

        # Build a character buffer for the scale
        # +2 for brackets + extra space for last label
        max_label_len = max(len(label) for _, label in ticks) if ticks else 4
        total_width = width + 2 + max_label_len
        scale_buf = [" "] * total_width

        for mb_val, label in ticks:
            pos = round(mb_val / gpu_total_mb * width) if gpu_total_mb > 0 else 0
            # Offset by 1 for the opening bracket
            char_pos = pos + 1
            # Place label starting at char_pos
            for i, ch in enumerate(label):
                idx = char_pos + i
                if 0 <= idx < total_width:
                    scale_buf[idx] = ch

        return "".join(scale_buf).rstrip()


# ---------------------------------------------------------------------------
# StatusPanel — composes VRAMBar with service list into Rich Panel
# ---------------------------------------------------------------------------


class StatusPanel:
    """Composes a VRAMBar with header and service list into a Rich Panel.

    Responsible for the full status display including GPU header,
    VRAM bar, and service list with state colors.
    """

    def __init__(self, bar_width: int = 50) -> None:
        self._bar = VRAMBar()
        self._bar_width = bar_width

    def render(self, system_status: SystemStatus) -> Panel:
        """Render a full system status panel.

        Parameters
        ----------
        system_status:
            The current system status including GPU, VRAM, and services.

        Returns
        -------
        Panel
            A Rich Panel containing header, bar, and service list.
        """
        if system_status.gpu is None:
            return self._render_no_gpu()

        gpu = system_status.gpu
        gpu_total = gpu.vram_total_mb

        # Build service tuples for VRAMBar
        service_tuples: list[tuple[str, int, str]] = []
        for si in system_status.services:
            vram = si.status.vram_mb if si.status.vram_mb is not None else si.service.vram_mb
            service_tuples.append((si.service.name, vram, si.status.state.value))

        # Header
        header = f"{gpu.name} VRAM ({gpu_total} MB)"

        # Bar
        bar_str = self._bar.render(gpu_total, service_tuples, width=self._bar_width)

        # Service list
        service_lines: list[str] = []
        for si in system_status.services:
            state = si.status.state.value
            color = self._bar.state_color(state)
            vram = si.status.vram_mb if si.status.vram_mb is not None else si.service.vram_mb
            name = sanitize_name(si.service.name)
            service_lines.append(
                f"  [{color}]{_FILLED_CHAR}[/{color}] {name}: {vram} MB ({state})"
            )

        # Compose into Rich Text
        content = Text.from_markup(f"[bold]{header}[/bold]\n{bar_str}")
        if service_lines:
            content.append_text(Text.from_markup("\n" + "\n".join(service_lines)))

        return Panel(content, title="GPU Status", border_style="blue")

    def _render_no_gpu(self) -> Panel:
        """Render a panel when no GPU is detected.

        Returns
        -------
        Panel
            A Rich Panel with a 'no GPU detected' message.
        """
        content = Text.from_markup("[dim]No GPU detected[/dim]")
        return Panel(content, title="GPU Status", border_style="red")


# ---------------------------------------------------------------------------
# ComparisonView — current vs proposed VRAM comparison
# ---------------------------------------------------------------------------


class ComparisonView:
    """Renders a side-by-side comparison of current vs proposed VRAM usage.

    Used to preview the effect of mode switches or service changes.
    """

    def __init__(self, bar_width: int = 50) -> None:
        self._bar = VRAMBar()
        self._bar_width = bar_width

    def render(
        self,
        current_services: list[tuple[str, int, str]],
        proposed_services: list[tuple[str, int, str]],
        gpu_total_mb: int,
    ) -> str:
        """Render current and proposed VRAM bars for comparison.

        Parameters
        ----------
        current_services:
            List of (name, vram_mb, state) tuples for current services.
        proposed_services:
            List of (name, vram_mb, state) tuples for proposed services.
        gpu_total_mb:
            Total GPU VRAM in megabytes.

        Returns
        -------
        str
            Multi-line string with "Current:" and "Proposed:" bars.
        """
        current_bar = self._bar.render(gpu_total_mb, current_services, width=self._bar_width)
        proposed_bar = self._bar.render(gpu_total_mb, proposed_services, width=self._bar_width)

        current_used = sum(v for _, v, _ in current_services)
        proposed_used = sum(v for _, v, _ in proposed_services)

        lines = [
            f"Current: ({current_used} MB used)",
            current_bar,
            "",
            f"Proposed: ({proposed_used} MB used)",
            proposed_bar,
        ]

        return "\n".join(lines)
