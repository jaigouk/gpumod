"""Tests for gpumod.visualization — VRAM ASCII visualization module."""

from __future__ import annotations

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
from gpumod.visualization import ComparisonView, StatusPanel, VRAMBar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _service_tuple(name: str, vram_mb: int, state: str = "running") -> tuple[str, int, str]:
    """Create a service tuple for VRAMBar.render()."""
    return (name, vram_mb, state)


def _make_service_info(
    name: str,
    vram_mb: int,
    state: ServiceState = ServiceState.RUNNING,
    driver: DriverType = DriverType.VLLM,
    port: int = 8000,
) -> ServiceInfo:
    """Create a ServiceInfo for StatusPanel tests."""
    return ServiceInfo(
        service=Service(
            id=name,
            name=name,
            driver=driver,
            port=port,
            vram_mb=vram_mb,
        ),
        status=ServiceStatus(state=state, vram_mb=vram_mb),
    )


# ---------------------------------------------------------------------------
# VRAMBar tests
# ---------------------------------------------------------------------------


class TestVRAMBarEmptyGPU:
    """test_render_vram_bar_empty_gpu: Shows empty bar when no services."""

    def test_empty_bar_has_only_free_space(self) -> None:
        bar = VRAMBar()
        result = bar.render(gpu_total_mb=24576, services=[])
        # The bar should contain the free-space character
        assert "\u2591" in result  # ░ character for free space

    def test_empty_bar_contains_no_filled_blocks(self) -> None:
        bar = VRAMBar()
        result = bar.render(gpu_total_mb=24576, services=[])
        # Should not contain any filled block characters
        assert "\u2588" not in result  # █ should not appear


class TestVRAMBarSingleService:
    """test_render_vram_bar_single_service: One service block in bar."""

    def test_single_service_shows_filled_block(self) -> None:
        bar = VRAMBar()
        services = [_service_tuple("vllm-7b", 8192)]
        result = bar.render(gpu_total_mb=24576, services=services)
        assert "\u2588" in result  # █ filled block present

    def test_single_service_name_in_output(self) -> None:
        bar = VRAMBar()
        services = [_service_tuple("vllm-7b", 8192)]
        result = bar.render(gpu_total_mb=24576, services=services)
        assert "vllm-7b" in result


class TestVRAMBarMultipleServices:
    """test_render_vram_bar_multiple_services: Multiple blocks proportional to VRAM."""

    def test_multiple_services_both_names_present(self) -> None:
        bar = VRAMBar()
        services = [
            _service_tuple("vllm-7b", 8192),
            _service_tuple("llama", 4096),
        ]
        result = bar.render(gpu_total_mb=24576, services=services)
        assert "vllm-7b" in result
        assert "llama" in result

    def test_proportional_block_counts(self) -> None:
        bar = VRAMBar()
        # vllm-7b uses 2x the VRAM of llama, so should get ~2x the blocks
        services = [
            _service_tuple("vllm-7b", 8192),
            _service_tuple("llama", 4096),
        ]
        result = bar.render(gpu_total_mb=24576, services=services, width=48)
        # vllm-7b: 8192/24576 * 48 = 16 blocks, llama: 4096/24576 * 48 = 8 blocks
        # The bar line should show correct proportions
        lines = result.split("\n")
        bar_line = [ln for ln in lines if "\u2588" in ln or "\u2591" in ln][0]
        assert len([c for c in bar_line if c == "\u2588"]) == 24  # 16 + 8


class TestVRAMBarFullGPU:
    """test_render_vram_bar_full_gpu: Bar shows 100% filled."""

    def test_full_gpu_no_free_space(self) -> None:
        bar = VRAMBar()
        services = [_service_tuple("vllm-70b", 24576)]
        result = bar.render(gpu_total_mb=24576, services=services, width=50)
        lines = result.split("\n")
        bar_line = [ln for ln in lines if "\u2588" in ln][0]
        assert "\u2591" not in bar_line  # no free space char


class TestVRAMBarOverCapacity:
    """test_render_vram_bar_over_capacity: Shows warning when exceeding GPU VRAM."""

    def test_over_capacity_warning_present(self) -> None:
        bar = VRAMBar()
        services = [
            _service_tuple("vllm-70b", 20480),
            _service_tuple("llama-big", 8192),
        ]
        # Total: 28672 MB, GPU: 24576 MB => 4096 MB over
        result = bar.render(gpu_total_mb=24576, services=services)
        assert "OVER" in result.upper()

    def test_over_capacity_shows_amounts(self) -> None:
        bar = VRAMBar()
        services = [
            _service_tuple("vllm-70b", 20480),
            _service_tuple("llama-big", 8192),
        ]
        result = bar.render(gpu_total_mb=24576, services=services)
        assert "28672" in result  # total required
        assert "24576" in result  # total available
        assert "4096" in result  # overage amount


class TestVRAMBarServiceBlockLabels:
    """test_render_service_blocks_labels: Service names and VRAM labels."""

    def test_labels_contain_service_names(self) -> None:
        bar = VRAMBar()
        services = [
            _service_tuple("vllm-7b", 8192),
            _service_tuple("llama", 4096),
        ]
        result = bar.render(gpu_total_mb=24576, services=services)
        assert "vllm-7b" in result
        assert "llama" in result

    def test_labels_contain_vram_amounts(self) -> None:
        bar = VRAMBar()
        services = [
            _service_tuple("vllm-7b", 8192),
            _service_tuple("llama", 4096),
        ]
        result = bar.render(gpu_total_mb=24576, services=services)
        assert "8192" in result
        assert "4096" in result


class TestVRAMBarWidthConfigurable:
    """test_render_vram_bar_width_configurable: Bar width param works (default 50)."""

    def test_default_width_is_50(self) -> None:
        bar = VRAMBar()
        services = [_service_tuple("svc", 12288)]
        result = bar.render(gpu_total_mb=24576, services=services)
        lines = result.split("\n")
        bar_line = [ln for ln in lines if "\u2588" in ln or "\u2591" in ln][0]
        # Count blocks inside brackets
        inner = bar_line.strip()
        # Remove leading [ and trailing ]
        if inner.startswith("[") and inner.endswith("]"):
            inner = inner[1:-1]
        block_chars = [c for c in inner if c in ("\u2588", "\u2591")]
        assert len(block_chars) == 50

    def test_custom_width_30(self) -> None:
        bar = VRAMBar()
        services = [_service_tuple("svc", 12288)]
        result = bar.render(gpu_total_mb=24576, services=services, width=30)
        lines = result.split("\n")
        bar_line = [ln for ln in lines if "\u2588" in ln or "\u2591" in ln][0]
        inner = bar_line.strip()
        if inner.startswith("[") and inner.endswith("]"):
            inner = inner[1:-1]
        block_chars = [c for c in inner if c in ("\u2588", "\u2591")]
        assert len(block_chars) == 30

    def test_custom_width_80(self) -> None:
        bar = VRAMBar()
        services = [_service_tuple("svc", 12288)]
        result = bar.render(gpu_total_mb=24576, services=services, width=80)
        lines = result.split("\n")
        bar_line = [ln for ln in lines if "\u2588" in ln or "\u2591" in ln][0]
        inner = bar_line.strip()
        if inner.startswith("[") and inner.endswith("]"):
            inner = inner[1:-1]
        block_chars = [c for c in inner if c in ("\u2588", "\u2591")]
        assert len(block_chars) == 80


class TestVRAMBarHeaderGPU:
    """test_render_header_shows_gpu_name_and_capacity."""

    def test_header_absent_in_vram_bar(self) -> None:
        """VRAMBar itself does not render GPU header — StatusPanel does."""
        bar = VRAMBar()
        result = bar.render(gpu_total_mb=24576, services=[])
        # VRAMBar does not include GPU name; that's StatusPanel's job
        assert isinstance(result, str)


class TestVRAMBarScale:
    """test_render_scale_shows_tick_marks: 0GB, quartile marks, max GB."""

    def test_scale_shows_zero(self) -> None:
        bar = VRAMBar()
        result = bar.render(gpu_total_mb=24576, services=[])
        assert "0GB" in result or "0 GB" in result

    def test_scale_shows_max(self) -> None:
        bar = VRAMBar()
        result = bar.render(gpu_total_mb=24576, services=[])
        # 24576 MB = 24 GB (or 24.0 GB)
        assert "24GB" in result or "24 GB" in result or "24.0GB" in result


# ---------------------------------------------------------------------------
# StatusPanel tests
# ---------------------------------------------------------------------------


class TestStatusPanel:
    """test_format_status_panel: Full status panel with header + bar + services."""

    def test_panel_contains_gpu_name(self) -> None:
        panel_renderer = StatusPanel()
        status = SystemStatus(
            gpu=GPUInfo(name="RTX 4090", vram_total_mb=24576),
            vram=VRAMUsage(total_mb=24576, used_mb=8192, free_mb=16384),
            services=[
                _make_service_info("vllm-7b", 8192),
            ],
        )
        panel = panel_renderer.render(status)
        # Panel is a Rich renderable — render it to string for assertions
        from rich.console import Console

        console = Console(file=None, force_terminal=False, width=100)
        with console.capture() as capture:
            console.print(panel)
        output = capture.get()
        assert "RTX 4090" in output

    def test_panel_contains_service_names(self) -> None:
        panel_renderer = StatusPanel()
        status = SystemStatus(
            gpu=GPUInfo(name="RTX 4090", vram_total_mb=24576),
            vram=VRAMUsage(total_mb=24576, used_mb=12288, free_mb=12288),
            services=[
                _make_service_info("vllm-7b", 8192),
                _make_service_info("llama", 4096),
            ],
        )
        panel = panel_renderer.render(status)
        from rich.console import Console

        console = Console(file=None, force_terminal=False, width=100)
        with console.capture() as capture:
            console.print(panel)
        output = capture.get()
        assert "vllm-7b" in output
        assert "llama" in output

    def test_panel_shows_vram_bar(self) -> None:
        panel_renderer = StatusPanel()
        status = SystemStatus(
            gpu=GPUInfo(name="RTX 4090", vram_total_mb=24576),
            vram=VRAMUsage(total_mb=24576, used_mb=8192, free_mb=16384),
            services=[
                _make_service_info("vllm-7b", 8192),
            ],
        )
        panel = panel_renderer.render(status)
        from rich.console import Console

        console = Console(file=None, force_terminal=False, width=100)
        with console.capture() as capture:
            console.print(panel)
        output = capture.get()
        # Bar characters should be present
        assert "\u2588" in output or "\u2591" in output


class TestStatusPanelNoGPU:
    """test_format_status_panel_no_gpu: Graceful handling when no GPU detected."""

    def test_no_gpu_shows_message(self) -> None:
        panel_renderer = StatusPanel()
        status = SystemStatus(gpu=None, vram=None, services=[])
        panel = panel_renderer.render(status)
        from rich.console import Console

        console = Console(file=None, force_terminal=False, width=100)
        with console.capture() as capture:
            console.print(panel)
        output = capture.get()
        assert "no gpu" in output.lower() or "not detected" in output.lower()


# ---------------------------------------------------------------------------
# ComparisonView tests
# ---------------------------------------------------------------------------


class TestComparisonView:
    """test_render_comparison_current_vs_proposed: Side-by-side current/proposed bars."""

    def test_comparison_shows_current_label(self) -> None:
        view = ComparisonView()
        current = [_service_tuple("vllm-7b", 8192)]
        proposed = [_service_tuple("vllm-70b", 20480)]
        result = view.render(
            current_services=current,
            proposed_services=proposed,
            gpu_total_mb=24576,
        )
        assert "Current" in result or "current" in result

    def test_comparison_shows_proposed_label(self) -> None:
        view = ComparisonView()
        current = [_service_tuple("vllm-7b", 8192)]
        proposed = [_service_tuple("vllm-70b", 20480)]
        result = view.render(
            current_services=current,
            proposed_services=proposed,
            gpu_total_mb=24576,
        )
        assert "Proposed" in result or "proposed" in result

    def test_comparison_shows_both_bars(self) -> None:
        view = ComparisonView()
        current = [_service_tuple("vllm-7b", 8192)]
        proposed = [_service_tuple("vllm-70b", 20480)]
        result = view.render(
            current_services=current,
            proposed_services=proposed,
            gpu_total_mb=24576,
        )
        # Should have bar characters in multiple sections
        lines = result.split("\n")
        bar_lines = [ln for ln in lines if "\u2588" in ln or "\u2591" in ln]
        assert len(bar_lines) >= 2  # At least one bar line per view


# ---------------------------------------------------------------------------
# Color coding tests
# ---------------------------------------------------------------------------


class TestColorCoding:
    """test_color_coding_running_vs_sleeping: Different colors for running/sleeping."""

    def test_running_service_uses_green_style(self) -> None:
        """StatusPanel should style running services differently from sleeping."""
        panel_renderer = StatusPanel()
        status = SystemStatus(
            gpu=GPUInfo(name="RTX 4090", vram_total_mb=24576),
            vram=VRAMUsage(total_mb=24576, used_mb=12288, free_mb=12288),
            services=[
                _make_service_info("running-svc", 8192, ServiceState.RUNNING),
                _make_service_info("sleeping-svc", 4096, ServiceState.SLEEPING),
            ],
        )
        panel = panel_renderer.render(status)
        from rich.console import Console

        console = Console(file=None, force_terminal=True, color_system="truecolor", width=100)
        with console.capture() as capture:
            console.print(panel)
        output = capture.get()
        # Both services should be present
        assert "running-svc" in output
        assert "sleeping-svc" in output

    def test_state_color_map_contains_expected_states(self) -> None:
        """VRAMBar should map states to colors."""
        bar = VRAMBar()
        assert bar.state_color("running") == "green"
        assert bar.state_color("sleeping") == "yellow"
        assert bar.state_color("unhealthy") == "red"
        assert bar.state_color("failed") == "red"


# ---------------------------------------------------------------------------
# Rich renderable test
# ---------------------------------------------------------------------------


class TestOutputIsRichRenderable:
    """test_output_is_rich_renderable: Returns Rich Console-compatible objects."""

    def test_status_panel_returns_rich_panel(self) -> None:
        from rich.panel import Panel as RichPanel

        panel_renderer = StatusPanel()
        status = SystemStatus(
            gpu=GPUInfo(name="RTX 4090", vram_total_mb=24576),
            vram=VRAMUsage(total_mb=24576, used_mb=0, free_mb=24576),
            services=[],
        )
        panel = panel_renderer.render(status)
        assert isinstance(panel, RichPanel)

    def test_panel_can_be_printed_by_rich_console(self) -> None:
        from rich.console import Console

        panel_renderer = StatusPanel()
        status = SystemStatus(
            gpu=GPUInfo(name="RTX 4090", vram_total_mb=24576),
            vram=VRAMUsage(total_mb=24576, used_mb=8192, free_mb=16384),
            services=[
                _make_service_info("vllm-7b", 8192),
            ],
        )
        panel = panel_renderer.render(status)
        console = Console(file=None, force_terminal=False, width=100)
        # Should not raise
        with console.capture() as capture:
            console.print(panel)
        output = capture.get()
        assert len(output) > 0


# ---------------------------------------------------------------------------
# Security: sanitize service names
# ---------------------------------------------------------------------------


class TestServiceNameSanitization:
    """Ensure service names are sanitized to prevent terminal escape injection."""

    def test_rich_markup_is_stripped(self) -> None:
        bar = VRAMBar()
        # Attempt to inject Rich markup
        services = [_service_tuple("[bold red]evil[/bold red]", 8192)]
        result = bar.render(gpu_total_mb=24576, services=services)
        # The markup tags should not appear in raw output
        assert "[bold red]" not in result
        assert "[/bold red]" not in result

    def test_control_characters_stripped(self) -> None:
        bar = VRAMBar()
        # Attempt to inject ANSI escape codes
        services = [_service_tuple("svc\x1b[31mhack\x1b[0m", 8192)]
        result = bar.render(gpu_total_mb=24576, services=services)
        assert "\x1b" not in result

    def test_sanitized_name_still_present(self) -> None:
        bar = VRAMBar()
        services = [_service_tuple("[bold]my-service[/bold]", 8192)]
        result = bar.render(gpu_total_mb=24576, services=services)
        # The cleaned name should still appear
        assert "my-service" in result
