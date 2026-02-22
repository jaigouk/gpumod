"""Tests for lifecycle error classification and structured error reporting (gpumod-bc1)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpumod.models import DriverType, Service, ServiceState, ServiceStatus, SleepMode
from gpumod.services.lifecycle import LifecycleError, LifecycleManager, classify_journal_error

# ── Journal error classification ──────────────────────────────────────


class TestClassifyJournalError:
    """classify_journal_error() should detect known fatal patterns in journal lines."""

    def test_detects_cuda_malloc_oom(self) -> None:
        lines = [
            "Loading model...",
            "RuntimeError: cudaMalloc failed: out of memory",
            "Traceback (most recent call last):",
        ]
        result = classify_journal_error(lines)
        assert result is not None
        assert result["category"] == "cuda_oom"

    def test_detects_cuda_error_oom(self) -> None:
        lines = ["CUDA error: out of memory"]
        result = classify_journal_error(lines)
        assert result is not None
        assert result["category"] == "cuda_oom"

    def test_detects_torch_oom(self) -> None:
        lines = ["torch.OutOfMemoryError: CUDA out of memory"]
        result = classify_journal_error(lines)
        assert result is not None
        assert result["category"] == "cuda_oom"

    def test_detects_torch_cuda_oom(self) -> None:
        lines = ["torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"]
        result = classify_journal_error(lines)
        assert result is not None
        assert result["category"] == "cuda_oom"

    def test_detects_model_load_failure(self) -> None:
        lines = ["ERROR: failed to load model '/models/my-model.gguf'"]
        result = classify_journal_error(lines)
        assert result is not None
        assert result["category"] == "model_load_failed"

    def test_detects_missing_model_file(self) -> None:
        lines = ["No such file or directory: '/models/qwen.gguf'"]
        result = classify_journal_error(lines)
        assert result is not None
        assert result["category"] == "model_not_found"

    def test_detects_missing_safetensors_file(self) -> None:
        lines = ["FileNotFoundError: No such file or directory: '/models/model.safetensors'"]
        result = classify_journal_error(lines)
        assert result is not None
        assert result["category"] == "model_not_found"

    def test_returns_none_for_no_match(self) -> None:
        lines = ["Starting server...", "Listening on port 8080"]
        result = classify_journal_error(lines)
        assert result is None

    def test_returns_none_for_empty_lines(self) -> None:
        result = classify_journal_error([])
        assert result is None

    def test_result_contains_matched_line(self) -> None:
        lines = [
            "Starting up...",
            "RuntimeError: cudaMalloc failed: out of memory",
        ]
        result = classify_journal_error(lines)
        assert result is not None
        assert "cudaMalloc" in result["matched_line"]

    def test_result_contains_suggestion(self) -> None:
        lines = ["torch.OutOfMemoryError: CUDA out of memory"]
        result = classify_journal_error(lines)
        assert result is not None
        assert "suggestion" in result
        assert len(result["suggestion"]) > 0

    # ── Edge cases ────────────────────────────────────────────────────

    def test_first_match_wins_when_multiple_patterns(self) -> None:
        """When multiple patterns match different lines, the first match wins."""
        lines = [
            "CUDA error: out of memory",
            "failed to load model '/models/foo.gguf'",
        ]
        result = classify_journal_error(lines)
        assert result is not None
        assert result["category"] == "cuda_oom"

    def test_pattern_case_insensitive(self) -> None:
        lines = ["CUDAMALLOC FAILED: OUT OF MEMORY"]
        result = classify_journal_error(lines)
        assert result is not None
        assert result["category"] == "cuda_oom"

    def test_pattern_embedded_in_longer_line(self) -> None:
        lines = [
            "Feb 21 12:00:00 host python[1234]: "
            "RuntimeError: cudaMalloc failed: out of memory (tried to allocate 4GB)"
        ]
        result = classify_journal_error(lines)
        assert result is not None
        assert result["category"] == "cuda_oom"

    def test_no_false_positive_on_oom_substring(self) -> None:
        """'memory' alone or partial matches should not trigger."""
        lines = ["Allocated 8GB of memory successfully", "Memory usage: 4GB"]
        result = classify_journal_error(lines)
        assert result is None

    def test_no_false_positive_on_unrelated_file_not_found(self) -> None:
        """File-not-found without .gguf/.safetensors extension should not match."""
        lines = ["No such file or directory: '/etc/config.yaml'"]
        result = classify_journal_error(lines)
        assert result is None

    def test_handles_binary_garbage_in_lines(self) -> None:
        """Lines with non-UTF8 or weird chars should not crash."""
        lines = ["\x00\xff\xfe binary garbage", "normal line"]
        result = classify_journal_error(lines)
        assert result is None


# ── MCP tool error handling ───────────────────────────────────────────


def _make_mock_ctx(*, manager: AsyncMock | None = None) -> MagicMock:
    """Minimal mock FastMCP context with a manager in lifespan state."""
    ctx = MagicMock()
    lifespan: dict[str, Any] = {}
    if manager is not None:
        lifespan["manager"] = manager
    ctx.fastmcp._lifespan_result = lifespan
    return ctx


class TestMcpStartServiceErrorHandling:
    """start_service() should catch LifecycleError and return structured error."""

    @pytest.mark.asyncio
    async def test_lifecycle_error_returns_error_dict(self) -> None:
        from gpumod.mcp_tools import start_service

        manager = AsyncMock()
        manager.start_service.side_effect = LifecycleError(
            service_id="vllm-chat",
            operation="start",
            reason=(
                "process exited (failed)\n--- journal tail ---\ncudaMalloc failed: out of memory"
            ),
        )
        ctx = _make_mock_ctx(manager=manager)

        result = await start_service(service_id="vllm-chat", ctx=ctx)

        assert result["success"] is False
        assert "vllm-chat" in result["service_id"]
        assert "error" in result

    @pytest.mark.asyncio
    async def test_lifecycle_error_includes_reason(self) -> None:
        from gpumod.mcp_tools import start_service

        manager = AsyncMock()
        manager.start_service.side_effect = LifecycleError(
            service_id="vllm-chat",
            operation="start",
            reason="health check timed out after 120s",
        )
        ctx = _make_mock_ctx(manager=manager)

        result = await start_service(service_id="vllm-chat", ctx=ctx)

        assert result["success"] is False
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_insufficient_vram_error_returns_error_dict(self) -> None:
        from gpumod.mcp_tools import start_service
        from gpumod.services.vram import InsufficientVRAMError

        manager = AsyncMock()
        manager.start_service.side_effect = InsufficientVRAMError(
            required_mb=22000, available_mb=15000
        )
        ctx = _make_mock_ctx(manager=manager)

        result = await start_service(service_id="vllm-chat", ctx=ctx)

        assert result["success"] is False
        assert result["required_mb"] == 22000
        assert result["available_mb"] == 15000


class TestMcpSwitchModeErrorHandling:
    """switch_mode() should catch LifecycleError and return structured error."""

    @pytest.mark.asyncio
    async def test_lifecycle_error_returns_error_dict(self) -> None:
        from gpumod.mcp_tools import switch_mode

        manager = AsyncMock()
        manager.switch_mode.side_effect = LifecycleError(
            service_id="vllm-chat",
            operation="start",
            reason="process exited (failed)",
        )
        ctx = _make_mock_ctx(manager=manager)

        result = await switch_mode(mode_id="code", ctx=ctx)

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_insufficient_vram_error_returns_error_dict(self) -> None:
        from gpumod.mcp_tools import switch_mode
        from gpumod.services.vram import InsufficientVRAMError

        manager = AsyncMock()
        manager.switch_mode.side_effect = InsufficientVRAMError(
            required_mb=22000, available_mb=15000
        )
        ctx = _make_mock_ctx(manager=manager)

        result = await switch_mode(mode_id="code", ctx=ctx)

        assert result["success"] is False
        assert result["required_mb"] == 22000
        assert result["available_mb"] == 15000


# ── VRAM preflight in lifecycle.start() ───────────────────────────────


def _make_service(
    service_id: str = "vllm-chat",
    vram_mb: int = 8000,
    driver: DriverType = DriverType.VLLM,
) -> Service:
    return Service(
        id=service_id,
        name=f"Test {service_id}",
        driver=driver,
        port=8000,
        vram_mb=vram_mb,
        sleep_mode=SleepMode.NONE,
        health_endpoint="/health",
        unit_name=f"{service_id}.service",
        depends_on=[],
        startup_timeout=60,
        extra_config={},
    )


def _build_mock_registry(service: Service) -> AsyncMock:
    registry = AsyncMock()
    registry.get = AsyncMock(return_value=service)
    registry.list_all = AsyncMock(return_value=[service])
    registry.get_dependents = AsyncMock(return_value=[])
    return registry


def _build_mock_driver(
    healthy: bool = True, state: ServiceState = ServiceState.STOPPED
) -> AsyncMock:
    driver = AsyncMock()
    driver.start = AsyncMock()
    driver.stop = AsyncMock()
    driver.supports_sleep = False
    driver.health_check = AsyncMock(return_value=healthy)
    driver.status = AsyncMock(return_value=ServiceStatus(state=state))
    return driver


class TestVRAMPreflight:
    """lifecycle.start() uses run_preflight() to validate before starting services."""

    @pytest.mark.asyncio
    async def test_raises_when_preflight_has_errors(self) -> None:
        """start() raises LifecycleError when preflight detects errors."""
        from unittest.mock import patch

        from gpumod.preflight import CheckResult

        service = _make_service(vram_mb=22000)
        registry = _build_mock_registry(service)
        driver = _build_mock_driver()
        registry.get_driver = lambda dtype: driver

        error_results = (
            {
                "vram": CheckResult(
                    passed=False,
                    severity="error",
                    message="VRAM insufficient: 22000 MB required exceeds 15000 MB",
                    remediation="Reduce n_gpu_layers",
                ),
            },
            True,
        )

        lifecycle_mgr = LifecycleManager(registry)

        with (
            patch(
                "gpumod.preflight.run_preflight",
                new_callable=AsyncMock,
                return_value=error_results,
            ),
            pytest.raises(LifecycleError, match="VRAM insufficient"),
        ):
            await lifecycle_mgr.start("vllm-chat")

        driver.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_proceeds_when_preflight_passes(self) -> None:
        """start() proceeds normally when preflight passes."""
        from unittest.mock import patch

        service = _make_service(vram_mb=8000)
        registry = _build_mock_registry(service)
        driver = _build_mock_driver()
        registry.get_driver = lambda dtype: driver

        lifecycle_mgr = LifecycleManager(registry)

        with patch(
            "gpumod.preflight.run_preflight",
            new_callable=AsyncMock,
            return_value=({}, False),
        ):
            await lifecycle_mgr.start("vllm-chat")

        driver.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_already_running_service(self) -> None:
        """Already-running services should not be preflight-checked."""
        from unittest.mock import patch

        service = _make_service(vram_mb=22000)
        registry = _build_mock_registry(service)
        driver = _build_mock_driver(state=ServiceState.RUNNING)
        registry.get_driver = lambda dtype: driver

        lifecycle_mgr = LifecycleManager(registry)

        with patch(
            "gpumod.preflight.run_preflight",
            new_callable=AsyncMock,
            return_value=({}, False),
        ) as mock_preflight:
            await lifecycle_mgr.start("vllm-chat")

        # Should skip — already running, no preflight or start called
        driver.start.assert_not_called()
        mock_preflight.assert_not_called()
