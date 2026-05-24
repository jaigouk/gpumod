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

        mock_db = AsyncMock()
        mock_db.get_setting = AsyncMock(return_value=None)
        lifecycle_mgr = LifecycleManager(registry, db=mock_db)

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


class TestPreflightRunningServicesContext:
    """gpumod-lgt: when preflight refuses to start a service, the
    LifecycleError must enumerate currently-running gpumod services so
    the operator knows what to stop."""

    @pytest.mark.asyncio
    async def test_running_services_appended_to_error_message(self) -> None:
        from unittest.mock import patch

        from gpumod.preflight import CheckResult

        target = _make_service("qwen36-35b-a3b-mtp-iq4xs-preserve", vram_mb=22000)
        # Other running services that should appear in the error context
        running_others = [
            _make_service("vllm-embedding-code", vram_mb=2500),
            _make_service("qwen36-35b-a3b-iq4xs", vram_mb=22000),
        ]
        registry = _build_mock_registry(target)
        registry.list_running = AsyncMock(return_value=running_others)
        driver = _build_mock_driver()
        registry.get_driver = lambda dtype: driver

        error_results = (
            {
                "ram": CheckResult(
                    passed=False,
                    severity="error",
                    message="System RAM insufficient to mmap 18000 MB model",
                    remediation="See gpu-stability.md",
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
            pytest.raises(LifecycleError) as exc_info,
        ):
            await lifecycle_mgr.start("qwen36-35b-a3b-mtp-iq4xs-preserve")

        msg = exc_info.value.reason
        # The original preflight error must be preserved
        assert "RAM insufficient" in msg
        # The running-services context must be appended
        assert "vllm-embedding-code" in msg
        assert "qwen36-35b-a3b-iq4xs" in msg
        # Actionable stop commands present
        assert "gpumod service stop vllm-embedding-code" in msg
        assert "gpumod service stop qwen36-35b-a3b-iq4xs" in msg
        # The service being attempted must NOT be in the running list
        assert "gpumod service stop qwen36-35b-a3b-iq4xs-preserve" not in msg

    @pytest.mark.asyncio
    async def test_no_context_block_when_no_other_services_running(self) -> None:
        from unittest.mock import patch

        from gpumod.preflight import CheckResult

        target = _make_service("solo-service", vram_mb=22000)
        registry = _build_mock_registry(target)
        registry.list_running = AsyncMock(return_value=[])
        driver = _build_mock_driver()
        registry.get_driver = lambda dtype: driver

        error_results = (
            {
                "vram": CheckResult(
                    passed=False,
                    severity="error",
                    message="VRAM insufficient",
                    remediation="Reduce ctx_size",
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
            pytest.raises(LifecycleError) as exc_info,
        ):
            await lifecycle_mgr.start("solo-service")

        msg = exc_info.value.reason
        assert "VRAM insufficient" in msg
        # No running-services section appended when none are running
        assert "Currently running gpumod services" not in msg

    @pytest.mark.asyncio
    async def test_registry_failure_does_not_mask_original_error(self) -> None:
        """If list_running() blows up, the original preflight error is
        still surfaced — best-effort enricher."""
        from unittest.mock import patch

        from gpumod.preflight import CheckResult

        target = _make_service("flaky-svc", vram_mb=22000)
        registry = _build_mock_registry(target)
        registry.list_running = AsyncMock(side_effect=RuntimeError("registry-boom"))
        driver = _build_mock_driver()
        registry.get_driver = lambda dtype: driver

        error_results = (
            {
                "vram": CheckResult(
                    passed=False,
                    severity="error",
                    message="VRAM insufficient (original)",
                    remediation="Reduce vram_mb",
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
            pytest.raises(LifecycleError) as exc_info,
        ):
            await lifecycle_mgr.start("flaky-svc")

        # Original failure still visible despite enrichment failure
        assert "VRAM insufficient (original)" in exc_info.value.reason


# ── Quiesce gate integration ─────────────────────────────────────────────


class TestQuiesceGate:
    """gpumod-jj0: quiesce gate blocks heavy starts within cooldown window."""

    @pytest.mark.asyncio
    async def test_heavy_stop_then_heavy_start_within_window_raises(self) -> None:
        """Stopping a heavy service then starting another heavy within
        the quiesce window must raise LifecycleError."""
        import time
        from unittest.mock import patch

        service = _make_service("vllm-new", vram_mb=8000)
        registry = _build_mock_registry(service)
        driver = _build_mock_driver()
        registry.get_driver = lambda dtype: driver

        # Simulate a recent heavy stop (3s ago, window 10s)
        recent_stop = str(time.time() - 3)
        mock_db = AsyncMock()
        mock_db.get_setting = AsyncMock(return_value=recent_stop)

        lifecycle_mgr = LifecycleManager(registry, db=mock_db)

        with (
            patch(
                "gpumod.preflight.run_preflight",
                new_callable=AsyncMock,
                return_value=({}, False),
            ),
            pytest.raises(LifecycleError, match="Quiesce period active"),
        ):
            await lifecycle_mgr.start("vllm-new")

        driver.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_heavy_stop_then_heavy_start_outside_window_allowed(self) -> None:
        """Starting a heavy service after the quiesce window has elapsed succeeds."""
        import time
        from unittest.mock import patch

        service = _make_service("vllm-new", vram_mb=8000)
        registry = _build_mock_registry(service)
        driver = _build_mock_driver()
        registry.get_driver = lambda dtype: driver

        # Heavy stop 20s ago, window 10s → allowed
        old_stop = str(time.time() - 20)
        mock_db = AsyncMock()
        mock_db.get_setting = AsyncMock(return_value=old_stop)

        lifecycle_mgr = LifecycleManager(registry, db=mock_db)

        with patch(
            "gpumod.preflight.run_preflight",
            new_callable=AsyncMock,
            return_value=({}, False),
        ):
            await lifecycle_mgr.start("vllm-new")

        driver.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_heavy_stop_then_light_start_no_quiesce_needed(self) -> None:
        """Starting a light service (vram_mb=0) never triggers the quiesce gate."""
        import time
        from unittest.mock import patch

        service = _make_service("api-svc", vram_mb=0)
        registry = _build_mock_registry(service)
        driver = _build_mock_driver()
        registry.get_driver = lambda dtype: driver

        # Even with a very recent heavy stop, light services are unaffected
        recent_stop = str(time.time() - 1)
        mock_db = AsyncMock()
        mock_db.get_setting = AsyncMock(return_value=recent_stop)

        lifecycle_mgr = LifecycleManager(registry, db=mock_db)

        with patch(
            "gpumod.preflight.run_preflight",
            new_callable=AsyncMock,
            return_value=({}, False),
        ):
            await lifecycle_mgr.start("api-svc")

        driver.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_quiesce_flag_bypasses_gate(self) -> None:
        """--no-quiesce bypasses the quiesce gate even within window."""
        import time
        from unittest.mock import patch

        service = _make_service("vllm-new", vram_mb=8000)
        registry = _build_mock_registry(service)
        driver = _build_mock_driver()
        registry.get_driver = lambda dtype: driver

        # Very recent heavy stop
        recent_stop = str(time.time() - 1)
        mock_db = AsyncMock()
        mock_db.get_setting = AsyncMock(return_value=recent_stop)

        lifecycle_mgr = LifecycleManager(registry, db=mock_db)

        with patch(
            "gpumod.preflight.run_preflight",
            new_callable=AsyncMock,
            return_value=({}, False),
        ):
            await lifecycle_mgr.start("vllm-new", no_quiesce=True)

        driver.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_heavy_service_records_timestamp(self) -> None:
        """Stopping a heavy service records the timestamp in the DB."""
        from gpumod.services.heavy import QUIESCE_LAST_HEAVY_STOP_KEY

        service = _make_service("vllm-chat", vram_mb=8000)
        registry = _build_mock_registry(service)
        driver = _build_mock_driver(state=ServiceState.RUNNING)
        registry.get_driver = lambda dtype: driver

        mock_db = AsyncMock()
        mock_db.set_setting = AsyncMock()

        lifecycle_mgr = LifecycleManager(registry, db=mock_db)
        await lifecycle_mgr.stop("vllm-chat")

        mock_db.set_setting.assert_called_once()
        call_args = mock_db.set_setting.call_args
        assert call_args[0][0] == QUIESCE_LAST_HEAVY_STOP_KEY

    @pytest.mark.asyncio
    async def test_stop_light_service_does_not_record(self) -> None:
        """Stopping a light service does NOT record a quiesce timestamp."""
        service = _make_service("api-svc", vram_mb=0)
        registry = _build_mock_registry(service)
        driver = _build_mock_driver(state=ServiceState.RUNNING)
        registry.get_driver = lambda dtype: driver

        mock_db = AsyncMock()
        mock_db.set_setting = AsyncMock()

        lifecycle_mgr = LifecycleManager(registry, db=mock_db)
        await lifecycle_mgr.stop("api-svc")

        mock_db.set_setting.assert_not_called()
