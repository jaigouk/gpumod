"""Tests for MCP tools — 9 tools (6 read-only + 3 mutating) with validation and error handling."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Helpers / Factories
# ---------------------------------------------------------------------------


def _make_service(
    *,
    service_id: str = "vllm-chat",
    name: str = "vLLM Chat",
    driver: str = "vllm",
    port: int = 8000,
    vram_mb: int = 4000,
    model_id: str | None = None,
) -> Any:
    """Create a Service model instance."""
    from gpumod.models import DriverType, Service

    return Service(
        id=service_id,
        name=name,
        driver=DriverType(driver),
        port=port,
        vram_mb=vram_mb,
        model_id=model_id,
    )


def _make_mode(
    *,
    mode_id: str = "chat",
    name: str = "Chat Mode",
    description: str | None = "Conversational AI",
    total_vram_mb: int | None = 8000,
) -> Any:
    """Create a Mode model instance."""
    from gpumod.models import Mode

    return Mode(
        id=mode_id,
        name=name,
        description=description,
        total_vram_mb=total_vram_mb,
    )


def _make_model_info(
    *,
    model_id: str = "meta-llama/Llama-3.1-8B",
    source: str = "huggingface",
    parameters_b: float | None = 8.0,
    architecture: str | None = "llama",
    base_vram_mb: int | None = 16000,
) -> Any:
    """Create a ModelInfo model instance."""
    from gpumod.models import ModelInfo, ModelSource

    return ModelInfo(
        id=model_id,
        source=ModelSource(source),
        parameters_b=parameters_b,
        architecture=architecture,
        base_vram_mb=base_vram_mb,
    )


def _make_system_status(
    *,
    current_mode: str | None = "chat",
    services: list[Any] | None = None,
) -> Any:
    """Create a SystemStatus model instance."""
    from gpumod.models import (
        GPUInfo,
        ServiceInfo,
        ServiceState,
        ServiceStatus,
        SystemStatus,
        VRAMUsage,
    )

    gpu = GPUInfo(name="RTX 4090", vram_total_mb=24576)
    vram = VRAMUsage(total_mb=24576, used_mb=8000, free_mb=16576)
    svc_infos = services or [
        ServiceInfo(
            service=_make_service(),
            status=ServiceStatus(state=ServiceState.RUNNING, vram_mb=4000),
        )
    ]
    return SystemStatus(
        gpu=gpu,
        vram=vram,
        current_mode=current_mode,
        services=svc_infos,
    )


def _make_mode_result(
    *,
    success: bool = True,
    mode_id: str = "chat",
    started: list[str] | None = None,
    stopped: list[str] | None = None,
) -> Any:
    """Create a ModeResult model instance."""
    from gpumod.models import ModeResult

    return ModeResult(
        success=success,
        mode_id=mode_id,
        started=started or [],
        stopped=stopped or [],
    )


def _make_simulation_result(
    *,
    fits: bool = True,
    gpu_total_mb: int = 24576,
    proposed_usage_mb: int = 8000,
) -> Any:
    """Create a SimulationResult model instance."""
    from gpumod.models import SimulationResult

    return SimulationResult(
        fits=fits,
        gpu_total_mb=gpu_total_mb,
        current_usage_mb=0,
        proposed_usage_mb=proposed_usage_mb,
        headroom_mb=gpu_total_mb - proposed_usage_mb,
        services=[_make_service()],
        alternatives=[],
    )


def _make_mock_ctx(
    *,
    manager: AsyncMock | None = None,
    db: AsyncMock | None = None,
    model_registry: AsyncMock | None = None,
    simulation: AsyncMock | None = None,
    lifecycle: AsyncMock | None = None,
) -> MagicMock:
    """Create a mock FastMCP Context with lifespan context dict."""
    ctx = MagicMock()
    lifespan: dict[str, Any] = {}
    if manager is not None:
        lifespan["manager"] = manager
    if db is not None:
        lifespan["db"] = db
    if model_registry is not None:
        lifespan["model_registry"] = model_registry
    if simulation is not None:
        lifespan["simulation"] = simulation
    if lifecycle is not None:
        lifespan["lifecycle"] = lifecycle
    ctx.fastmcp._lifespan_result = lifespan
    return ctx


# ---------------------------------------------------------------------------
# TestReadOnlyTools
# ---------------------------------------------------------------------------


class TestReadOnlyTools:
    """Tests for the 6 read-only MCP tools."""

    async def test_gpu_status_returns_system_status(self) -> None:
        from gpumod.mcp_tools import gpu_status

        status = _make_system_status()
        manager = AsyncMock()
        manager.get_status.return_value = status
        ctx = _make_mock_ctx(manager=manager)

        result = await gpu_status(ctx=ctx)

        assert result == status.model_dump(mode="json")
        manager.get_status.assert_awaited_once()

    async def test_list_services_returns_all_services(self) -> None:
        from gpumod.mcp_tools import list_services

        services = [_make_service(), _make_service(service_id="fastapi-web", driver="fastapi")]
        db = AsyncMock()
        db.list_services.return_value = services
        ctx = _make_mock_ctx(db=db)

        result = await list_services(ctx=ctx)

        assert result["services"] is not None
        assert len(result["services"]) == 2
        db.list_services.assert_awaited_once()

    async def test_list_modes_returns_all_modes(self) -> None:
        from gpumod.mcp_tools import list_modes

        modes = [_make_mode(), _make_mode(mode_id="code", name="Code Mode")]
        db = AsyncMock()
        db.list_modes.return_value = modes
        ctx = _make_mock_ctx(db=db)

        result = await list_modes(ctx=ctx)

        assert result["modes"] is not None
        assert len(result["modes"]) == 2
        db.list_modes.assert_awaited_once()

    async def test_service_info_returns_service_details(self) -> None:
        from gpumod.mcp_tools import service_info

        svc = _make_service()
        db = AsyncMock()
        db.get_service.return_value = svc
        ctx = _make_mock_ctx(db=db)

        result = await service_info(service_id="vllm-chat", ctx=ctx)

        assert result["id"] == "vllm-chat"
        assert result["driver"] == "vllm"
        db.get_service.assert_awaited_once_with("vllm-chat")

    async def test_service_info_not_found_returns_error(self) -> None:
        from gpumod.mcp_tools import service_info

        db = AsyncMock()
        db.get_service.return_value = None
        ctx = _make_mock_ctx(db=db)

        result = await service_info(service_id="nonexistent", ctx=ctx)

        assert result["code"] == "NOT_FOUND"
        assert "error" in result

    async def test_model_info_returns_model_details(self) -> None:
        from gpumod.mcp_tools import model_info

        model = _make_model_info()
        model_reg = AsyncMock()
        model_reg.get.return_value = model
        ctx = _make_mock_ctx(model_registry=model_reg)

        result = await model_info(model_id="meta-llama/Llama-3.1-8B", ctx=ctx)

        assert result["id"] == "meta-llama/Llama-3.1-8B"
        assert result["source"] == "huggingface"
        model_reg.get.assert_awaited_once_with("meta-llama/Llama-3.1-8B")

    async def test_model_info_not_found_returns_error(self) -> None:
        from gpumod.mcp_tools import model_info

        model_reg = AsyncMock()
        model_reg.get.return_value = None
        ctx = _make_mock_ctx(model_registry=model_reg)

        result = await model_info(model_id="nonexistent-model", ctx=ctx)

        assert result["code"] == "NOT_FOUND"
        assert "error" in result

    async def test_simulate_mode_returns_result(self) -> None:
        from gpumod.mcp_tools import simulate_mode

        sim_result = _make_simulation_result()
        simulation = AsyncMock()
        simulation.simulate_mode.return_value = sim_result
        ctx = _make_mock_ctx(simulation=simulation)

        result = await simulate_mode(mode_id="chat", ctx=ctx)

        assert result["fits"] is True
        assert result["gpu_total_mb"] == 24576
        simulation.simulate_mode.assert_awaited_once()

    async def test_simulate_mode_with_add_remove(self) -> None:
        from gpumod.mcp_tools import simulate_mode

        sim_result = _make_simulation_result()
        simulation = AsyncMock()
        simulation.simulate_mode.return_value = sim_result
        ctx = _make_mock_ctx(simulation=simulation)

        result = await simulate_mode(
            mode_id="chat",
            ctx=ctx,
            add_services=["fastapi-web"],
            remove_services=["vllm-chat"],
            context_overrides={"fastapi-web": 2048},
        )

        assert result["fits"] is True
        call_kwargs = simulation.simulate_mode.call_args
        assert call_kwargs.kwargs["add"] == ["fastapi-web"]
        assert call_kwargs.kwargs["remove"] == ["vllm-chat"]
        assert call_kwargs.kwargs["context_overrides"] == {"fastapi-web": 2048}

    async def test_simulate_mode_not_found_returns_error(self) -> None:
        from gpumod.mcp_tools import simulate_mode

        simulation = AsyncMock()
        simulation.simulate_mode.side_effect = ValueError("Mode not found: 'nonexistent'")
        ctx = _make_mock_ctx(simulation=simulation)

        result = await simulate_mode(mode_id="nonexistent", ctx=ctx)

        assert result["code"] == "NOT_FOUND"
        assert "error" in result


# ---------------------------------------------------------------------------
# TestMutatingTools
# ---------------------------------------------------------------------------


class TestMutatingTools:
    """Tests for the 3 mutating MCP tools."""

    async def test_switch_mode_calls_manager(self) -> None:
        from gpumod.mcp_tools import switch_mode

        mode_result = _make_mode_result()
        manager = AsyncMock()
        manager.switch_mode.return_value = mode_result
        ctx = _make_mock_ctx(manager=manager)

        await switch_mode(mode_id="chat", ctx=ctx)

        manager.switch_mode.assert_awaited_once_with("chat")

    async def test_switch_mode_returns_mode_result(self) -> None:
        from gpumod.mcp_tools import switch_mode

        mode_result = _make_mode_result(started=["vllm-chat"], stopped=["code-svc"])
        manager = AsyncMock()
        manager.switch_mode.return_value = mode_result
        ctx = _make_mock_ctx(manager=manager)

        result = await switch_mode(mode_id="chat", ctx=ctx)

        assert result["success"] is True
        assert result["mode_id"] == "chat"
        assert "vllm-chat" in result["started"]
        assert "code-svc" in result["stopped"]

    async def test_start_service_calls_lifecycle(self) -> None:
        from gpumod.mcp_tools import start_service

        manager = AsyncMock()
        manager.start_service.return_value = None
        ctx = _make_mock_ctx(manager=manager)

        result = await start_service(service_id="vllm-chat", ctx=ctx)

        manager.start_service.assert_awaited_once_with("vllm-chat")
        assert result["success"] is True
        assert result["service_id"] == "vllm-chat"

    async def test_stop_service_calls_lifecycle(self) -> None:
        from gpumod.mcp_tools import stop_service

        manager = AsyncMock()
        manager.stop_service.return_value = None
        ctx = _make_mock_ctx(manager=manager)

        result = await stop_service(service_id="vllm-chat", ctx=ctx)

        manager.stop_service.assert_awaited_once_with("vllm-chat")
        assert result["success"] is True
        assert result["service_id"] == "vllm-chat"


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input validation on MCP tools (SEC-V1)."""

    async def test_rejects_invalid_service_id(self) -> None:
        from gpumod.mcp_tools import service_info

        ctx = _make_mock_ctx()

        result = await service_info(service_id="; rm -rf /", ctx=ctx)

        assert result["code"] == "VALIDATION_ERROR"
        assert "Invalid" in result["error"]

    async def test_rejects_invalid_mode_id(self) -> None:
        from gpumod.mcp_tools import switch_mode

        ctx = _make_mock_ctx()

        result = await switch_mode(mode_id="../etc/passwd", ctx=ctx)

        assert result["code"] == "VALIDATION_ERROR"
        assert "Invalid" in result["error"]

    async def test_rejects_invalid_model_id(self) -> None:
        from gpumod.mcp_tools import model_info

        ctx = _make_mock_ctx()

        result = await model_info(model_id="../../passwd", ctx=ctx)

        assert result["code"] == "VALIDATION_ERROR"
        assert "Invalid" in result["error"]

    async def test_rejects_negative_context_override(self) -> None:
        from gpumod.mcp_tools import simulate_mode

        ctx = _make_mock_ctx()

        result = await simulate_mode(
            mode_id="chat",
            ctx=ctx,
            context_overrides={"vllm-chat": 0},
        )

        assert result["code"] == "VALIDATION_ERROR"
        assert "Invalid" in result["error"]

    async def test_rejects_oversized_context_override(self) -> None:
        from gpumod.mcp_tools import simulate_mode

        ctx = _make_mock_ctx()

        result = await simulate_mode(
            mode_id="chat",
            ctx=ctx,
            context_overrides={"vllm-chat": 200000},
        )

        assert result["code"] == "VALIDATION_ERROR"
        assert "Invalid" in result["error"]


# ---------------------------------------------------------------------------
# TestOutputSanitization
# ---------------------------------------------------------------------------


class TestInputValidationExtended:
    """Additional input validation tests for full coverage."""

    async def test_rejects_invalid_add_service_in_simulate(self) -> None:
        from gpumod.mcp_tools import simulate_mode

        ctx = _make_mock_ctx()

        result = await simulate_mode(
            mode_id="chat",
            ctx=ctx,
            add_services=["; rm -rf /"],
        )

        assert result["code"] == "VALIDATION_ERROR"
        assert "Invalid" in result["error"]

    async def test_rejects_invalid_remove_service_in_simulate(self) -> None:
        from gpumod.mcp_tools import simulate_mode

        ctx = _make_mock_ctx()

        result = await simulate_mode(
            mode_id="chat",
            ctx=ctx,
            remove_services=["{{injection}}"],
        )

        assert result["code"] == "VALIDATION_ERROR"
        assert "Invalid" in result["error"]

    async def test_rejects_invalid_context_override_key(self) -> None:
        from gpumod.mcp_tools import simulate_mode

        ctx = _make_mock_ctx()

        result = await simulate_mode(
            mode_id="chat",
            ctx=ctx,
            context_overrides={"; evil": 4096},
        )

        assert result["code"] == "VALIDATION_ERROR"
        assert "Invalid" in result["error"]

    async def test_rejects_invalid_start_service_id(self) -> None:
        from gpumod.mcp_tools import start_service

        ctx = _make_mock_ctx()

        result = await start_service(service_id="'; DROP TABLE", ctx=ctx)

        assert result["code"] == "VALIDATION_ERROR"
        assert "Invalid" in result["error"]

    async def test_rejects_invalid_stop_service_id(self) -> None:
        from gpumod.mcp_tools import stop_service

        ctx = _make_mock_ctx()

        result = await stop_service(service_id="../etc/passwd", ctx=ctx)

        assert result["code"] == "VALIDATION_ERROR"
        assert "Invalid" in result["error"]


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling paths in MCP tools."""

    async def test_simulate_mode_simulation_error(self) -> None:
        from gpumod.mcp_tools import simulate_mode
        from gpumod.simulation import SimulationError

        simulation = AsyncMock()
        simulation.simulate_mode.side_effect = SimulationError("GPU info unavailable")
        ctx = _make_mock_ctx(simulation=simulation)

        result = await simulate_mode(mode_id="chat", ctx=ctx)

        assert result["code"] == "SIMULATION_ERROR"
        assert "GPU info unavailable" in result["error"]

    async def test_switch_mode_value_error(self) -> None:
        from gpumod.mcp_tools import switch_mode

        manager = AsyncMock()
        manager.switch_mode.side_effect = ValueError("Mode not found")
        ctx = _make_mock_ctx(manager=manager)

        result = await switch_mode(mode_id="nonexistent", ctx=ctx)

        assert result["code"] == "NOT_FOUND"
        assert "Mode not found" in result["error"]

    async def test_start_service_value_error(self) -> None:
        from gpumod.mcp_tools import start_service

        manager = AsyncMock()
        manager.start_service.side_effect = ValueError("Service not found")
        ctx = _make_mock_ctx(manager=manager)

        result = await start_service(service_id="nonexistent", ctx=ctx)

        assert result["code"] == "NOT_FOUND"
        assert "Service not found" in result["error"]

    async def test_stop_service_value_error(self) -> None:
        from gpumod.mcp_tools import stop_service

        manager = AsyncMock()
        manager.stop_service.side_effect = ValueError("Service not found")
        ctx = _make_mock_ctx(manager=manager)

        result = await stop_service(service_id="nonexistent", ctx=ctx)

        assert result["code"] == "NOT_FOUND"
        assert "Service not found" in result["error"]

    async def test_lifespan_not_available_raises_runtime_error(self) -> None:
        import pytest

        from gpumod.mcp_tools import gpu_status

        ctx = MagicMock()
        ctx.fastmcp._lifespan_result = None

        with pytest.raises(RuntimeError, match="Lifespan context not available"):
            await gpu_status(ctx=ctx)


# ---------------------------------------------------------------------------
# TestRegistration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Tests for register_tools registering all 9 tools."""

    def test_register_tools_adds_all_tools(self) -> None:
        from gpumod.mcp_tools import register_tools

        server = MagicMock()
        # server.tool(...)(fn) pattern: server.tool returns a callable
        mock_decorator = MagicMock()
        server.tool.return_value = mock_decorator

        register_tools(server)

        # Should have been called 14 times (6 read-only + 3 mutating + 5 discovery)
        assert server.tool.call_count == 14

        # Verify all tool names were registered
        tool_names = {call.kwargs["name"] for call in server.tool.call_args_list}
        expected_names = {
            "gpu_status",
            "list_services",
            "list_modes",
            "service_info",
            "model_info",
            "simulate_mode",
            "switch_mode",
            "start_service",
            "stop_service",
            # Discovery tools
            "search_hf_models",
            "list_gguf_files",
            "list_model_files",
            "fetch_model_config",
            "generate_preset",
        }
        assert tool_names == expected_names

    def test_register_tools_mutating_tools_have_mutating_description(self) -> None:
        from gpumod.mcp_tools import register_tools

        server = MagicMock()
        mock_decorator = MagicMock()
        server.tool.return_value = mock_decorator

        register_tools(server)

        mutating_descs = []
        for call in server.tool.call_args_list:
            name = call.kwargs["name"]
            desc = call.kwargs["description"]
            if name in {"switch_mode", "start_service", "stop_service"}:
                mutating_descs.append(desc)

        for desc in mutating_descs:
            assert "[MUTATING]" in desc


class TestOutputSanitization:
    """Tests for output sanitization — no internal paths, JSON-serializable."""

    async def test_error_response_no_internal_paths(self) -> None:
        from gpumod.mcp_tools import service_info

        db = AsyncMock()
        db.get_service.return_value = None
        ctx = _make_mock_ctx(db=db)

        result = await service_info(service_id="nonexistent", ctx=ctx)

        # Error dict must not contain file system paths
        result_str = json.dumps(result)
        assert "/home/" not in result_str
        assert "/tmp/" not in result_str
        assert ".py" not in result_str
        assert ".db" not in result_str

    async def test_success_response_is_json_serializable(self) -> None:
        from gpumod.mcp_tools import gpu_status

        status = _make_system_status()
        manager = AsyncMock()
        manager.get_status.return_value = status
        ctx = _make_mock_ctx(manager=manager)

        result = await gpu_status(ctx=ctx)

        # Must be JSON serializable without error
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
        # Round-trip must produce equivalent dict
        deserialized = json.loads(serialized)
        assert deserialized == result


# ---------------------------------------------------------------------------
# TestOutputNameSanitization (SEC-E3)
# ---------------------------------------------------------------------------


class TestOutputNameSanitization:
    """Tests that service/mode/model names with ANSI or Rich markup are stripped in tool output."""

    async def test_list_services_strips_ansi_from_names(self) -> None:
        from gpumod.mcp_tools import list_services

        svc = _make_service(name="\x1b[31mEvil Service\x1b[0m")
        db = AsyncMock()
        db.list_services.return_value = [svc]
        ctx = _make_mock_ctx(db=db)

        result = await list_services(ctx=ctx)

        for s in result["services"]:
            assert "\x1b[" not in s["name"]
            assert "Evil Service" in s["name"]

    async def test_list_modes_strips_rich_markup_from_names(self) -> None:
        from gpumod.mcp_tools import list_modes

        mode = _make_mode(name="[bold red]Malicious Mode[/bold red]")
        db = AsyncMock()
        db.list_modes.return_value = [mode]
        ctx = _make_mock_ctx(db=db)

        result = await list_modes(ctx=ctx)

        for m in result["modes"]:
            assert "[bold" not in m["name"]
            assert "Malicious Mode" in m["name"]

    async def test_service_info_strips_ansi_from_name(self) -> None:
        from gpumod.mcp_tools import service_info

        svc = _make_service(name="\x1b[32mGreen\x1b[0m Service")
        db = AsyncMock()
        db.get_service.return_value = svc
        ctx = _make_mock_ctx(db=db)

        result = await service_info(service_id="vllm-chat", ctx=ctx)

        assert "\x1b[" not in result["name"]
        assert "Green Service" in result["name"]

    async def test_gpu_status_strips_ansi_from_service_names(self) -> None:
        from gpumod.mcp_tools import gpu_status
        from gpumod.models import ServiceInfo, ServiceState, ServiceStatus

        svc = _make_service(name="\x1b[31mBad\x1b[0m")
        status_obj = _make_system_status(
            services=[
                ServiceInfo(
                    service=svc,
                    status=ServiceStatus(state=ServiceState.RUNNING, vram_mb=4000),
                )
            ]
        )
        manager = AsyncMock()
        manager.get_status.return_value = status_obj
        ctx = _make_mock_ctx(manager=manager)

        result = await gpu_status(ctx=ctx)

        serialized = json.dumps(result)
        assert "\x1b[" not in serialized
