"""Tests for MCP resources — browsing modes, services, and models."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_db(
    *,
    modes: list[object] | None = None,
    services: list[object] | None = None,
    models: list[object] | None = None,
    mode_services: list[object] | None = None,
    mode: object | None = None,
    service: object | None = None,
    model: object | None = None,
) -> AsyncMock:
    """Create a mock Database with configurable return values."""
    db = AsyncMock()
    db.list_modes.return_value = modes or []
    db.list_services.return_value = services or []
    db.list_models.return_value = models or []
    db.get_mode_services.return_value = mode_services or []
    db.get_mode.return_value = mode
    db.get_service.return_value = service
    db.get_model.return_value = model
    db.get_setting.return_value = None
    return db


def _make_mock_ctx(db: AsyncMock) -> MagicMock:
    """Create a mock Context whose fastmcp._lifespan_result has the given db."""
    ctx = MagicMock()
    ctx.fastmcp._lifespan_result = {"db": db}
    return ctx


def _make_mode(
    *,
    mode_id: str = "chat",
    name: str = "Chat Mode",
    description: str | None = "Conversational AI",
    total_vram_mb: int | None = 8000,
) -> object:
    """Create a mock Mode object."""
    from gpumod.models import Mode

    return Mode(
        id=mode_id,
        name=name,
        description=description,
        total_vram_mb=total_vram_mb,
    )


def _make_service(
    *,
    service_id: str = "vllm-chat",
    name: str = "vLLM Chat",
    driver: str = "vllm",
    port: int = 8000,
    vram_mb: int = 4000,
) -> object:
    """Create a mock Service object."""
    from gpumod.models import DriverType, Service

    return Service(
        id=service_id,
        name=name,
        driver=DriverType(driver),
        port=port,
        vram_mb=vram_mb,
    )


def _make_model(
    *,
    model_id: str = "meta-llama/Llama-3.1-8B",
    source: str = "huggingface",
    parameters_b: float = 8.0,
    architecture: str | None = "llama",
    base_vram_mb: int | None = 16000,
    quantizations: list[str] | None = None,
    capabilities: list[str] | None = None,
) -> object:
    """Create a mock ModelInfo object."""
    from gpumod.models import ModelInfo, ModelSource

    return ModelInfo(
        id=model_id,
        source=ModelSource(source),
        parameters_b=parameters_b,
        architecture=architecture,
        base_vram_mb=base_vram_mb,
        quantizations=quantizations or ["fp16", "int8"],
        capabilities=capabilities or ["text-generation"],
    )


# ---------------------------------------------------------------------------
# TestStaticResources
# ---------------------------------------------------------------------------


class TestStaticResources:
    """Tests for static resources (help, config)."""

    def test_help_resource_returns_markdown(self) -> None:
        from gpumod.mcp_resources import help_resource

        result = help_resource()
        assert isinstance(result, str)
        assert "gpumod" in result.lower()
        # Should describe tools and resources
        assert "tool" in result.lower() or "resource" in result.lower()

    async def test_config_resource_returns_settings(self) -> None:
        from gpumod.mcp_resources import config_resource

        db = _make_mock_db()
        db.get_setting.side_effect = [None, None]  # current_mode, gpu_total_mb
        ctx = _make_mock_ctx(db)
        result = await config_resource(ctx=ctx)
        assert isinstance(result, str)
        # Should contain config information
        assert "config" in result.lower() or "setting" in result.lower()


# ---------------------------------------------------------------------------
# TestModeResources
# ---------------------------------------------------------------------------


class TestModeResources:
    """Tests for mode browsing resources."""

    async def test_modes_list_returns_markdown_table(self) -> None:
        from gpumod.mcp_resources import modes_resource

        mode = _make_mode()
        db = _make_mock_db(modes=[mode])
        ctx = _make_mock_ctx(db)
        result = await modes_resource(ctx=ctx)
        assert isinstance(result, str)
        assert "| " in result  # Markdown table
        assert "chat" in result.lower()

    async def test_modes_list_empty_shows_message(self) -> None:
        from gpumod.mcp_resources import modes_resource

        db = _make_mock_db(modes=[])
        ctx = _make_mock_ctx(db)
        result = await modes_resource(ctx=ctx)
        assert isinstance(result, str)
        assert "no modes" in result.lower()

    async def test_mode_detail_shows_services(self) -> None:
        from gpumod.mcp_resources import mode_detail_resource

        mode = _make_mode()
        svc = _make_service()
        db = _make_mock_db(mode=mode, mode_services=[svc])
        ctx = _make_mock_ctx(db)
        result = await mode_detail_resource(mode_id="chat", ctx=ctx)
        assert isinstance(result, str)
        assert "chat" in result.lower()
        assert "vllm" in result.lower()

    async def test_mode_detail_no_services(self) -> None:
        from gpumod.mcp_resources import mode_detail_resource

        mode = _make_mode()
        db = _make_mock_db(mode=mode, mode_services=[])
        ctx = _make_mock_ctx(db)
        result = await mode_detail_resource(mode_id="chat", ctx=ctx)
        assert isinstance(result, str)
        assert "no services" in result.lower()

    async def test_mode_detail_not_found(self) -> None:
        from gpumod.mcp_resources import mode_detail_resource

        db = _make_mock_db(mode=None)
        ctx = _make_mock_ctx(db)
        result = await mode_detail_resource(mode_id="nonexistent", ctx=ctx)
        assert isinstance(result, str)
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# TestServiceResources
# ---------------------------------------------------------------------------


class TestServiceResources:
    """Tests for service browsing resources."""

    async def test_services_list_returns_markdown_table(self) -> None:
        from gpumod.mcp_resources import services_resource

        svc = _make_service()
        db = _make_mock_db(services=[svc])
        ctx = _make_mock_ctx(db)
        result = await services_resource(ctx=ctx)
        assert isinstance(result, str)
        assert "| " in result
        assert "vllm" in result.lower()

    async def test_services_list_empty_shows_message(self) -> None:
        from gpumod.mcp_resources import services_resource

        db = _make_mock_db(services=[])
        ctx = _make_mock_ctx(db)
        result = await services_resource(ctx=ctx)
        assert isinstance(result, str)
        assert "no services" in result.lower()

    async def test_service_detail_shows_config(self) -> None:
        from gpumod.mcp_resources import service_detail_resource

        svc = _make_service()
        db = _make_mock_db(service=svc)
        ctx = _make_mock_ctx(db)
        result = await service_detail_resource(service_id="vllm-chat", ctx=ctx)
        assert isinstance(result, str)
        assert "vllm-chat" in result.lower()
        assert "4000" in result  # VRAM

    async def test_service_detail_shows_dependencies(self) -> None:
        from gpumod.mcp_resources import service_detail_resource
        from gpumod.models import DriverType, Service

        svc = Service(
            id="api-svc",
            name="API Service",
            driver=DriverType.FASTAPI,
            port=9000,
            vram_mb=2000,
            depends_on=["vllm-chat", "embed-svc"],
        )
        db = _make_mock_db(service=svc)
        ctx = _make_mock_ctx(db)
        result = await service_detail_resource(service_id="api-svc", ctx=ctx)
        assert "vllm-chat" in result
        assert "embed-svc" in result

    async def test_service_detail_not_found(self) -> None:
        from gpumod.mcp_resources import service_detail_resource

        db = _make_mock_db(service=None)
        ctx = _make_mock_ctx(db)
        result = await service_detail_resource(service_id="nonexistent", ctx=ctx)
        assert isinstance(result, str)
        assert "not found" in result.lower()

    async def test_service_detail_invalid_id_returns_not_found(self) -> None:
        from gpumod.mcp_resources import service_detail_resource

        db = _make_mock_db(service=None)
        ctx = _make_mock_ctx(db)
        result = await service_detail_resource(service_id="'; DROP TABLE--", ctx=ctx)
        assert isinstance(result, str)
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# TestModelResources
# ---------------------------------------------------------------------------


class TestModelResources:
    """Tests for model browsing resources."""

    async def test_models_list_returns_markdown_table(self) -> None:
        from gpumod.mcp_resources import models_resource

        model = _make_model()
        db = _make_mock_db(models=[model])
        ctx = _make_mock_ctx(db)
        result = await models_resource(ctx=ctx)
        assert isinstance(result, str)
        assert "| " in result
        assert "llama" in result.lower()

    async def test_models_list_empty_shows_message(self) -> None:
        from gpumod.mcp_resources import models_resource

        db = _make_mock_db(models=[])
        ctx = _make_mock_ctx(db)
        result = await models_resource(ctx=ctx)
        assert isinstance(result, str)
        assert "no models" in result.lower()

    async def test_model_detail_shows_vram_info(self) -> None:
        from gpumod.mcp_resources import model_detail_resource

        model = _make_model()
        db = _make_mock_db(model=model)
        ctx = _make_mock_ctx(db)
        result = await model_detail_resource(model_id="meta-llama/Llama-3.1-8B", ctx=ctx)
        assert isinstance(result, str)
        assert "llama" in result.lower()
        assert "16000" in result  # base_vram_mb

    async def test_model_detail_shows_notes(self) -> None:
        from gpumod.mcp_resources import model_detail_resource
        from gpumod.models import ModelInfo, ModelSource

        model = ModelInfo(
            id="test/model-7b",
            source=ModelSource.HUGGINGFACE,
            parameters_b=7.0,
            base_vram_mb=14000,
            notes="Optimized for chat",
        )
        db = _make_mock_db(model=model)
        ctx = _make_mock_ctx(db)
        result = await model_detail_resource(model_id="test/model-7b", ctx=ctx)
        assert "Optimized for chat" in result

    async def test_model_detail_not_found(self) -> None:
        from gpumod.mcp_resources import model_detail_resource

        db = _make_mock_db(model=None)
        ctx = _make_mock_ctx(db)
        result = await model_detail_resource(model_id="nonexistent-model", ctx=ctx)
        assert isinstance(result, str)
        assert "not found" in result.lower()

    async def test_model_detail_invalid_id_returns_not_found(self) -> None:
        from gpumod.mcp_resources import model_detail_resource

        db = _make_mock_db(model=None)
        ctx = _make_mock_ctx(db)
        result = await model_detail_resource(model_id="../../etc/passwd", ctx=ctx)
        assert isinstance(result, str)
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# TestResourceSecurity
# ---------------------------------------------------------------------------


class TestResourceRegistration:
    """Tests for resource registration on the FastMCP server."""

    def test_register_resources_adds_resources_to_server(self) -> None:
        from gpumod.mcp_server import create_mcp_server

        # create_mcp_server calls register_resources internally
        server = create_mcp_server()
        # Verify the server was created successfully with resources registered
        assert server is not None
        assert server.name == "gpumod"

    def test_register_resources_on_fresh_server(self) -> None:
        from fastmcp import FastMCP

        from gpumod.mcp_resources import register_resources

        server: FastMCP = FastMCP("test")  # type: ignore[type-arg]
        register_resources(server)
        # No error means all 8 resources registered successfully


class TestResourceSecurity:
    """Tests for resource security — SEC-V1 and SEC-E2."""

    async def test_invalid_id_returns_not_found(self) -> None:
        from gpumod.mcp_resources import mode_detail_resource

        db = _make_mock_db(mode=None)
        ctx = _make_mock_ctx(db)
        # Injection attempt should return not-found, never raise
        result = await mode_detail_resource(mode_id="'; DROP TABLE--", ctx=ctx)
        assert isinstance(result, str)
        assert "not found" in result.lower()

    async def test_no_internal_paths_in_output(self) -> None:
        from gpumod.mcp_resources import (
            config_resource,
            help_resource,
            modes_resource,
            services_resource,
        )

        mode = _make_mode()
        svc = _make_service()
        db = _make_mock_db(modes=[mode], services=[svc])
        db.get_setting.return_value = None
        ctx = _make_mock_ctx(db)

        outputs = [
            help_resource(),
            await config_resource(ctx=ctx),
            await modes_resource(ctx=ctx),
            await services_resource(ctx=ctx),
        ]
        for output in outputs:
            assert "/home/" not in output
            assert "/tmp/" not in output
            assert "/var/" not in output
