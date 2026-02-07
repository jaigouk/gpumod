"""Integration tests for the MCP server running end-to-end.

Uses FastMCP's in-process :class:`Client` to talk to a fully
configured MCP server backed by a pre-populated SQLite database.
GPU hardware (nvidia-smi) is *not* mocked -- tests that depend on
GPU info verify the graceful error path instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastmcp import Client

from gpumod.db import Database
from gpumod.mcp_server import create_mcp_server
from gpumod.models import (
    DriverType,
    Mode,
    ModelInfo,
    ModelSource,
    Service,
    SleepMode,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def mcp_client(tmp_path: Path) -> Client:
    """Create an MCP Client backed by a pre-populated database.

    The database is written, closed, then re-opened by the MCP server
    lifespan so the server owns its own connection.
    """
    db_path = tmp_path / "mcp_test.db"

    # Pre-populate the database
    db = Database(db_path)
    await db.connect()

    await db.insert_service(
        Service(
            id="vllm-chat",
            name="vLLM Chat",
            driver=DriverType.VLLM,
            port=8000,
            vram_mb=8000,
            sleep_mode=SleepMode.L1,
            model_id="meta-llama/Llama-3.1-8B",
        )
    )
    await db.insert_service(
        Service(
            id="vllm-embed",
            name="vLLM Embed",
            driver=DriverType.VLLM,
            port=8001,
            vram_mb=4000,
        )
    )
    await db.insert_service(
        Service(
            id="llama-code",
            name="Llama Code",
            driver=DriverType.LLAMACPP,
            port=8002,
            vram_mb=12000,
            model_id="codellama/CodeLlama-34B",
        )
    )
    await db.insert_service(
        Service(
            id="fastapi-app",
            name="FastAPI App",
            driver=DriverType.FASTAPI,
            port=9000,
            vram_mb=1000,
        )
    )

    await db.insert_mode(Mode(id="chat", name="Chat Mode", description="Chat services"))
    await db.set_mode_services("chat", ["vllm-chat", "fastapi-app"])

    await db.insert_mode(Mode(id="code", name="Code Mode", description="Code services"))
    await db.set_mode_services("code", ["llama-code", "fastapi-app"])

    await db.insert_model(
        ModelInfo(
            id="meta-llama/Llama-3.1-8B",
            source=ModelSource.HUGGINGFACE,
            parameters_b=8.0,
            base_vram_mb=7000,
            kv_cache_per_1k_tokens_mb=100,
        )
    )
    await db.insert_model(
        ModelInfo(
            id="codellama/CodeLlama-34B",
            source=ModelSource.HUGGINGFACE,
            parameters_b=34.0,
            base_vram_mb=11000,
            kv_cache_per_1k_tokens_mb=200,
        )
    )

    await db.close()

    # Create the MCP server pointing to the pre-populated DB
    server = create_mcp_server(db_path=db_path)
    return Client(server)


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


class TestMCPTools:
    """Integration tests for MCP tool calls via the Client."""

    async def test_mcp_gpu_status_returns_valid_json(
        self,
        mcp_client: Client,
    ) -> None:
        """gpu_status should return a valid JSON dict with expected keys.

        On CI without nvidia-smi, gpu and vram will be None, but the
        response structure must still be valid.
        """
        async with mcp_client as client:
            result = await client.call_tool("gpu_status", {})

        data = result.data
        assert isinstance(data, dict)
        assert "gpu" in data
        assert "vram" in data
        assert "current_mode" in data
        assert "services" in data
        assert isinstance(data["services"], list)

    async def test_mcp_list_services_returns_all(
        self,
        mcp_client: Client,
    ) -> None:
        """list_services should return all 4 pre-populated services."""
        async with mcp_client as client:
            result = await client.call_tool("list_services", {})

        data = result.data
        assert isinstance(data, dict)
        assert "services" in data
        services = data["services"]
        assert len(services) == 4
        service_ids = {s["id"] for s in services}
        assert service_ids == {"vllm-chat", "vllm-embed", "llama-code", "fastapi-app"}

    async def test_mcp_list_modes_returns_all(
        self,
        mcp_client: Client,
    ) -> None:
        """list_modes should return both pre-populated modes."""
        async with mcp_client as client:
            result = await client.call_tool("list_modes", {})

        data = result.data
        assert isinstance(data, dict)
        assert "modes" in data
        modes = data["modes"]
        assert len(modes) == 2
        mode_ids = {m["id"] for m in modes}
        assert mode_ids == {"chat", "code"}

    async def test_mcp_simulate_mode_returns_result(
        self,
        mcp_client: Client,
    ) -> None:
        """simulate_mode returns a result dict (may be an error on CI
        without nvidia-smi, but must have the expected structure).
        """
        async with mcp_client as client:
            result = await client.call_tool("simulate_mode", {"mode_id": "chat"})

        data = result.data
        assert isinstance(data, dict)
        # Either a valid simulation result or a controlled error
        if "error" in data:
            # Error response from missing nvidia-smi
            assert "code" in data
            assert data["code"] == "SIMULATION_ERROR"
        else:
            # Full simulation result
            assert "fits" in data
            assert "gpu_total_mb" in data
            assert "proposed_usage_mb" in data
            assert "services" in data

    async def test_mcp_service_info_validates_input(
        self,
        mcp_client: Client,
    ) -> None:
        """Shell injection in service_id returns a VALIDATION_ERROR."""
        async with mcp_client as client:
            result = await client.call_tool("service_info", {"service_id": "; rm -rf /"})

        data = result.data
        assert isinstance(data, dict)
        assert data["code"] == "VALIDATION_ERROR"
        assert "Invalid service_id" in data["error"]

    async def test_mcp_invalid_tool_arg_rejected(
        self,
        mcp_client: Client,
    ) -> None:
        """Template injection in service_id returns a VALIDATION_ERROR."""
        async with mcp_client as client:
            result = await client.call_tool("service_info", {"service_id": "{{7*7}}"})

        data = result.data
        assert isinstance(data, dict)
        assert data["code"] == "VALIDATION_ERROR"
        assert "Invalid service_id" in data["error"]

    async def test_mcp_error_response_sanitized(
        self,
        mcp_client: Client,
    ) -> None:
        """Error responses must not leak internal file paths.

        The simulate_mode tool on a system without nvidia-smi returns
        a SIMULATION_ERROR. The message should not contain system paths.
        """
        async with mcp_client as client:
            result = await client.call_tool("simulate_mode", {"mode_id": "chat"})

        data = result.data
        if "error" in data:
            error_msg = data["error"]
            # The error should not contain home directory paths
            assert "/home/" not in error_msg
            assert "/tmp/" not in error_msg
            assert "/var/" not in error_msg

    async def test_mcp_model_info_returns_data(
        self,
        mcp_client: Client,
    ) -> None:
        """model_info with a valid model ID returns model metadata."""
        async with mcp_client as client:
            result = await client.call_tool("model_info", {"model_id": "meta-llama/Llama-3.1-8B"})

        data = result.data
        assert isinstance(data, dict)
        assert data["id"] == "meta-llama/Llama-3.1-8B"
        assert data["source"] == "huggingface"
        assert data["parameters_b"] == 8.0
        assert data["base_vram_mb"] == 7000


# ---------------------------------------------------------------------------
# Resource tests
# ---------------------------------------------------------------------------


class TestMCPResources:
    """Integration tests for MCP resource reads via the Client."""

    async def test_mcp_resource_help_returns_content(
        self,
        mcp_client: Client,
    ) -> None:
        """The help resource should return Markdown with key headings."""
        async with mcp_client as client:
            contents = await client.read_resource("gpumod://help")

        assert len(contents) >= 1
        text = contents[0].text
        assert "# gpumod" in text
        assert "GPU Service Manager" in text
        assert "## Resources" in text
        assert "## Tools" in text

    async def test_mcp_resource_modes_returns_markdown(
        self,
        mcp_client: Client,
    ) -> None:
        """The modes resource should list both modes in a Markdown table."""
        async with mcp_client as client:
            contents = await client.read_resource("gpumod://modes")

        assert len(contents) >= 1
        text = contents[0].text
        assert "# Modes" in text
        assert "chat" in text
        assert "code" in text
        # Should be a Markdown table
        assert "| ID |" in text

    async def test_mcp_resource_services_returns_markdown(
        self,
        mcp_client: Client,
    ) -> None:
        """The services resource should list all 4 services in a table."""
        async with mcp_client as client:
            contents = await client.read_resource("gpumod://services")

        assert len(contents) >= 1
        text = contents[0].text
        assert "# Services" in text
        assert "vllm-chat" in text
        assert "llama-code" in text
        assert "fastapi-app" in text
        # Verify it is a Markdown table with expected columns
        assert "| ID |" in text
        assert "Driver" in text

    async def test_mcp_resource_models_returns_markdown(
        self,
        mcp_client: Client,
    ) -> None:
        """The models resource should list both registered models."""
        async with mcp_client as client:
            contents = await client.read_resource("gpumod://models")

        assert len(contents) >= 1
        text = contents[0].text
        assert "# Models" in text
        assert "meta-llama/Llama-3.1-8B" in text
        assert "codellama/CodeLlama-34B" in text
