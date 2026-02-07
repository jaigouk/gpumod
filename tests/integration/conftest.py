"""Shared fixtures for integration tests.

Provides a populated in-memory Database, a mocked VRAMTracker
returning a 24 GB GPU, and a real ModelRegistry backed by the
same database.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from gpumod.db import Database
from gpumod.models import (
    DriverType,
    GPUInfo,
    Mode,
    ModelInfo,
    ModelSource,
    Service,
    SleepMode,
)
from gpumod.registry import ModelRegistry
from gpumod.services.vram import VRAMTracker

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


@pytest.fixture
async def populated_db(tmp_path: Path) -> AsyncGenerator[Database, None]:
    """Database with realistic test data for integration tests."""
    db = Database(tmp_path / "test.db")
    await db.connect()

    # Services
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

    # Modes
    await db.insert_mode(Mode(id="chat", name="Chat Mode", description="Chat services"))
    await db.set_mode_services("chat", ["vllm-chat", "fastapi-app"])

    await db.insert_mode(Mode(id="code", name="Code Mode", description="Code services"))
    await db.set_mode_services("code", ["llama-code", "fastapi-app"])

    # Models
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

    yield db
    await db.close()


@pytest.fixture
def mock_vram_24gb() -> AsyncMock:
    """VRAMTracker mock returning a 24 GB GPU."""
    vram = AsyncMock(spec=VRAMTracker)
    vram.get_gpu_info.return_value = GPUInfo(
        name="RTX 4090",
        vram_total_mb=24576,
        driver="535.129.03",
    )
    vram.estimate_service_vram.side_effect = lambda svc: svc.vram_mb
    return vram


@pytest.fixture
def model_registry(populated_db: Database) -> ModelRegistry:
    """Real ModelRegistry backed by the populated database."""
    return ModelRegistry(populated_db)
