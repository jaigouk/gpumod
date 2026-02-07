"""E2E test fixtures — real DB, GPU/Docker detection, cleanup.

Fixtures detect hardware availability and skip tests gracefully
on CPU-only CI machines. No orphaned processes, temp files cleaned.

CI configuration:
  CPU-only CI:  pytest tests/ -m "not gpu_required and not docker_required"
  GPU CI:       pytest tests/ -m ""  (all tests)
  Docker CI:    pytest tests/ -m "not gpu_required"
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from gpumod.db import Database
from gpumod.models import (
    DriverType,
    Mode,
    Service,
    SleepMode,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


def _has_nvidia_gpu() -> bool:
    """Detect whether nvidia-smi is available and reports a GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_docker() -> bool:
    """Detect whether Docker daemon is available."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Cache detection results at module load (run once per test session)
_GPU_AVAILABLE = _has_nvidia_gpu()
_DOCKER_AVAILABLE = _has_docker()


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu_required: requires NVIDIA GPU")
    config.addinivalue_line("markers", "docker_required: requires Docker daemon")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip tests based on hardware availability."""
    skip_gpu = pytest.mark.skip(reason="No NVIDIA GPU detected (nvidia-smi not available)")
    skip_docker = pytest.mark.skip(reason="Docker daemon not available")

    for item in items:
        if "gpu_required" in item.keywords and not _GPU_AVAILABLE:
            item.add_marker(skip_gpu)
        if "docker_required" in item.keywords and not _DOCKER_AVAILABLE:
            item.add_marker(skip_docker)


# ---------------------------------------------------------------------------
# E2E database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def e2e_db(tmp_path: Path) -> AsyncGenerator[Database, None]:
    """Real SQLite database for E2E tests, cleaned up after use."""
    db_path = tmp_path / "e2e_test.db"
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
            unit_name="gpumod-vllm-chat.service",
        )
    )
    await db.insert_service(
        Service(
            id="vllm-embed",
            name="vLLM Embed",
            driver=DriverType.VLLM,
            port=8001,
            vram_mb=4000,
            unit_name="gpumod-vllm-embed.service",
        )
    )
    await db.insert_service(
        Service(
            id="fastapi-app",
            name="FastAPI App",
            driver=DriverType.FASTAPI,
            port=9000,
            vram_mb=1000,
            unit_name="gpumod-fastapi.service",
        )
    )

    await db.insert_mode(Mode(id="chat", name="Chat Mode", description="Chat services"))
    await db.set_mode_services("chat", ["vllm-chat", "fastapi-app"])

    await db.insert_mode(Mode(id="embed", name="Embed Mode", description="Embed services"))
    await db.set_mode_services("embed", ["vllm-embed", "fastapi-app"])

    yield db
    await db.close()
