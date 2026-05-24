"""Tests for gpumod-lgt running-services context helper.

When preflight refuses to start a service, the operator needs to know
which OTHER gpumod services are currently running so they can decide
what to stop. This helper formats that context block.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _svc(id_: str, vram_mb: int, model_path: str | None = None) -> MagicMock:
    s = MagicMock()
    s.id = id_
    s.vram_mb = vram_mb
    s.extra_config = {"unit_vars": {"model_path": model_path}} if model_path else {}
    return s


@pytest.mark.asyncio
async def test_empty_when_no_running_services() -> None:
    from gpumod.services.running_context import format_running_services

    registry = MagicMock()
    registry.list_running = AsyncMock(return_value=[])

    text = await format_running_services(registry, exclude_id=None)
    # Empty list -> distinct sentinel so callers can branch on it
    assert text == ""


@pytest.mark.asyncio
async def test_lists_running_services_with_vram() -> None:
    from gpumod.services.running_context import format_running_services

    registry = MagicMock()
    registry.list_running = AsyncMock(
        return_value=[
            _svc("vllm-embedding-code", vram_mb=2500),
            _svc("qwen36-35b-a3b-iq4xs", vram_mb=22000),
        ]
    )

    text = await format_running_services(registry, exclude_id=None)
    assert "vllm-embedding-code" in text
    assert "qwen36-35b-a3b-iq4xs" in text
    assert "2500" in text
    assert "22000" in text


@pytest.mark.asyncio
async def test_excludes_specified_id() -> None:
    """The service being attempted shouldn't appear in the running list."""
    from gpumod.services.running_context import format_running_services

    registry = MagicMock()
    registry.list_running = AsyncMock(
        return_value=[
            _svc("qwen36-27b-mtp-q4", vram_mb=19000),
            _svc("vllm-embedding-code", vram_mb=2500),
        ]
    )

    text = await format_running_services(registry, exclude_id="qwen36-27b-mtp-q4")
    assert "vllm-embedding-code" in text
    assert "qwen36-27b-mtp-q4" not in text


@pytest.mark.asyncio
async def test_returns_empty_when_only_excluded_service_is_running() -> None:
    from gpumod.services.running_context import format_running_services

    registry = MagicMock()
    registry.list_running = AsyncMock(return_value=[_svc("self", vram_mb=20000)])

    text = await format_running_services(registry, exclude_id="self")
    assert text == ""


@pytest.mark.asyncio
async def test_suggests_stop_commands_for_heaviest_first() -> None:
    """Heaviest VRAM consumer suggested first — gives the operator the
    biggest single-stop win."""
    from gpumod.services.running_context import format_running_services

    registry = MagicMock()
    registry.list_running = AsyncMock(
        return_value=[
            _svc("small", vram_mb=2500),
            _svc("medium", vram_mb=10000),
            _svc("huge", vram_mb=22000),
        ]
    )

    text = await format_running_services(registry, exclude_id=None)
    # Each running service should produce a stop hint
    assert "gpumod service stop huge" in text
    assert "gpumod service stop medium" in text
    assert "gpumod service stop small" in text
    # Heaviest must appear before lighter ones (ordering signal)
    assert text.index("gpumod service stop huge") < text.index("gpumod service stop small")


@pytest.mark.asyncio
async def test_graceful_when_registry_raises() -> None:
    """If we can't enumerate running services, return empty string —
    don't make the original error message disappear behind a follow-on."""
    from gpumod.services.running_context import format_running_services

    registry = MagicMock()
    registry.list_running = AsyncMock(side_effect=RuntimeError("boom"))

    text = await format_running_services(registry, exclude_id=None)
    assert text == ""
