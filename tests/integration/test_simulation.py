"""Integration tests for SimulationEngine with a real DB and mocked VRAMTracker.

These tests exercise the SimulationEngine end-to-end: lookups go
through the real SQLite Database, VRAM estimation goes through the
real ModelRegistry, and only the GPU hardware layer (VRAMTracker)
is mocked to avoid requiring an actual NVIDIA GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpumod.simulation import SimulationEngine

if TYPE_CHECKING:
    from unittest.mock import AsyncMock

    from gpumod.db import Database
    from gpumod.registry import ModelRegistry


class TestSimulateModeIntegration:
    """Simulate modes with a real database backend."""

    async def test_simulate_chat_mode_fits_24gb(
        self,
        populated_db: Database,
        mock_vram_24gb: AsyncMock,
        model_registry: ModelRegistry,
    ) -> None:
        """Chat mode (8000 + 1000 = 9000 MB) fits in 24576 MB GPU."""
        engine = SimulationEngine(
            db=populated_db, vram=mock_vram_24gb, model_registry=model_registry
        )

        result = await engine.simulate_mode("chat")

        assert result.fits is True
        assert result.proposed_usage_mb == 9000
        assert result.gpu_total_mb == 24576
        assert result.headroom_mb == 24576 - 9000
        assert result.alternatives == []
        # Verify the correct services were included
        service_ids = {s.id for s in result.services}
        assert service_ids == {"vllm-chat", "fastapi-app"}

    async def test_simulate_code_mode_fits_24gb(
        self,
        populated_db: Database,
        mock_vram_24gb: AsyncMock,
        model_registry: ModelRegistry,
    ) -> None:
        """Code mode (12000 + 1000 = 13000 MB) fits in 24576 MB GPU."""
        engine = SimulationEngine(
            db=populated_db, vram=mock_vram_24gb, model_registry=model_registry
        )

        result = await engine.simulate_mode("code")

        assert result.fits is True
        assert result.proposed_usage_mb == 13000
        assert result.headroom_mb == 24576 - 13000
        assert result.alternatives == []
        service_ids = {s.id for s in result.services}
        assert service_ids == {"llama-code", "fastapi-app"}

    async def test_simulate_all_services_exceeds(
        self,
        populated_db: Database,
        mock_vram_24gb: AsyncMock,
        model_registry: ModelRegistry,
    ) -> None:
        """All 4 services (8000+4000+12000+1000=25000) exceeds 24576 MB GPU.

        Alternatives should be generated when the total VRAM exceeds capacity.
        """
        engine = SimulationEngine(
            db=populated_db, vram=mock_vram_24gb, model_registry=model_registry
        )

        result = await engine.simulate_services(
            ["vllm-chat", "vllm-embed", "llama-code", "fastapi-app"]
        )

        assert result.fits is False
        assert result.proposed_usage_mb == 25000
        assert result.headroom_mb == 24576 - 25000  # negative
        assert len(result.alternatives) > 0

    async def test_simulate_add_service_to_mode(
        self,
        populated_db: Database,
        mock_vram_24gb: AsyncMock,
        model_registry: ModelRegistry,
    ) -> None:
        """Adding llama-code to chat mode: 8000+1000+12000=21000, still fits."""
        engine = SimulationEngine(
            db=populated_db, vram=mock_vram_24gb, model_registry=model_registry
        )

        result = await engine.simulate_mode("chat", add=["llama-code"])

        assert result.fits is True
        assert result.proposed_usage_mb == 21000
        service_ids = {s.id for s in result.services}
        assert service_ids == {"vllm-chat", "fastapi-app", "llama-code"}

    async def test_simulate_context_override_changes_vram(
        self,
        populated_db: Database,
        mock_vram_24gb: AsyncMock,
        model_registry: ModelRegistry,
    ) -> None:
        """Reducing context via override lowers VRAM for the affected service.

        vllm-chat has model_id 'meta-llama/Llama-3.1-8B' which has:
        - base_vram_mb=7000, kv_cache_per_1k_tokens_mb=100
        With context_override of 2048 tokens:
        - estimated = 7000 + (2048/1000)*100 = 7000 + 204 = 7204 MB
        Without override, vllm-chat uses its stored vram_mb=8000.
        """
        engine = SimulationEngine(
            db=populated_db, vram=mock_vram_24gb, model_registry=model_registry
        )

        # Baseline: no overrides
        baseline = await engine.simulate_mode("chat")
        assert baseline.proposed_usage_mb == 9000  # 8000 + 1000

        # With context override: vllm-chat gets re-estimated
        result = await engine.simulate_mode("chat", context_overrides={"vllm-chat": 2048})

        # 7000 + int(2048/1000 * 100) = 7000 + 204 = 7204
        # fastapi-app = 1000 (no model_id, no override)
        assert result.proposed_usage_mb == 7204 + 1000
        assert result.proposed_usage_mb < baseline.proposed_usage_mb
        assert result.fits is True


class TestAlternativesIntegration:
    """Verify alternative generation with real DB data."""

    async def test_alternatives_include_sleep_strategy(
        self,
        populated_db: Database,
        mock_vram_24gb: AsyncMock,
        model_registry: ModelRegistry,
    ) -> None:
        """When VRAM exceeds capacity, sleepable services appear in alternatives.

        vllm-chat has sleep_mode=L1, so a sleep alternative should be generated.
        """
        engine = SimulationEngine(
            db=populated_db, vram=mock_vram_24gb, model_registry=model_registry
        )

        # Force all services to exceed GPU capacity
        result = await engine.simulate_services(
            ["vllm-chat", "vllm-embed", "llama-code", "fastapi-app"]
        )

        assert result.fits is False
        sleep_alts = [a for a in result.alternatives if a.strategy == "sleep"]
        assert len(sleep_alts) >= 1
        # vllm-chat is the only sleepable service (SleepMode.L1)
        sleep_service_ids = {sid for alt in sleep_alts for sid in alt.affected_services}
        assert "vllm-chat" in sleep_service_ids

    async def test_alternatives_sorted_by_savings(
        self,
        populated_db: Database,
        mock_vram_24gb: AsyncMock,
        model_registry: ModelRegistry,
    ) -> None:
        """Alternatives should be sorted by vram_saved_mb in descending order."""
        engine = SimulationEngine(
            db=populated_db, vram=mock_vram_24gb, model_registry=model_registry
        )

        result = await engine.simulate_services(
            ["vllm-chat", "vllm-embed", "llama-code", "fastapi-app"]
        )

        assert result.fits is False
        assert len(result.alternatives) > 1
        saved_values = [a.vram_saved_mb for a in result.alternatives]
        assert saved_values == sorted(saved_values, reverse=True)
