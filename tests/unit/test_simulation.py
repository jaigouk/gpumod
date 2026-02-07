"""Tests for the SimulationEngine, validation module, and simulation models."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gpumod.models import (
    DriverType,
    GPUInfo,
    Service,
    SimulationAlternative,
    SimulationResult,
    SleepMode,
)
from gpumod.simulation import SimulationEngine, SimulationError
from gpumod.validation import (
    validate_context_override,
    validate_mode_id,
    validate_service_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GPU_24GB = GPUInfo(name="RTX 4090", vram_total_mb=24576)


def _make_service(
    sid: str,
    vram: int,
    *,
    sleep_mode: SleepMode = SleepMode.NONE,
    model_id: str | None = None,
    driver: DriverType = DriverType.VLLM,
) -> Service:
    return Service(
        id=sid,
        name=sid,
        driver=driver,
        vram_mb=vram,
        sleep_mode=sleep_mode,
        model_id=model_id,
    )


def _make_engine(
    *,
    services: list[Service] | None = None,
    modes: dict[str, list[Service]] | None = None,
    gpu_info: GPUInfo = GPU_24GB,
    gpu_raises: bool = False,
    estimate_vram_fn: AsyncMock | None = None,
) -> tuple[SimulationEngine, MagicMock, MagicMock, MagicMock]:
    """Build a SimulationEngine with mocked dependencies."""
    db = MagicMock()
    vram = MagicMock()
    registry = MagicMock()

    services = services or []
    modes = modes or {}

    # db.get_service
    async def _get_service(sid: str) -> Service | None:
        for s in services:
            if s.id == sid:
                return s
        return None

    db.get_service = AsyncMock(side_effect=_get_service)
    db.list_services = AsyncMock(return_value=services)

    # db.get_mode / get_mode_services
    async def _get_mode(mid: str) -> MagicMock | None:
        if mid in modes:
            m = MagicMock()
            m.id = mid
            m.name = mid
            return m
        return None

    async def _get_mode_services(mid: str) -> list[Service]:
        return modes.get(mid, [])

    db.get_mode = AsyncMock(side_effect=_get_mode)
    db.get_mode_services = AsyncMock(side_effect=_get_mode_services)

    # vram.get_gpu_info
    if gpu_raises:
        vram.get_gpu_info = AsyncMock(side_effect=RuntimeError("nvidia-smi failed"))
    else:
        vram.get_gpu_info = AsyncMock(return_value=gpu_info)

    # model_registry.estimate_vram
    if estimate_vram_fn is not None:
        registry.estimate_vram = estimate_vram_fn
    else:
        registry.estimate_vram = AsyncMock(return_value=8000)

    engine = SimulationEngine(db=db, vram=vram, model_registry=registry)
    return engine, db, vram, registry


# ---------------------------------------------------------------------------
# SimulationResult model tests
# ---------------------------------------------------------------------------


class TestSimulationResultModel:
    """Tests for the SimulationResult Pydantic model."""

    def test_simulation_result_fits_true_has_empty_alternatives(self) -> None:
        result = SimulationResult(
            fits=True,
            gpu_total_mb=24576,
            current_usage_mb=0,
            proposed_usage_mb=16000,
            headroom_mb=8576,
            services=[],
            alternatives=[],
        )
        assert result.fits is True
        assert result.alternatives == []

    def test_simulation_result_fits_false_has_alternatives(self) -> None:
        alt = SimulationAlternative(
            id="alt-1",
            strategy="sleep",
            description="Sleep vllm-chat",
            affected_services=["vllm-chat"],
            vram_saved_mb=8000,
            projected_total_mb=20000,
            trade_offs=["Increased latency on wake"],
        )
        result = SimulationResult(
            fits=False,
            gpu_total_mb=24576,
            current_usage_mb=0,
            proposed_usage_mb=28000,
            headroom_mb=-3424,
            services=[],
            alternatives=[alt],
        )
        assert result.fits is False
        assert len(result.alternatives) == 1
        assert result.alternatives[0].strategy == "sleep"

    def test_simulation_result_headroom_calculation(self) -> None:
        result = SimulationResult(
            fits=True,
            gpu_total_mb=24576,
            current_usage_mb=0,
            proposed_usage_mb=20000,
            headroom_mb=4576,
            services=[],
            alternatives=[],
        )
        assert result.headroom_mb == result.gpu_total_mb - result.proposed_usage_mb


# ---------------------------------------------------------------------------
# simulate_mode tests
# ---------------------------------------------------------------------------


class TestSimulateMode:
    """Tests for SimulationEngine.simulate_mode."""

    async def test_simulate_mode_fits(self) -> None:
        svc_a = _make_service("svc-a", 8000)
        svc_b = _make_service("svc-b", 6000)
        engine, *_ = _make_engine(
            services=[svc_a, svc_b],
            modes={"dev": [svc_a, svc_b]},
        )

        result = await engine.simulate_mode("dev")

        assert result.fits is True
        assert result.proposed_usage_mb == 14000
        assert result.headroom_mb == 24576 - 14000
        assert result.alternatives == []

    async def test_simulate_mode_exceeds(self) -> None:
        svc_a = _make_service("svc-a", 16000, sleep_mode=SleepMode.L1)
        svc_b = _make_service("svc-b", 12000)
        engine, *_ = _make_engine(
            services=[svc_a, svc_b],
            modes={"heavy": [svc_a, svc_b]},
        )

        result = await engine.simulate_mode("heavy")

        assert result.fits is False
        assert result.proposed_usage_mb == 28000
        assert len(result.alternatives) > 0

    async def test_simulate_mode_not_found(self) -> None:
        engine, *_ = _make_engine()

        with pytest.raises(ValueError, match="not found"):
            await engine.simulate_mode("nonexistent")

    async def test_simulate_mode_with_add(self) -> None:
        svc_a = _make_service("svc-a", 8000)
        svc_b = _make_service("svc-b", 4000)
        engine, *_ = _make_engine(
            services=[svc_a, svc_b],
            modes={"base": [svc_a]},
        )

        result = await engine.simulate_mode("base", add=["svc-b"])

        assert result.proposed_usage_mb == 12000
        assert result.fits is True

    async def test_simulate_mode_with_remove(self) -> None:
        svc_a = _make_service("svc-a", 8000)
        svc_b = _make_service("svc-b", 6000)
        engine, *_ = _make_engine(
            services=[svc_a, svc_b],
            modes={"full": [svc_a, svc_b]},
        )

        result = await engine.simulate_mode("full", remove=["svc-b"])

        assert result.proposed_usage_mb == 8000
        assert result.fits is True

    async def test_simulate_mode_with_context_override(self) -> None:
        svc_a = _make_service("svc-a", 8000, model_id="meta-llama/Llama-3.1-8B")
        estimate_fn = AsyncMock(return_value=12000)
        engine, *_ = _make_engine(
            services=[svc_a],
            modes={"llm": [svc_a]},
            estimate_vram_fn=estimate_fn,
        )

        result = await engine.simulate_mode("llm", context_overrides={"svc-a": 32768})

        assert result.proposed_usage_mb == 12000
        estimate_fn.assert_awaited_once_with("meta-llama/Llama-3.1-8B", 32768)


# ---------------------------------------------------------------------------
# simulate_services tests
# ---------------------------------------------------------------------------


class TestSimulateServices:
    """Tests for SimulationEngine.simulate_services."""

    async def test_simulate_services_fits(self) -> None:
        svc_a = _make_service("svc-a", 8000)
        svc_b = _make_service("svc-b", 4000)
        engine, *_ = _make_engine(services=[svc_a, svc_b])

        result = await engine.simulate_services(["svc-a", "svc-b"])

        assert result.fits is True
        assert result.proposed_usage_mb == 12000

    async def test_simulate_services_unknown_service(self) -> None:
        engine, *_ = _make_engine()

        with pytest.raises(ValueError, match="not found"):
            await engine.simulate_services(["unknown-svc"])

    async def test_empty_service_list_fits(self) -> None:
        engine, *_ = _make_engine()

        result = await engine.simulate_services([])

        assert result.fits is True
        assert result.proposed_usage_mb == 0
        assert result.headroom_mb == 24576


# ---------------------------------------------------------------------------
# Alternative strategy tests
# ---------------------------------------------------------------------------


class TestAlternativeStrategies:
    """Tests for alternative generation strategies."""

    async def test_alternatives_sleep_strategy(self) -> None:
        svc_a = _make_service("svc-a", 16000, sleep_mode=SleepMode.L1)
        svc_b = _make_service("svc-b", 12000)
        engine, *_ = _make_engine(
            services=[svc_a, svc_b],
            modes={"heavy": [svc_a, svc_b]},
        )

        result = await engine.simulate_mode("heavy")

        sleep_alts = [a for a in result.alternatives if a.strategy == "sleep"]
        assert len(sleep_alts) >= 1
        assert "svc-a" in sleep_alts[0].affected_services
        assert sleep_alts[0].vram_saved_mb > 0

    async def test_alternatives_context_reduction(self) -> None:
        svc_a = _make_service("svc-a", 16000, model_id="meta-llama/Llama-3.1-8B")
        svc_b = _make_service("svc-b", 12000)

        # The estimate_vram mock returns less VRAM for smaller context
        async def _estimate(model_id: str, context_size: int) -> int:
            # Simulate: halving context saves ~4000MB
            if context_size <= 2048:
                return 12000
            return 16000

        estimate_fn = AsyncMock(side_effect=_estimate)
        engine, *_ = _make_engine(
            services=[svc_a, svc_b],
            modes={"heavy": [svc_a, svc_b]},
            estimate_vram_fn=estimate_fn,
        )

        result = await engine.simulate_mode("heavy")

        ctx_alts = [a for a in result.alternatives if a.strategy == "context_reduction"]
        assert len(ctx_alts) >= 1
        assert ctx_alts[0].vram_saved_mb > 0

    async def test_alternatives_service_removal(self) -> None:
        svc_a = _make_service("svc-a", 16000)
        svc_b = _make_service("svc-b", 12000)
        engine, *_ = _make_engine(
            services=[svc_a, svc_b],
            modes={"heavy": [svc_a, svc_b]},
        )

        result = await engine.simulate_mode("heavy")

        removal_alts = [a for a in result.alternatives if a.strategy == "service_removal"]
        assert len(removal_alts) >= 1
        # Should suggest removing the largest service first
        assert removal_alts[0].affected_services == ["svc-a"]
        assert removal_alts[0].vram_saved_mb == 16000

    async def test_alternatives_ranked_by_vram_saved(self) -> None:
        svc_a = _make_service("svc-a", 16000, sleep_mode=SleepMode.L1)
        svc_b = _make_service("svc-b", 12000)
        engine, *_ = _make_engine(
            services=[svc_a, svc_b],
            modes={"heavy": [svc_a, svc_b]},
        )

        result = await engine.simulate_mode("heavy")

        vram_saved_values = [a.vram_saved_mb for a in result.alternatives]
        assert vram_saved_values == sorted(vram_saved_values, reverse=True)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error scenarios."""

    async def test_gpu_info_unavailable(self) -> None:
        engine, *_ = _make_engine(gpu_raises=True)

        with pytest.raises(SimulationError, match="GPU"):
            await engine.simulate_services([])


# ---------------------------------------------------------------------------
# Input validation tests (SEC-V1)
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input validation via validation.py."""

    def test_validate_service_id_rejects_shell_injection(self) -> None:
        with pytest.raises(ValueError, match="service_id"):
            validate_service_id("; rm -rf /")

    def test_validate_mode_id_rejects_sql_injection(self) -> None:
        with pytest.raises(ValueError, match="mode_id"):
            validate_mode_id("'; DROP TABLE services--")

    def test_validate_service_id_accepts_valid(self) -> None:
        assert validate_service_id("vllm-chat-01") == "vllm-chat-01"

    def test_validate_mode_id_accepts_valid(self) -> None:
        assert validate_mode_id("chat-mode") == "chat-mode"

    def test_validate_context_override_rejects_too_large(self) -> None:
        with pytest.raises(ValueError, match="context"):
            validate_context_override("svc-a", 200000)

    def test_validate_context_override_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="context"):
            validate_context_override("svc-a", 0)

    def test_validate_context_override_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="context"):
            validate_context_override("svc-a", -1)

    def test_validate_context_override_accepts_valid(self) -> None:
        key, val = validate_context_override("svc-a", 4096)
        assert key == "svc-a"
        assert val == 4096

    def test_validate_service_id_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="service_id"):
            validate_service_id("")

    def test_validate_service_id_rejects_template_injection(self) -> None:
        with pytest.raises(ValueError, match="service_id"):
            validate_service_id("{{7*7}}")
