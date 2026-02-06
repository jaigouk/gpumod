"""Comprehensive tests for gpumod.models Pydantic models and enums."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gpumod.models import (
    DriverType,
    GPUInfo,
    Mode,
    ModeResult,
    Service,
    ServiceInfo,
    ServiceState,
    ServiceStatus,
    SleepMode,
    SystemStatus,
    VRAMUsage,
)

# ── Enum membership & string values ──────────────────────────────────────


class TestServiceStateEnum:
    def test_members(self) -> None:
        assert ServiceState.UNKNOWN == "unknown"
        assert ServiceState.STOPPED == "stopped"
        assert ServiceState.STARTING == "starting"
        assert ServiceState.RUNNING == "running"
        assert ServiceState.SLEEPING == "sleeping"
        assert ServiceState.UNHEALTHY == "unhealthy"
        assert ServiceState.STOPPING == "stopping"
        assert ServiceState.FAILED == "failed"

    def test_member_count(self) -> None:
        assert len(ServiceState) == 8

    def test_string_conversion(self) -> None:
        assert ServiceState.RUNNING.value == "running"

    def test_value_lookup(self) -> None:
        assert ServiceState("running") is ServiceState.RUNNING


class TestDriverTypeEnum:
    def test_members(self) -> None:
        assert DriverType.VLLM == "vllm"
        assert DriverType.LLAMACPP == "llamacpp"
        assert DriverType.FASTAPI == "fastapi"
        assert DriverType.DOCKER == "docker"

    def test_member_count(self) -> None:
        assert len(DriverType) == 4

    def test_string_conversion(self) -> None:
        assert DriverType.VLLM.value == "vllm"


class TestSleepModeEnum:
    def test_members(self) -> None:
        assert SleepMode.NONE == "none"
        assert SleepMode.L1 == "l1"
        assert SleepMode.L2 == "l2"
        assert SleepMode.ROUTER == "router"

    def test_member_count(self) -> None:
        assert len(SleepMode) == 4

    def test_string_conversion(self) -> None:
        assert SleepMode.L1.value == "l1"


# ── Service model ────────────────────────────────────────────────────────


class TestServiceModel:
    def test_required_fields(self) -> None:
        svc = Service(id="svc1", name="Test", driver=DriverType.VLLM, vram_mb=4096)
        assert svc.id == "svc1"
        assert svc.name == "Test"
        assert svc.driver == DriverType.VLLM
        assert svc.vram_mb == 4096

    def test_defaults(self) -> None:
        svc = Service(id="svc1", name="Test", driver=DriverType.VLLM, vram_mb=4096)
        assert svc.port is None
        assert svc.sleep_mode == SleepMode.NONE
        assert svc.health_endpoint == "/health"
        assert svc.model_id is None
        assert svc.unit_name is None
        assert svc.depends_on == []
        assert svc.startup_timeout == 120
        assert svc.extra_config == {}

    def test_all_fields_populated(self) -> None:
        svc = Service(
            id="vllm-chat",
            name="vLLM Chat",
            driver=DriverType.VLLM,
            port=8000,
            vram_mb=8000,
            sleep_mode=SleepMode.L1,
            health_endpoint="/v1/health",
            model_id="meta-llama/Llama-3-8B",
            unit_name="vllm-chat.service",
            depends_on=["embedding"],
            startup_timeout=300,
            extra_config={"gpu_mem_util": 0.9},
        )
        assert svc.port == 8000
        assert svc.sleep_mode == SleepMode.L1
        assert svc.health_endpoint == "/v1/health"
        assert svc.model_id == "meta-llama/Llama-3-8B"
        assert svc.unit_name == "vllm-chat.service"
        assert svc.depends_on == ["embedding"]
        assert svc.startup_timeout == 300
        assert svc.extra_config == {"gpu_mem_util": 0.9}

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            Service(id="svc1", name="Test", driver=DriverType.VLLM)  # type: ignore[call-arg]

    def test_invalid_driver_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            Service(id="svc1", name="Test", driver="invalid", vram_mb=4096)  # type: ignore[arg-type]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Service(
                id="svc1",
                name="Test",
                driver=DriverType.VLLM,
                vram_mb=4096,
                nonexistent_field="bad",  # type: ignore[call-arg]
            )


# ── ServiceStatus model ─────────────────────────────────────────────────


class TestServiceStatusModel:
    def test_all_none_optional_fields(self) -> None:
        status = ServiceStatus(state=ServiceState.STOPPED)
        assert status.state == ServiceState.STOPPED
        assert status.vram_mb is None
        assert status.uptime_seconds is None
        assert status.health_ok is None
        assert status.sleep_level is None
        assert status.last_error is None

    def test_all_fields_populated(self) -> None:
        status = ServiceStatus(
            state=ServiceState.RUNNING,
            vram_mb=8000,
            uptime_seconds=3600,
            health_ok=True,
            sleep_level="l1",
            last_error=None,
        )
        assert status.state == ServiceState.RUNNING
        assert status.vram_mb == 8000
        assert status.uptime_seconds == 3600
        assert status.health_ok is True
        assert status.sleep_level == "l1"

    def test_with_error(self) -> None:
        status = ServiceStatus(
            state=ServiceState.FAILED,
            last_error="Connection refused",
        )
        assert status.last_error == "Connection refused"

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ServiceStatus(state=ServiceState.RUNNING, bogus=True)  # type: ignore[call-arg]


# ── Mode model serialization round-trip ──────────────────────────────────


class TestModeModel:
    def test_creation(self) -> None:
        mode = Mode(
            id="code",
            name="Code Mode",
            description="Agentic coding",
            services=["embedding-code", "glm-code"],
            total_vram_mb=22000,
        )
        assert mode.id == "code"
        assert mode.name == "Code Mode"
        assert mode.services == ["embedding-code", "glm-code"]
        assert mode.total_vram_mb == 22000

    def test_defaults(self) -> None:
        mode = Mode(id="blank", name="Blank Mode")
        assert mode.description is None
        assert mode.services == []
        assert mode.total_vram_mb is None

    def test_serialization_round_trip(self) -> None:
        mode = Mode(
            id="rag",
            name="RAG Mode",
            description="Full RAG pipeline",
            services=["embedding", "hyde", "reranker", "chat"],
            total_vram_mb=13500,
        )
        data = mode.model_dump()
        restored = Mode.model_validate(data)
        assert restored == mode

    def test_json_round_trip(self) -> None:
        mode = Mode(
            id="speak",
            name="Speak Mode",
            services=["embedding", "asr", "tts", "chat"],
            total_vram_mb=23000,
        )
        json_str = mode.model_dump_json()
        restored = Mode.model_validate_json(json_str)
        assert restored == mode


# ── ModeResult model ─────────────────────────────────────────────────────


class TestModeResultModel:
    def test_success(self) -> None:
        result = ModeResult(
            success=True,
            mode_id="code",
            started=["embedding-code", "glm-code"],
            stopped=[],
            message="Switched to code mode",
        )
        assert result.success is True
        assert result.mode_id == "code"
        assert result.started == ["embedding-code", "glm-code"]
        assert result.stopped == []
        assert result.message == "Switched to code mode"

    def test_failure(self) -> None:
        result = ModeResult(
            success=False,
            mode_id="rag",
            started=[],
            stopped=[],
            message="VRAM exceeded by 3.7GB",
            errors=["embedding failed health check"],
        )
        assert result.success is False
        assert result.errors == ["embedding failed health check"]

    def test_defaults(self) -> None:
        result = ModeResult(success=True, mode_id="blank")
        assert result.started == []
        assert result.stopped == []
        assert result.message is None
        assert result.errors == []


# ── GPUInfo model ────────────────────────────────────────────────────────


class TestGPUInfoModel:
    def test_creation(self) -> None:
        gpu = GPUInfo(
            name="NVIDIA GeForce RTX 4090",
            vram_total_mb=24576,
            architecture="Ada Lovelace",
        )
        assert gpu.name == "NVIDIA GeForce RTX 4090"
        assert gpu.vram_total_mb == 24576
        assert gpu.architecture == "Ada Lovelace"

    def test_optional_architecture(self) -> None:
        gpu = GPUInfo(name="RTX 3090", vram_total_mb=24576)
        assert gpu.architecture is None


# ── VRAMUsage model ──────────────────────────────────────────────────────


class TestVRAMUsageModel:
    def test_creation(self) -> None:
        usage = VRAMUsage(
            total_mb=24576,
            used_mb=21700,
            free_mb=2876,
        )
        assert usage.total_mb == 24576
        assert usage.used_mb == 21700
        assert usage.free_mb == 2876

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VRAMUsage(total_mb=24576, used_mb=21700, free_mb=2876, extra=1)  # type: ignore[call-arg]


# ── ServiceInfo model ────────────────────────────────────────────────────


class TestServiceInfoModel:
    def test_creation(self) -> None:
        info = ServiceInfo(
            service=Service(
                id="vllm-chat",
                name="vLLM Chat",
                driver=DriverType.VLLM,
                vram_mb=8000,
            ),
            status=ServiceStatus(state=ServiceState.RUNNING, vram_mb=8000, health_ok=True),
        )
        assert info.service.id == "vllm-chat"
        assert info.status.state == ServiceState.RUNNING


# ── SystemStatus model ───────────────────────────────────────────────────


class TestSystemStatusModel:
    def test_creation(self) -> None:
        status = SystemStatus(
            gpu=GPUInfo(name="RTX 4090", vram_total_mb=24576),
            vram=VRAMUsage(total_mb=24576, used_mb=21700, free_mb=2876),
            current_mode="code",
            services=[
                ServiceInfo(
                    service=Service(
                        id="emb",
                        name="Embedding",
                        driver=DriverType.VLLM,
                        vram_mb=2500,
                    ),
                    status=ServiceStatus(state=ServiceState.RUNNING),
                ),
            ],
        )
        assert status.gpu.name == "RTX 4090"
        assert status.current_mode == "code"
        assert len(status.services) == 1

    def test_defaults(self) -> None:
        status = SystemStatus(
            gpu=GPUInfo(name="RTX 4090", vram_total_mb=24576),
            vram=VRAMUsage(total_mb=24576, used_mb=0, free_mb=24576),
        )
        assert status.current_mode is None
        assert status.services == []
