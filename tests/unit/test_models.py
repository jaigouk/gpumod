"""Comprehensive tests for gpumod.models Pydantic models and enums."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gpumod.models import (
    DriverType,
    GPUInfo,
    Mode,
    ModelInfo,
    ModelSource,
    ModeResult,
    PresetConfig,
    Service,
    ServiceInfo,
    ServiceState,
    ServiceStatus,
    ServiceTemplate,
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


# ── ModelSource enum ─────────────────────────────────────────────────────


class TestModelSourceEnum:
    def test_members(self) -> None:
        assert ModelSource.HUGGINGFACE == "huggingface"
        assert ModelSource.GGUF == "gguf"
        assert ModelSource.LOCAL == "local"

    def test_member_count(self) -> None:
        assert len(ModelSource) == 3

    def test_string_conversion(self) -> None:
        assert ModelSource.HUGGINGFACE.value == "huggingface"

    def test_value_lookup(self) -> None:
        assert ModelSource("gguf") is ModelSource.GGUF


# ── ModelInfo model ──────────────────────────────────────────────────────


class TestModelInfoModel:
    def test_required_fields(self) -> None:
        model = ModelInfo(id="meta-llama/Llama-3-8B", source=ModelSource.HUGGINGFACE)
        assert model.id == "meta-llama/Llama-3-8B"
        assert model.source == ModelSource.HUGGINGFACE

    def test_defaults(self) -> None:
        model = ModelInfo(id="test-model", source=ModelSource.LOCAL)
        assert model.parameters_b is None
        assert model.architecture is None
        assert model.base_vram_mb is None
        assert model.kv_cache_per_1k_tokens_mb is None
        assert model.quantizations == []
        assert model.capabilities == []
        assert model.fetched_at is None
        assert model.notes is None

    def test_all_fields_populated(self) -> None:
        model = ModelInfo(
            id="meta-llama/Llama-3-8B",
            source=ModelSource.HUGGINGFACE,
            parameters_b=8.0,
            architecture="llama",
            base_vram_mb=16000,
            kv_cache_per_1k_tokens_mb=64,
            quantizations=["fp16", "q4_k_m", "q8_0"],
            capabilities=["chat", "code"],
            fetched_at="2025-01-15T10:00:00Z",
            notes="Popular coding model",
        )
        assert model.parameters_b == 8.0
        assert model.architecture == "llama"
        assert model.base_vram_mb == 16000
        assert model.kv_cache_per_1k_tokens_mb == 64
        assert model.quantizations == ["fp16", "q4_k_m", "q8_0"]
        assert model.capabilities == ["chat", "code"]
        assert model.fetched_at == "2025-01-15T10:00:00Z"
        assert model.notes == "Popular coding model"

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelInfo(
                id="test",
                source=ModelSource.HUGGINGFACE,
                bogus="bad",  # type: ignore[call-arg]
            )

    def test_invalid_source_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelInfo(id="test", source="invalid")  # type: ignore[arg-type]

    def test_serialization_round_trip(self) -> None:
        model = ModelInfo(
            id="meta-llama/Llama-3-70B",
            source=ModelSource.HUGGINGFACE,
            parameters_b=70.0,
            architecture="llama",
            base_vram_mb=140000,
            quantizations=["fp16", "q4_k_m"],
        )
        data = model.model_dump()
        restored = ModelInfo.model_validate(data)
        assert restored == model

    def test_json_round_trip(self) -> None:
        model = ModelInfo(
            id="local/my-gguf",
            source=ModelSource.GGUF,
            parameters_b=7.0,
            base_vram_mb=5000,
        )
        json_str = model.model_dump_json()
        restored = ModelInfo.model_validate_json(json_str)
        assert restored == model


# ── ServiceTemplate model ────────────────────────────────────────────────


class TestServiceTemplateModel:
    def test_required_fields(self) -> None:
        tpl = ServiceTemplate(
            service_id="vllm-chat",
            unit_template="[Unit]\nDescription={{ name }}\n",
        )
        assert tpl.service_id == "vllm-chat"
        assert tpl.unit_template == "[Unit]\nDescription={{ name }}\n"

    def test_defaults(self) -> None:
        tpl = ServiceTemplate(
            service_id="vllm-chat",
            unit_template="[Unit]\n",
        )
        assert tpl.preset_template is None

    def test_all_fields_populated(self) -> None:
        tpl = ServiceTemplate(
            service_id="vllm-chat",
            unit_template="[Unit]\nDescription={{ name }}\n",
            preset_template="id: {{ id }}\ndriver: vllm\n",
        )
        assert tpl.preset_template == "id: {{ id }}\ndriver: vllm\n"

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ServiceTemplate(
                service_id="test",
                unit_template="[Unit]\n",
                bogus="bad",  # type: ignore[call-arg]
            )

    def test_serialization_round_trip(self) -> None:
        tpl = ServiceTemplate(
            service_id="llama-code",
            unit_template="[Unit]\nDescription=llama.cpp\n",
            preset_template="driver: llamacpp\n",
        )
        data = tpl.model_dump()
        restored = ServiceTemplate.model_validate(data)
        assert restored == tpl


# ── PresetConfig model ───────────────────────────────────────────────────


class TestPresetConfigModel:
    def test_required_fields(self) -> None:
        preset = PresetConfig(
            id="vllm-chat",
            name="vLLM Chat",
            driver=DriverType.VLLM,
            vram_mb=8000,
        )
        assert preset.id == "vllm-chat"
        assert preset.name == "vLLM Chat"
        assert preset.driver == DriverType.VLLM
        assert preset.vram_mb == 8000

    def test_defaults(self) -> None:
        preset = PresetConfig(
            id="test",
            name="Test",
            driver=DriverType.VLLM,
            vram_mb=4096,
        )
        assert preset.port is None
        assert preset.context_size is None
        assert preset.kv_cache_per_1k is None
        assert preset.model_id is None
        assert preset.model_path is None
        assert preset.health_endpoint == "/health"
        assert preset.startup_timeout == 60
        assert preset.supports_sleep is False
        assert preset.sleep_mode == SleepMode.NONE
        assert preset.unit_template is None
        assert preset.unit_vars == {}

    def test_all_fields_populated(self) -> None:
        preset = PresetConfig(
            id="vllm-chat",
            name="vLLM Chat",
            driver=DriverType.VLLM,
            port=8000,
            vram_mb=8000,
            context_size=4096,
            kv_cache_per_1k=64,
            model_id="meta-llama/Llama-3-8B",
            model_path="/models/llama-3-8b",
            health_endpoint="/v1/health",
            startup_timeout=300,
            supports_sleep=True,
            sleep_mode=SleepMode.L1,
            unit_template="[Unit]\nDescription={{ name }}\n",
            unit_vars={"gpu_mem_util": "0.9", "max_model_len": "4096"},
        )
        assert preset.port == 8000
        assert preset.context_size == 4096
        assert preset.kv_cache_per_1k == 64
        assert preset.model_id == "meta-llama/Llama-3-8B"
        assert preset.model_path == "/models/llama-3-8b"
        assert preset.health_endpoint == "/v1/health"
        assert preset.startup_timeout == 300
        assert preset.supports_sleep is True
        assert preset.sleep_mode == SleepMode.L1
        assert preset.unit_template == "[Unit]\nDescription={{ name }}\n"
        assert preset.unit_vars == {"gpu_mem_util": "0.9", "max_model_len": "4096"}

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PresetConfig(
                id="test",
                name="Test",
                driver=DriverType.VLLM,
                vram_mb=4096,
                bogus="bad",  # type: ignore[call-arg]
            )

    def test_invalid_driver_raises(self) -> None:
        with pytest.raises(ValidationError):
            PresetConfig(
                id="test",
                name="Test",
                driver="invalid",  # type: ignore[arg-type]
                vram_mb=4096,
            )

    def test_serialization_round_trip(self) -> None:
        preset = PresetConfig(
            id="llama-code",
            name="llama.cpp Code",
            driver=DriverType.LLAMACPP,
            port=8080,
            vram_mb=6000,
            model_path="/models/code.gguf",
            supports_sleep=True,
            sleep_mode=SleepMode.L2,
            unit_vars={"threads": "8"},
        )
        data = preset.model_dump()
        restored = PresetConfig.model_validate(data)
        assert restored == preset

    def test_json_round_trip(self) -> None:
        preset = PresetConfig(
            id="fastapi-proxy",
            name="FastAPI Proxy",
            driver=DriverType.FASTAPI,
            vram_mb=0,
            port=9000,
        )
        json_str = preset.model_dump_json()
        restored = PresetConfig.model_validate_json(json_str)
        assert restored == preset
