"""Comprehensive tests for gpumod.models Pydantic models and enums."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gpumod.models import (
    DriverType,
    GPUInfo,
    KVCacheProfile,
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
        assert svc.preflight_required is False  # gpumod-ecr opt-in flag

    def test_preflight_required_true(self) -> None:
        """preflight_required can be opted-in per service preset."""
        svc = Service(
            id="qwen36",
            name="Qwen 3.6",
            driver=DriverType.LLAMACPP,
            vram_mb=22000,
            preflight_required=True,
        )
        assert svc.preflight_required is True

    def test_compat_defaults_to_none(self) -> None:
        """compat defaults to None — no version contract declared (gpumod-ng7)."""
        svc = Service(id="svc1", name="Test", driver=DriverType.VLLM, vram_mb=4096)
        assert svc.compat is None

    def test_compat_dict_parses(self) -> None:
        """compat is a dict[str, str] of PEP 440 specifiers."""
        svc = Service(
            id="vllm-embedding",
            name="Embedding",
            driver=DriverType.VLLM,
            vram_mb=2500,
            compat={
                "vllm": ">=0.11.0,<0.12",
                "transformers": ">=4.55.2,<5.0",
                "huggingface-hub": ">=0.34.0,<1.0",
            },
        )
        assert svc.compat is not None
        assert svc.compat["vllm"] == ">=0.11.0,<0.12"
        assert svc.compat["transformers"] == ">=4.55.2,<5.0"

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
            services=["embedding-code", "qwen3-coder"],
            total_vram_mb=22000,
        )
        assert mode.id == "code"
        assert mode.name == "Code Mode"
        assert mode.services == ["embedding-code", "qwen3-coder"]
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
            started=["embedding-code", "qwen3-coder"],
            stopped=[],
            message="Switched to code mode",
        )
        assert result.success is True
        assert result.mode_id == "code"
        assert result.started == ["embedding-code", "qwen3-coder"]
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

    def test_kv_cache_profile_defaults_to_none(self) -> None:
        """kv_cache_profile defaults to None — backward compatible."""
        model = ModelInfo(id="test-model", source=ModelSource.LOCAL)
        assert model.kv_cache_profile is None

    def test_kv_cache_profile_coexists_with_scalar(self) -> None:
        """Both kv_cache_per_1k_tokens_mb and kv_cache_profile can be set."""
        profile = KVCacheProfile(
            num_sliding_layers=0,
            num_global_layers=64,
            head_dim=128,
            num_kv_heads=8,
        )
        model = ModelInfo(
            id="Qwen/Qwen3-32B",
            source=ModelSource.HUGGINGFACE,
            kv_cache_per_1k_tokens_mb=250,
            kv_cache_profile=profile,
        )
        assert model.kv_cache_per_1k_tokens_mb == 250
        assert model.kv_cache_profile is not None
        assert model.kv_cache_profile.num_global_layers == 64

    def test_model_info_with_profile_serialization_round_trip(self) -> None:
        """ModelInfo with kv_cache_profile survives dict round-trip."""
        profile = KVCacheProfile(
            num_sliding_layers=28,
            num_global_layers=7,
            num_kv_shared_layers=15,
            sliding_window=512,
            head_dim=256,
            num_kv_heads=2,
        )
        model = ModelInfo(
            id="google/gemma-3n-E4B-it",
            source=ModelSource.HUGGINGFACE,
            kv_cache_per_1k_tokens_mb=69,
            kv_cache_profile=profile,
        )
        data = model.model_dump()
        restored = ModelInfo.model_validate(data)
        assert restored == model
        assert restored.kv_cache_profile == profile

    def test_model_info_with_profile_json_round_trip(self) -> None:
        """ModelInfo with kv_cache_profile survives JSON round-trip."""
        profile = KVCacheProfile(
            num_sliding_layers=52,
            num_global_layers=10,
            sliding_window=1024,
            head_dim=128,
            num_kv_heads=16,
        )
        model = ModelInfo(
            id="google/gemma-3-27b-it",
            source=ModelSource.HUGGINGFACE,
            kv_cache_per_1k_tokens_mb=485,
            kv_cache_profile=profile,
        )
        json_str = model.model_dump_json()
        restored = ModelInfo.model_validate_json(json_str)
        assert restored == model
        assert restored.kv_cache_profile is not None
        assert restored.kv_cache_profile.sliding_window == 1024


# ── KVCacheProfile model ───────────────────────────────────────────────


class TestKVCacheProfileModel:
    def test_defaults(self) -> None:
        """KVCacheProfile with no args uses sensible defaults."""
        profile = KVCacheProfile()
        assert profile.num_sliding_layers == 0
        assert profile.num_global_layers == 0
        assert profile.num_kv_shared_layers == 0
        assert profile.sliding_window is None
        assert profile.head_dim == 128
        assert profile.global_head_dim is None
        assert profile.num_kv_heads == 1
        assert profile.num_global_kv_heads is None
        assert profile.attention_k_eq_v is False
        assert profile.triattn_budget is None
        assert profile.kv_per_1k_at_inf is None

    def test_dense_model_profile(self) -> None:
        """Dense model (Qwen3-32B): all global layers, no sliding."""
        profile = KVCacheProfile(
            num_sliding_layers=0,
            num_global_layers=64,
            head_dim=128,
            num_kv_heads=8,
            kv_per_1k_at_inf=250,
        )
        assert profile.num_sliding_layers == 0
        assert profile.num_global_layers == 64
        assert profile.sliding_window is None
        assert profile.kv_per_1k_at_inf == 250

    def test_hybrid_model_profile(self) -> None:
        """Hybrid model (Gemma 3n E4B): sliding + global + shared layers."""
        profile = KVCacheProfile(
            num_sliding_layers=28,
            num_global_layers=7,
            num_kv_shared_layers=15,
            sliding_window=512,
            head_dim=256,
            num_kv_heads=2,
            kv_per_1k_at_inf=69,
        )
        assert profile.num_sliding_layers == 28
        assert profile.num_global_layers == 7
        assert profile.num_kv_shared_layers == 15
        assert profile.sliding_window == 512
        assert profile.head_dim == 256
        assert profile.num_kv_heads == 2

    def test_asymmetric_head_dim_profile(self) -> None:
        """Profile with different global head_dim and kv_heads (Gemma 4 style)."""
        profile = KVCacheProfile(
            num_sliding_layers=20,
            num_global_layers=10,
            head_dim=128,
            global_head_dim=256,
            num_kv_heads=8,
            num_global_kv_heads=16,
            attention_k_eq_v=True,
        )
        assert profile.global_head_dim == 256
        assert profile.num_global_kv_heads == 16
        assert profile.attention_k_eq_v is True

    def test_extra_fields_rejected(self) -> None:
        """ConfigDict(extra='forbid') rejects unknown fields."""
        with pytest.raises(ValidationError):
            KVCacheProfile(bogus="bad")  # type: ignore[call-arg]

    def test_serialization_round_trip(self) -> None:
        """KVCacheProfile survives dict round-trip."""
        profile = KVCacheProfile(
            num_sliding_layers=28,
            num_global_layers=7,
            num_kv_shared_layers=15,
            sliding_window=512,
            head_dim=256,
            num_kv_heads=2,
            attention_k_eq_v=False,
            triattn_budget=None,
            kv_per_1k_at_inf=69,
        )
        data = profile.model_dump()
        restored = KVCacheProfile.model_validate(data)
        assert restored == profile

    def test_json_round_trip(self) -> None:
        """KVCacheProfile survives JSON round-trip."""
        profile = KVCacheProfile(
            num_sliding_layers=52,
            num_global_layers=10,
            sliding_window=1024,
            head_dim=128,
            num_kv_heads=16,
        )
        json_str = profile.model_dump_json()
        restored = KVCacheProfile.model_validate_json(json_str)
        assert restored == profile


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
