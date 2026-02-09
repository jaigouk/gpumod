"""Tests for Phase A migration preset YAMLs.

Validates that all 8 production service presets:
- Parse into PresetConfig without errors
- Have correct ports, VRAM, drivers, sleep modes
- Have no duplicate IDs or port conflicts
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpumod.models import DriverType, PresetConfig, SleepMode
from gpumod.templates.presets import PresetLoader

PRESETS_DIR = Path(__file__).resolve().parents[2] / "presets"

EXPECTED_SERVICES: dict[str, dict[str, object]] = {
    "vllm-embedding": {
        "driver": DriverType.VLLM,
        "port": 8200,
        "vram_mb": 5000,
        "sleep_mode": SleepMode.NONE,
        "model_id": "Qwen/Qwen3-VL-Embedding-2B",
    },
    "vllm-embedding-code": {
        "driver": DriverType.VLLM,
        "port": 8210,
        "vram_mb": 2500,
        "sleep_mode": SleepMode.NONE,
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
    },
    "vllm-hyde": {
        "driver": DriverType.VLLM,
        "port": 8202,
        "vram_mb": 5000,
        "sleep_mode": SleepMode.L2,
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    },
    "vllm-reranker": {
        "driver": DriverType.VLLM,
        "port": 8201,
        "vram_mb": 6000,
        "sleep_mode": SleepMode.L2,
        "model_id": "Qwen/Qwen3-VL-Reranker-2B",
    },
    "vllm-chat": {
        "driver": DriverType.VLLM,
        "port": 7071,
        "vram_mb": 7000,
        "sleep_mode": SleepMode.L1,
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    },
    "qwen3-coder": {
        "driver": DriverType.LLAMACPP,
        "port": 7070,
        "vram_mb": 20000,
        "sleep_mode": SleepMode.ROUTER,
    },
    "qwen3-asr": {
        "driver": DriverType.FASTAPI,
        "port": 8203,
        "vram_mb": 5000,
        "sleep_mode": SleepMode.L1,
        "model_id": "Qwen/Qwen3-ASR-1.7B",
    },
    "vllm-tts": {
        "driver": DriverType.VLLM,
        "port": 8204,
        "vram_mb": 5000,
        "sleep_mode": SleepMode.NONE,
        "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    },
}


@pytest.fixture
def loader() -> PresetLoader:
    return PresetLoader(preset_dirs=[PRESETS_DIR])


@pytest.fixture
def all_presets(loader: PresetLoader) -> list[PresetConfig]:
    return loader.discover_presets()


class TestMigrationPresetsExist:
    """All 8 preset YAML files must exist on disk."""

    EXPECTED_FILES = [
        "embedding/vllm-embedding.yaml",
        "embedding/vllm-embedding-code.yaml",
        "llm/qwen3-coder.yaml",
        "llm/vllm-chat.yaml",
        "llm/vllm-hyde.yaml",
        "llm/vllm-reranker.yaml",
        "speech/qwen3-asr.yaml",
        "speech/vllm-tts.yaml",
    ]

    @pytest.mark.parametrize("rel_path", EXPECTED_FILES)
    def test_preset_file_exists(self, rel_path: str) -> None:
        full_path = PRESETS_DIR / rel_path
        assert full_path.exists(), f"Missing preset: {rel_path}"

    def test_total_migration_preset_count(self, all_presets: list[PresetConfig]) -> None:
        migration_ids = {p.id for p in all_presets if p.id in EXPECTED_SERVICES}
        assert len(migration_ids) == 8, (
            f"Expected 8 migration presets, found {len(migration_ids)}: {migration_ids}"
        )


class TestMigrationPresetsParseCorrectly:
    """Each YAML must parse into PresetConfig without validation errors."""

    @pytest.mark.parametrize("service_id", list(EXPECTED_SERVICES.keys()))
    def test_preset_parses(self, all_presets: list[PresetConfig], service_id: str) -> None:
        preset = next((p for p in all_presets if p.id == service_id), None)
        assert preset is not None, f"Preset '{service_id}' not found in loaded presets"
        assert isinstance(preset, PresetConfig)


class TestMigrationPresetValues:
    """Validate exact values match the production systemd services."""

    @pytest.mark.parametrize("service_id", list(EXPECTED_SERVICES.keys()))
    def test_port(self, all_presets: list[PresetConfig], service_id: str) -> None:
        expected = EXPECTED_SERVICES[service_id]
        preset = next(p for p in all_presets if p.id == service_id)
        assert preset.port == expected["port"], (
            f"{service_id}: port {preset.port} != {expected['port']}"
        )

    @pytest.mark.parametrize("service_id", list(EXPECTED_SERVICES.keys()))
    def test_vram(self, all_presets: list[PresetConfig], service_id: str) -> None:
        expected = EXPECTED_SERVICES[service_id]
        preset = next(p for p in all_presets if p.id == service_id)
        assert preset.vram_mb == expected["vram_mb"], (
            f"{service_id}: vram_mb {preset.vram_mb} != {expected['vram_mb']}"
        )

    @pytest.mark.parametrize("service_id", list(EXPECTED_SERVICES.keys()))
    def test_driver(self, all_presets: list[PresetConfig], service_id: str) -> None:
        expected = EXPECTED_SERVICES[service_id]
        preset = next(p for p in all_presets if p.id == service_id)
        assert preset.driver == expected["driver"], (
            f"{service_id}: driver {preset.driver} != {expected['driver']}"
        )

    @pytest.mark.parametrize("service_id", list(EXPECTED_SERVICES.keys()))
    def test_sleep_mode(self, all_presets: list[PresetConfig], service_id: str) -> None:
        expected = EXPECTED_SERVICES[service_id]
        preset = next(p for p in all_presets if p.id == service_id)
        assert preset.sleep_mode == expected["sleep_mode"], (
            f"{service_id}: sleep_mode {preset.sleep_mode} != {expected['sleep_mode']}"
        )

    @pytest.mark.parametrize(
        "service_id",
        [sid for sid, vals in EXPECTED_SERVICES.items() if "model_id" in vals],
    )
    def test_model_id(self, all_presets: list[PresetConfig], service_id: str) -> None:
        expected = EXPECTED_SERVICES[service_id]
        preset = next(p for p in all_presets if p.id == service_id)
        assert preset.model_id == expected["model_id"], (
            f"{service_id}: model_id {preset.model_id} != {expected['model_id']}"
        )


class TestNoDuplicates:
    """No duplicate IDs or port conflicts among migration presets."""

    def test_no_duplicate_ids(self, all_presets: list[PresetConfig]) -> None:
        ids = [p.id for p in all_presets]
        assert len(ids) == len(set(ids)), f"Duplicate preset IDs: {ids}"

    def test_no_port_conflicts_in_migration(self, all_presets: list[PresetConfig]) -> None:
        migration = [p for p in all_presets if p.id in EXPECTED_SERVICES]
        ports = [p.port for p in migration if p.port is not None]
        assert len(ports) == len(set(ports)), f"Port conflict: {ports}"


class TestDriverDistribution:
    """Verify the expected driver mix: 6 vllm, 1 llamacpp, 1 fastapi."""

    def test_vllm_count(self, all_presets: list[PresetConfig]) -> None:
        migration = [p for p in all_presets if p.id in EXPECTED_SERVICES]
        vllm_count = sum(1 for p in migration if p.driver == DriverType.VLLM)
        assert vllm_count == 6

    def test_llamacpp_count(self, all_presets: list[PresetConfig]) -> None:
        migration = [p for p in all_presets if p.id in EXPECTED_SERVICES]
        llamacpp_count = sum(1 for p in migration if p.driver == DriverType.LLAMACPP)
        assert llamacpp_count == 1

    def test_fastapi_count(self, all_presets: list[PresetConfig]) -> None:
        migration = [p for p in all_presets if p.id in EXPECTED_SERVICES]
        fastapi_count = sum(1 for p in migration if p.driver == DriverType.FASTAPI)
        assert fastapi_count == 1


class TestSleepModeConsistency:
    """supports_sleep must be True when sleep_mode is not none."""

    @pytest.mark.parametrize("service_id", list(EXPECTED_SERVICES.keys()))
    def test_supports_sleep_flag(self, all_presets: list[PresetConfig], service_id: str) -> None:
        preset = next(p for p in all_presets if p.id == service_id)
        if preset.sleep_mode != SleepMode.NONE:
            assert preset.supports_sleep is True, (
                f"{service_id}: sleep_mode={preset.sleep_mode} but supports_sleep=False"
            )
        else:
            assert preset.supports_sleep is False, (
                f"{service_id}: sleep_mode=none but supports_sleep=True"
            )


class TestHealthEndpoint:
    """All services must have /health endpoint."""

    @pytest.mark.parametrize("service_id", list(EXPECTED_SERVICES.keys()))
    def test_health_endpoint(self, all_presets: list[PresetConfig], service_id: str) -> None:
        preset = next(p for p in all_presets if p.id == service_id)
        assert preset.health_endpoint == "/health"


class TestGpuMemUtil:
    """vllm services should have gpu_mem_util in unit_vars."""

    EXPECTED_GPU_MEM_UTIL: dict[str, float] = {
        "vllm-embedding": 0.30,
        "vllm-embedding-code": 0.085,
        "vllm-hyde": 0.22,
        "vllm-reranker": 0.25,
        "vllm-chat": 0.35,
        "vllm-tts": 0.2,
    }

    @pytest.mark.parametrize("service_id", list(EXPECTED_GPU_MEM_UTIL.keys()))
    def test_gpu_mem_util(self, all_presets: list[PresetConfig], service_id: str) -> None:
        expected = self.EXPECTED_GPU_MEM_UTIL[service_id]
        preset = next(p for p in all_presets if p.id == service_id)
        actual = preset.unit_vars.get("gpu_mem_util")
        assert actual == pytest.approx(expected), (
            f"{service_id}: gpu_mem_util {actual} != {expected}"
        )
