"""Tests for Phase A migration mode definitions.

Validates that all 6 production modes:
- Have correct service lists matching existing config.py MODES dict
- All service IDs reference valid preset IDs
- Total VRAM per mode fits within RTX 4090 (24GB)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from gpumod.templates.presets import PresetLoader

if TYPE_CHECKING:
    from gpumod.models import PresetConfig

MODES_DIR = Path(__file__).resolve().parents[2] / "modes"
PRESETS_DIR = Path(__file__).resolve().parents[2] / "presets"
RTX_4090_VRAM_MB = 24000

EXPECTED_MODES: dict[str, dict[str, object]] = {
    "code": {
        "services": ["vllm-embedding-code", "glm-code"],
        "description": "Coding mode with Qwen3-Coder and code embedding",
    },
    "rag": {
        "services": ["vllm-embedding-code", "vllm-embedding"],
        "description": "RAG mode with dual embedding models",
    },
    "hacker": {
        "services": ["vllm-embedding-code", "glm-code"],
        "description": "Hacking mode with code LLM and embedding",
    },
    "speak": {
        "services": ["vllm-embedding", "qwen3-asr", "vllm-tts", "vllm-chat"],
        "description": "Voice mode with ASR, TTS, chat, and embedding",
    },
    "blank": {
        "services": ["vllm-embedding-code"],
        "description": "Minimal mode with only code embedding",
    },
    "finetuning": {
        "services": ["vllm-embedding-code"],
        "description": "Finetuning mode - minimal VRAM footprint",
    },
}


def _load_mode_yaml(mode_id: str) -> dict[str, object]:
    path = MODES_DIR / f"{mode_id}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


@pytest.fixture
def all_presets() -> list[PresetConfig]:
    loader = PresetLoader(preset_dirs=[PRESETS_DIR])
    return loader.discover_presets()


@pytest.fixture
def preset_vram_map(all_presets: list[PresetConfig]) -> dict[str, int]:
    return {p.id: p.vram_mb for p in all_presets}


@pytest.fixture
def preset_ids(all_presets: list[PresetConfig]) -> set[str]:
    return {p.id for p in all_presets}


class TestModeFilesExist:
    """All 6 mode YAML files must exist."""

    EXPECTED_FILES = [
        "code.yaml",
        "rag.yaml",
        "hacker.yaml",
        "speak.yaml",
        "blank.yaml",
        "finetuning.yaml",
    ]

    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_mode_file_exists(self, filename: str) -> None:
        assert (MODES_DIR / filename).exists(), f"Missing mode: {filename}"

    def test_exactly_7_modes(self) -> None:
        yamls = list(MODES_DIR.glob("*.yaml")) + list(MODES_DIR.glob("*.yml"))
        assert len(yamls) == 7, f"Expected 7 mode files, found {len(yamls)}"


class TestModeServiceLists:
    """Each mode must have the correct service list."""

    @pytest.mark.parametrize("mode_id", list(EXPECTED_MODES.keys()))
    def test_services_match(self, mode_id: str) -> None:
        data = _load_mode_yaml(mode_id)
        expected = EXPECTED_MODES[mode_id]["services"]
        assert data["services"] == expected, (
            f"{mode_id}: services {data['services']} != {expected}"
        )

    def test_code_and_hacker_identical_services(self) -> None:
        code = _load_mode_yaml("code")
        hacker = _load_mode_yaml("hacker")
        assert code["services"] == hacker["services"]

    def test_blank_and_finetuning_identical_services(self) -> None:
        blank = _load_mode_yaml("blank")
        finetuning = _load_mode_yaml("finetuning")
        assert blank["services"] == finetuning["services"]

    def test_speak_has_4_services(self) -> None:
        speak = _load_mode_yaml("speak")
        assert len(speak["services"]) == 4


class TestModeServiceReferences:
    """All service IDs in modes must reference valid presets."""

    @pytest.mark.parametrize("mode_id", list(EXPECTED_MODES.keys()))
    def test_all_services_are_valid_presets(self, mode_id: str, preset_ids: set[str]) -> None:
        data = _load_mode_yaml(mode_id)
        for svc_id in data["services"]:
            assert svc_id in preset_ids, f"Mode '{mode_id}' references unknown service '{svc_id}'"


class TestModeVramFit:
    """Total VRAM per mode must fit within RTX 4090."""

    @pytest.mark.parametrize("mode_id", list(EXPECTED_MODES.keys()))
    def test_fits_rtx_4090(self, mode_id: str, preset_vram_map: dict[str, int]) -> None:
        data = _load_mode_yaml(mode_id)
        total = sum(preset_vram_map[sid] for sid in data["services"])
        assert total <= RTX_4090_VRAM_MB, (
            f"Mode '{mode_id}' needs {total} MB but RTX 4090 has {RTX_4090_VRAM_MB} MB"
        )

    EXPECTED_VRAM: dict[str, int] = {
        "code": 22500,
        "rag": 7500,
        "hacker": 22500,
        "speak": 22000,
        "blank": 2500,
        "finetuning": 2500,
    }

    @pytest.mark.parametrize("mode_id", list(EXPECTED_VRAM.keys()))
    def test_vram_matches_expected(self, mode_id: str, preset_vram_map: dict[str, int]) -> None:
        data = _load_mode_yaml(mode_id)
        total = sum(preset_vram_map[sid] for sid in data["services"])
        expected = self.EXPECTED_VRAM[mode_id]
        assert total == expected, f"Mode '{mode_id}' VRAM {total} MB != expected {expected} MB"


class TestModeDescriptions:
    """All modes must have descriptions."""

    @pytest.mark.parametrize("mode_id", list(EXPECTED_MODES.keys()))
    def test_has_description(self, mode_id: str) -> None:
        data = _load_mode_yaml(mode_id)
        assert data.get("description"), f"Mode '{mode_id}' missing description"

    @pytest.mark.parametrize("mode_id", list(EXPECTED_MODES.keys()))
    def test_description_matches(self, mode_id: str) -> None:
        data = _load_mode_yaml(mode_id)
        expected = EXPECTED_MODES[mode_id]["description"]
        assert data["description"] == expected
