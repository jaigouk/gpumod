"""Tests for the ModeLoader -- discovers and parses mode YAML files."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

MODES_DIR = Path(__file__).resolve().parents[2] / "modes"
PRESETS_DIR = Path(__file__).resolve().parents[2] / "presets"


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def valid_mode_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "test-mode.yaml"
    p.write_text(
        yaml.dump(
            {
                "id": "test",
                "name": "Test Mode",
                "description": "A test mode",
                "services": ["svc-a", "svc-b"],
            }
        )
    )
    return p


@pytest.fixture
def mode_directory(tmp_path: Path) -> Path:
    (tmp_path / "alpha.yaml").write_text(
        yaml.dump(
            {
                "id": "alpha",
                "name": "Alpha",
                "description": "Alpha mode",
                "services": ["svc-a"],
            }
        )
    )
    (tmp_path / "beta.yml").write_text(
        yaml.dump(
            {
                "id": "beta",
                "name": "Beta",
                "description": "Beta mode",
                "services": ["svc-b"],
            }
        )
    )
    (tmp_path / "README.md").write_text("# Not a mode")
    return tmp_path


# ── load_file ────────────────────────────────────────────────────────────


class TestLoadFile:
    def test_parses_valid_yaml(self, valid_mode_yaml: Path) -> None:
        from gpumod.templates.modes import ModeLoader

        loader = ModeLoader()
        mode = loader.load_file(valid_mode_yaml)
        assert mode.id == "test"
        assert mode.name == "Test Mode"
        assert mode.description == "A test mode"
        assert mode.services == ["svc-a", "svc-b"]

    def test_total_vram_defaults_to_none(self, valid_mode_yaml: Path) -> None:
        from gpumod.templates.modes import ModeLoader

        loader = ModeLoader()
        mode = loader.load_file(valid_mode_yaml)
        assert mode.total_vram_mb is None

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        from gpumod.templates.modes import ModeLoader

        loader = ModeLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file(tmp_path / "nonexistent.yaml")

    def test_raises_on_invalid_yaml(self, tmp_path: Path) -> None:
        from gpumod.templates.modes import ModeLoader

        bad = tmp_path / "bad.yaml"
        bad.write_text("{{not yaml}}: [broken\n")
        loader = ModeLoader()
        with pytest.raises((yaml.YAMLError, ValueError)):
            loader.load_file(bad)

    def test_raises_on_missing_required_fields(self, tmp_path: Path) -> None:
        from pydantic import ValidationError

        from gpumod.templates.modes import ModeLoader

        bad = tmp_path / "missing.yaml"
        bad.write_text(yaml.dump({"name": "No ID"}))
        loader = ModeLoader()
        with pytest.raises(ValidationError, match="id"):
            loader.load_file(bad)


# ── load_directory ───────────────────────────────────────────────────────


class TestLoadDirectory:
    def test_loads_all_yaml_files(self, mode_directory: Path) -> None:
        from gpumod.templates.modes import ModeLoader

        loader = ModeLoader()
        modes = loader.load_directory(mode_directory)
        ids = {m.id for m in modes}
        assert ids == {"alpha", "beta"}

    def test_ignores_non_yaml(self, mode_directory: Path) -> None:
        from gpumod.templates.modes import ModeLoader

        loader = ModeLoader()
        modes = loader.load_directory(mode_directory)
        assert len(modes) == 2

    def test_empty_directory(self, tmp_path: Path) -> None:
        from gpumod.templates.modes import ModeLoader

        empty = tmp_path / "empty"
        empty.mkdir()
        loader = ModeLoader()
        assert loader.load_directory(empty) == []

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        from gpumod.templates.modes import ModeLoader

        loader = ModeLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_directory(tmp_path / "missing")


# ── discover_modes ───────────────────────────────────────────────────────


class TestDiscoverModes:
    def test_discovers_from_configured_dirs(self, mode_directory: Path) -> None:
        from gpumod.templates.modes import ModeLoader

        loader = ModeLoader(mode_dirs=[mode_directory])
        modes = loader.discover_modes()
        assert len(modes) == 2

    def test_empty_when_no_dirs(self) -> None:
        from gpumod.templates.modes import ModeLoader

        loader = ModeLoader(mode_dirs=[])
        assert loader.discover_modes() == []


# ── calculate_vram ───────────────────────────────────────────────────────


class TestCalculateVram:
    def test_sums_service_vram(self) -> None:
        from gpumod.models import Mode, PresetConfig
        from gpumod.templates.modes import ModeLoader

        loader = ModeLoader()
        mode = Mode(id="t", name="T", services=["a", "b"])
        presets = [
            PresetConfig(id="a", name="A", driver="vllm", vram_mb=5000),
            PresetConfig(id="b", name="B", driver="fastapi", vram_mb=3000),
        ]
        result = loader.calculate_vram(mode, presets)
        assert result == 8000

    def test_ignores_unlisted_services(self) -> None:
        from gpumod.models import Mode, PresetConfig
        from gpumod.templates.modes import ModeLoader

        loader = ModeLoader()
        mode = Mode(id="t", name="T", services=["a"])
        presets = [
            PresetConfig(id="a", name="A", driver="vllm", vram_mb=5000),
            PresetConfig(id="b", name="B", driver="fastapi", vram_mb=3000),
        ]
        assert loader.calculate_vram(mode, presets) == 5000

    def test_raises_on_unknown_service(self) -> None:
        from gpumod.models import Mode, PresetConfig
        from gpumod.templates.modes import ModeLoader

        loader = ModeLoader()
        mode = Mode(id="t", name="T", services=["missing"])
        presets = [
            PresetConfig(id="a", name="A", driver="vllm", vram_mb=5000),
        ]
        with pytest.raises(ValueError, match="missing"):
            loader.calculate_vram(mode, presets)


# ── Production modes from modes/ directory ───────────────────────────────


class TestProductionModes:
    """Validate the 6 production mode YAMLs load via ModeLoader."""

    def test_discovers_all_6_production_modes(self) -> None:
        from gpumod.templates.modes import ModeLoader

        loader = ModeLoader(mode_dirs=[MODES_DIR])
        modes = loader.discover_modes()
        ids = {m.id for m in modes}
        assert ids == {"code", "rag", "hacker", "speak", "blank", "finetuning"}

    def test_vram_calculation_with_real_presets(self) -> None:
        from gpumod.templates.modes import ModeLoader
        from gpumod.templates.presets import PresetLoader

        mode_loader = ModeLoader(mode_dirs=[MODES_DIR])
        preset_loader = PresetLoader(preset_dirs=[PRESETS_DIR])
        modes = mode_loader.discover_modes()
        presets = preset_loader.discover_presets()

        code_mode = next(m for m in modes if m.id == "code")
        vram = mode_loader.calculate_vram(code_mode, presets)
        assert vram == 22500

        speak_mode = next(m for m in modes if m.id == "speak")
        vram = mode_loader.calculate_vram(speak_mode, presets)
        assert vram == 22000
