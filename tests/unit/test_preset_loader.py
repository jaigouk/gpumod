"""Tests for the gpumod YAML preset loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

from gpumod.models import DriverType, SleepMode

if TYPE_CHECKING:
    from pathlib import Path


# ── Helper fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def valid_preset_yaml() -> str:
    """A valid YAML preset string."""
    return """\
id: devstral-small
name: Devstral Small 2505
driver: vllm
port: 8000
vram_mb: 15000
context_size: 32768
kv_cache_per_1k: 64
model_id: mistralai/Devstral-Small-2505
health_endpoint: /health
startup_timeout: 120
supports_sleep: true
sleep_mode: l1
unit_vars:
  gpu_mem_util: 0.9
  max_model_len: 32768
"""


@pytest.fixture
def valid_preset_file(tmp_path: Path, valid_preset_yaml: str) -> Path:
    """Create a valid YAML preset file on disk."""
    p = tmp_path / "devstral.yaml"
    p.write_text(valid_preset_yaml)
    return p


@pytest.fixture
def preset_directory(tmp_path: Path) -> Path:
    """Create a directory with multiple preset files."""
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()

    # vllm preset
    (llm_dir / "devstral.yaml").write_text(
        yaml.dump(
            {
                "id": "devstral",
                "name": "Devstral",
                "driver": "vllm",
                "port": 8000,
                "vram_mb": 15000,
                "model_id": "mistralai/Devstral-Small-2505",
            }
        )
    )

    # llamacpp preset (yml extension)
    (llm_dir / "codellama.yml").write_text(
        yaml.dump(
            {
                "id": "codellama",
                "name": "CodeLlama",
                "driver": "llamacpp",
                "port": 8080,
                "vram_mb": 6000,
                "model_path": "/models/codellama.gguf",
            }
        )
    )

    embed_dir = tmp_path / "embedding"
    embed_dir.mkdir()

    # fastapi preset in subdirectory
    (embed_dir / "nomic.yaml").write_text(
        yaml.dump(
            {
                "id": "nomic-embed",
                "name": "Nomic Embed",
                "driver": "fastapi",
                "port": 9000,
                "vram_mb": 2000,
            }
        )
    )

    # Non-yaml file should be ignored
    (llm_dir / "README.md").write_text("# Not a preset")

    return tmp_path


# ── load_file: valid YAML ───────────────────────────────────────────────


class TestLoadFile:
    def test_parses_valid_yaml(self, valid_preset_file: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        preset = loader.load_file(valid_preset_file)
        assert preset.id == "devstral-small"
        assert preset.name == "Devstral Small 2505"
        assert preset.driver == DriverType.VLLM
        assert preset.port == 8000
        assert preset.vram_mb == 15000

    def test_parses_all_fields(self, valid_preset_file: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        preset = loader.load_file(valid_preset_file)
        assert preset.context_size == 32768
        assert preset.kv_cache_per_1k == 64
        assert preset.model_id == "mistralai/Devstral-Small-2505"
        assert preset.health_endpoint == "/health"
        assert preset.startup_timeout == 120
        assert preset.supports_sleep is True
        assert preset.sleep_mode == SleepMode.L1
        assert preset.unit_vars == {"gpu_mem_util": 0.9, "max_model_len": 32768}

    def test_raises_on_missing_required_fields(self, tmp_path: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        # Missing 'id' and 'vram_mb'
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("name: Bad\ndriver: vllm\n")
        loader = PresetLoader()
        with pytest.raises(ValidationError):
            loader.load_file(bad_file)

    def test_raises_on_invalid_yaml(self, tmp_path: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        bad_file = tmp_path / "invalid.yaml"
        bad_file.write_text("{{not yaml}}: [broken\n")
        loader = PresetLoader()
        with pytest.raises((yaml.YAMLError, ValueError)):
            loader.load_file(bad_file)

    def test_raises_on_nonexistent_file(self, tmp_path: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file(tmp_path / "nonexistent.yaml")

    def test_raises_on_extra_fields(self, tmp_path: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        bad_file = tmp_path / "extra.yaml"
        bad_file.write_text(
            yaml.dump(
                {
                    "id": "test",
                    "name": "Test",
                    "driver": "vllm",
                    "vram_mb": 4096,
                    "unknown_field": "should_fail",
                }
            )
        )
        loader = PresetLoader()
        with pytest.raises(ValidationError):
            loader.load_file(bad_file)


# ── load_directory ──────────────────────────────────────────────────────


class TestLoadDirectory:
    def test_loads_all_yaml_files(self, preset_directory: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        presets = loader.load_directory(preset_directory)
        ids = {p.id for p in presets}
        assert "devstral" in ids
        assert "codellama" in ids
        assert "nomic-embed" in ids

    def test_loads_yml_extension(self, preset_directory: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        presets = loader.load_directory(preset_directory)
        ids = {p.id for p in presets}
        assert "codellama" in ids  # The .yml file

    def test_ignores_non_yaml_files(self, preset_directory: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        presets = loader.load_directory(preset_directory)
        # Should only have 3 presets, not 4 (README.md ignored)
        assert len(presets) == 3

    def test_empty_directory(self, tmp_path: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        empty = tmp_path / "empty"
        empty.mkdir()
        loader = PresetLoader()
        presets = loader.load_directory(empty)
        assert presets == []

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_directory(tmp_path / "missing")


# ── to_service ──────────────────────────────────────────────────────────


class TestToService:
    def test_converts_basic_fields(self, valid_preset_file: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        preset = loader.load_file(valid_preset_file)
        svc = loader.to_service(preset)
        assert svc.id == "devstral-small"
        assert svc.name == "Devstral Small 2505"
        assert svc.driver == DriverType.VLLM
        assert svc.port == 8000
        assert svc.vram_mb == 15000

    def test_maps_sleep_mode(self, valid_preset_file: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        preset = loader.load_file(valid_preset_file)
        svc = loader.to_service(preset)
        assert svc.sleep_mode == SleepMode.L1

    def test_maps_health_endpoint(self, valid_preset_file: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        preset = loader.load_file(valid_preset_file)
        svc = loader.to_service(preset)
        assert svc.health_endpoint == "/health"

    def test_maps_model_id(self, valid_preset_file: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        preset = loader.load_file(valid_preset_file)
        svc = loader.to_service(preset)
        assert svc.model_id == "mistralai/Devstral-Small-2505"

    def test_maps_startup_timeout(self, valid_preset_file: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        preset = loader.load_file(valid_preset_file)
        svc = loader.to_service(preset)
        assert svc.startup_timeout == 120

    def test_maps_unit_vars_to_extra_config(self, valid_preset_file: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        preset = loader.load_file(valid_preset_file)
        svc = loader.to_service(preset)
        assert svc.extra_config.get("unit_vars") == {
            "gpu_mem_util": 0.9,
            "max_model_len": 32768,
        }

    def test_minimal_preset_conversion(self, tmp_path: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        f = tmp_path / "minimal.yaml"
        f.write_text(
            yaml.dump(
                {
                    "id": "minimal",
                    "name": "Minimal",
                    "driver": "fastapi",
                    "vram_mb": 0,
                }
            )
        )
        loader = PresetLoader()
        preset = loader.load_file(f)
        svc = loader.to_service(preset)
        assert svc.id == "minimal"
        assert svc.sleep_mode == SleepMode.NONE
        assert svc.port is None

    def test_sets_unit_name_from_id(self, valid_preset_file: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        preset = loader.load_file(valid_preset_file)
        svc = loader.to_service(preset)
        assert svc.unit_name == "devstral-small.service"

    def test_minimal_preset_gets_unit_name(self, tmp_path: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        f = tmp_path / "minimal.yaml"
        f.write_text(
            yaml.dump(
                {
                    "id": "my-svc",
                    "name": "My Service",
                    "driver": "fastapi",
                    "vram_mb": 0,
                }
            )
        )
        loader = PresetLoader()
        preset = loader.load_file(f)
        svc = loader.to_service(preset)
        assert svc.unit_name == "my-svc.service"


# ── discover_presets ────────────────────────────────────────────────────


class TestDiscoverPresets:
    def test_discovers_from_configured_dirs(self, preset_directory: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader(preset_dirs=[preset_directory])
        presets = loader.discover_presets()
        assert len(presets) == 3

    def test_discovers_from_multiple_dirs(self, tmp_path: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        (dir1 / "a.yaml").write_text(
            yaml.dump({"id": "a", "name": "A", "driver": "vllm", "vram_mb": 4096})
        )

        dir2 = tmp_path / "dir2"
        dir2.mkdir()
        (dir2 / "b.yaml").write_text(
            yaml.dump({"id": "b", "name": "B", "driver": "fastapi", "vram_mb": 0})
        )

        loader = PresetLoader(preset_dirs=[dir1, dir2])
        presets = loader.discover_presets()
        ids = {p.id for p in presets}
        assert ids == {"a", "b"}

    def test_empty_when_no_dirs_configured(self) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader(preset_dirs=[])
        presets = loader.discover_presets()
        assert presets == []


# ── Security tests ──────────────────────────────────────────────────────


class TestPresetSecurity:
    def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        from gpumod.templates.presets import PresetLoader

        loader = PresetLoader()
        traversal = tmp_path / ".." / ".." / "etc" / "passwd"
        with pytest.raises((ValueError, FileNotFoundError)):
            loader.load_file(traversal)

    def test_uses_safe_load(self, tmp_path: Path) -> None:
        """Ensure YAML loading uses safe_load (no arbitrary code execution)."""
        from gpumod.templates.presets import PresetLoader

        # Create a YAML file with a Python object tag (should fail with safe_load)
        evil_file = tmp_path / "evil.yaml"
        evil_file.write_text("!!python/object/apply:os.system ['echo pwned']")
        loader = PresetLoader()
        with pytest.raises(yaml.YAMLError):
            loader.load_file(evil_file)

    def test_env_var_expansion_in_model_path(self, tmp_path: Path) -> None:
        """Environment variables in model_path should be expanded."""
        import os

        from gpumod.templates.presets import PresetLoader

        os.environ["GPUMOD_TEST_MODELS"] = "/data/models"
        try:
            f = tmp_path / "envvar.yaml"
            f.write_text(
                yaml.dump(
                    {
                        "id": "env-test",
                        "name": "Env Test",
                        "driver": "llamacpp",
                        "vram_mb": 6000,
                        "model_path": "$GPUMOD_TEST_MODELS/code.gguf",
                    }
                )
            )
            loader = PresetLoader()
            preset = loader.load_file(f)
            assert preset.model_path == "/data/models/code.gguf"
        finally:
            del os.environ["GPUMOD_TEST_MODELS"]

    def test_env_var_expansion_in_unit_vars(self, tmp_path: Path) -> None:
        """Environment variables in unit_vars string values should be expanded."""
        import os

        from gpumod.templates.presets import PresetLoader

        os.environ["GPUMOD_TEST_HOME"] = "/home/testuser"
        try:
            f = tmp_path / "unitvar.yaml"
            f.write_text(
                yaml.dump(
                    {
                        "id": "unitvar-test",
                        "name": "UnitVar Test",
                        "driver": "llamacpp",
                        "vram_mb": 20000,
                        "unit_vars": {
                            "models_dir": "$GPUMOD_TEST_HOME/bin",
                            "models_max": 1,
                            "jinja": True,
                        },
                    }
                )
            )
            loader = PresetLoader()
            preset = loader.load_file(f)
            assert preset.unit_vars["models_dir"] == "/home/testuser/bin"
            assert preset.unit_vars["models_max"] == 1
            assert preset.unit_vars["jinja"] is True
        finally:
            del os.environ["GPUMOD_TEST_HOME"]
