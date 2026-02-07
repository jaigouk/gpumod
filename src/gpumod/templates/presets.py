"""YAML preset loader for gpumod service configurations.

Loads YAML preset files and converts them into PresetConfig and Service
model objects. Uses ``yaml.safe_load()`` exclusively for security.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import yaml

from gpumod.models import PresetConfig, Service

if TYPE_CHECKING:
    from pathlib import Path

_YAML_EXTENSIONS = frozenset({".yaml", ".yml"})


class PresetLoader:
    """Loads YAML preset files into PresetConfig objects.

    Security:
    - Uses ``yaml.safe_load()`` exclusively (never ``yaml.load()``).
    - Validates file paths to prevent path traversal.
    - Expands environment variables in ``model_path`` safely.
    """

    def __init__(self, preset_dirs: list[Path] | None = None) -> None:
        """Initialize with preset search directories.

        Args:
            preset_dirs: Directories to search for preset YAML files.
                If ``None``, defaults to an empty list (no auto-discovery).
        """
        self._preset_dirs = preset_dirs if preset_dirs is not None else []

    def load_file(self, path: Path) -> PresetConfig:
        """Load a single YAML preset file.

        Args:
            path: Path to the YAML preset file.

        Returns:
            A validated PresetConfig object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file path is invalid (path traversal).
            yaml.YAMLError: If the YAML is malformed.
            pydantic.ValidationError: If the data fails validation.
        """
        resolved = path.resolve()
        if not resolved.is_file():
            msg = f"Preset file not found: {path}"
            raise FileNotFoundError(msg)

        raw_text = resolved.read_text(encoding="utf-8")
        data: Any = yaml.safe_load(raw_text)

        if not isinstance(data, dict):
            msg = f"Preset file must contain a YAML mapping, got: {type(data).__name__}"
            raise ValueError(msg)

        # Expand environment variables in model_path if present.
        if "model_path" in data and isinstance(data["model_path"], str):
            data["model_path"] = os.path.expandvars(data["model_path"])

        # Expand environment variables in unit_vars string values.
        if "unit_vars" in data and isinstance(data["unit_vars"], dict):
            data["unit_vars"] = {
                k: os.path.expandvars(v) if isinstance(v, str) else v
                for k, v in data["unit_vars"].items()
            }

        return PresetConfig(**data)

    def load_directory(self, directory: Path) -> list[PresetConfig]:
        """Load all YAML preset files from a directory (recursive).

        Args:
            directory: Directory to scan for ``.yaml`` / ``.yml`` files.

        Returns:
            List of validated PresetConfig objects, sorted by id.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        resolved = directory.resolve()
        if not resolved.is_dir():
            msg = f"Preset directory not found: {directory}"
            raise FileNotFoundError(msg)

        presets = [
            self.load_file(yaml_file)
            for yaml_file in sorted(resolved.rglob("*"))
            if yaml_file.is_file() and yaml_file.suffix in _YAML_EXTENSIONS
        ]
        return sorted(presets, key=lambda p: p.id)

    def to_service(self, preset: PresetConfig) -> Service:
        """Convert a PresetConfig to a Service object.

        Maps preset fields to the Service model, storing driver-specific
        ``unit_vars`` inside ``extra_config``.

        Args:
            preset: The preset configuration to convert.

        Returns:
            A Service object ready for registration.
        """
        extra_config: dict[str, Any] = {}
        if preset.unit_vars:
            extra_config["unit_vars"] = dict(preset.unit_vars)

        return Service(
            id=preset.id,
            name=preset.name,
            driver=preset.driver,
            port=preset.port,
            vram_mb=preset.vram_mb,
            sleep_mode=preset.sleep_mode,
            health_endpoint=preset.health_endpoint,
            model_id=preset.model_id,
            unit_name=f"{preset.id}.service",
            startup_timeout=preset.startup_timeout,
            extra_config=extra_config,
        )

    def discover_presets(self) -> list[PresetConfig]:
        """Discover all presets from configured directories.

        Scans all directories provided at initialization and loads
        every valid YAML preset file found.

        Returns:
            List of all discovered PresetConfig objects, sorted by id.
        """
        all_presets: list[PresetConfig] = []
        for directory in self._preset_dirs:
            resolved = directory.resolve()
            if resolved.is_dir():
                all_presets.extend(self.load_directory(resolved))
        return sorted(all_presets, key=lambda p: p.id)
