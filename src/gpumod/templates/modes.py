"""YAML mode loader for gpumod mode definitions.

Loads YAML mode files and converts them into Mode objects.
Uses ``yaml.safe_load()`` exclusively for security.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from gpumod.models import Mode

if TYPE_CHECKING:
    from pathlib import Path

    from gpumod.models import PresetConfig

_YAML_EXTENSIONS = frozenset({".yaml", ".yml"})


class ModeLoader:
    """Loads YAML mode files into Mode objects.

    Security:
    - Uses ``yaml.safe_load()`` exclusively (never ``yaml.load()``).
    """

    def __init__(self, mode_dirs: list[Path] | None = None) -> None:
        self._mode_dirs = mode_dirs if mode_dirs is not None else []

    def load_file(self, path: Path) -> Mode:
        """Load a single YAML mode file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML is not a mapping.
            yaml.YAMLError: If the YAML is malformed.
            pydantic.ValidationError: If the data fails validation.
        """
        resolved = path.resolve()
        if not resolved.is_file():
            msg = f"Mode file not found: {path}"
            raise FileNotFoundError(msg)

        raw_text = resolved.read_text(encoding="utf-8")
        data: Any = yaml.safe_load(raw_text)

        if not isinstance(data, dict):
            msg = f"Mode file must contain a YAML mapping, got: {type(data).__name__}"
            raise ValueError(msg)

        return Mode(**data)

    def load_directory(self, directory: Path) -> list[Mode]:
        """Load all YAML mode files from a directory.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        resolved = directory.resolve()
        if not resolved.is_dir():
            msg = f"Mode directory not found: {directory}"
            raise FileNotFoundError(msg)

        modes = [
            self.load_file(yaml_file)
            for yaml_file in sorted(resolved.rglob("*"))
            if yaml_file.is_file() and yaml_file.suffix in _YAML_EXTENSIONS
        ]
        return sorted(modes, key=lambda m: m.id)

    def discover_modes(self) -> list[Mode]:
        """Discover all modes from configured directories."""
        all_modes: list[Mode] = []
        for directory in self._mode_dirs:
            resolved = directory.resolve()
            if resolved.is_dir():
                all_modes.extend(self.load_directory(resolved))
        return sorted(all_modes, key=lambda m: m.id)

    @staticmethod
    def calculate_vram(mode: Mode, presets: list[PresetConfig]) -> int:
        """Calculate total VRAM for a mode from its service presets.

        Raises:
            ValueError: If a service ID in the mode is not found in presets.
        """
        preset_map = {p.id: p.vram_mb for p in presets}
        total = 0
        for svc_id in mode.services:
            if svc_id not in preset_map:
                msg = f"Mode '{mode.id}' references unknown service '{svc_id}'"
                raise ValueError(msg)
            total += preset_map[svc_id]
        return total
