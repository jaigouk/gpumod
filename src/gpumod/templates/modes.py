"""YAML mode loader for gpumod mode definitions.

Loads YAML mode files and converts them into Mode objects.
Uses ``yaml.safe_load()`` exclusively for security.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

from gpumod.models import Mode

if TYPE_CHECKING:
    from pathlib import Path

    from gpumod.db import Database
    from gpumod.models import PresetConfig

logger = logging.getLogger(__name__)

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

        Files that fail to parse (malformed YAML, validation errors) are
        logged and skipped rather than failing the entire load.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        resolved = directory.resolve()
        if not resolved.is_dir():
            msg = f"Mode directory not found: {directory}"
            raise FileNotFoundError(msg)

        modes: list[Mode] = []
        for yaml_file in sorted(resolved.rglob("*")):
            if yaml_file.is_file() and yaml_file.suffix in _YAML_EXTENSIONS:
                try:
                    modes.append(self.load_file(yaml_file))
                except Exception as exc:
                    logger.warning("Failed to load mode %s: %s", yaml_file.name, exc)
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


@dataclass
class ModeSyncResult:
    """Result of a mode sync operation."""

    inserted: int
    updated: int
    unchanged: int
    deleted: int
    warnings: list[str] = field(default_factory=list)


async def sync_modes(db: Database, loader: ModeLoader) -> ModeSyncResult:
    """Sync YAML modes into the database.

    YAML files are the source of truth. For each mode discovered by *loader*:
    - If it doesn't exist in the DB, insert it.
    - If it exists but fields differ, update it.
    - If it exists and matches, skip it.

    Modes in the DB whose IDs match a known mode naming pattern
    (i.e. they were previously synced from a YAML file) but no longer
    have a corresponding YAML file are deleted.

    The ``total_vram_mb`` is calculated from current DB service VRAM values.
    The ``mode_services`` junction table is updated when service lists change.

    Returns a :class:`ModeSyncResult` with counts.
    """
    modes = loader.discover_modes()
    mode_ids = {m.id for m in modes}

    # Build service VRAM map from DB for VRAM calculation
    services = await db.list_services()
    service_vram = {s.id: s.vram_mb for s in services}
    service_ids_in_db = set(service_vram.keys())

    inserted = 0
    updated = 0
    unchanged = 0
    deleted = 0
    warnings: list[str] = []

    for mode in modes:
        # Filter out services that don't exist in DB, warn about them
        valid_services: list[str] = []
        for svc_id in mode.services:
            if svc_id in service_ids_in_db:
                valid_services.append(svc_id)
            else:
                warnings.append(
                    f"Mode '{mode.id}' references unknown service '{svc_id}' — skipping"
                )

        # Calculate total VRAM from valid services
        total_vram = sum(service_vram.get(sid, 0) for sid in valid_services)

        existing = await db.get_mode(mode.id)

        if existing is None:
            # Insert new mode
            mode_to_insert = Mode(
                id=mode.id,
                name=mode.name,
                description=mode.description,
                services=valid_services,
                total_vram_mb=total_vram,
            )
            await db.insert_mode(mode_to_insert)
            await db.set_mode_services(mode.id, valid_services)
            inserted += 1
        elif _mode_differs(existing, mode, valid_services, total_vram):
            # Update existing mode
            mode_to_update = Mode(
                id=mode.id,
                name=mode.name,
                description=mode.description,
                services=valid_services,
                total_vram_mb=total_vram,
            )
            await db.update_mode(mode_to_update)
            await db.set_mode_services(mode.id, valid_services)
            updated += 1
        else:
            unchanged += 1

    # Delete DB modes that came from YAML but whose YAML was removed.
    # Only delete modes when we have at least one YAML file discovered.
    # If no YAMLs were found, don't delete anything — this preserves
    # manually created modes and handles the case where the modes
    # directory might be temporarily empty/missing.
    if mode_ids:  # Only delete if we found at least one YAML mode
        all_modes = await db.list_modes()
        for m in all_modes:
            if m.id not in mode_ids:
                await db.delete_mode(m.id)
                deleted += 1

    return ModeSyncResult(
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
        deleted=deleted,
        warnings=warnings,
    )


def _mode_differs(
    existing: Mode, new: Mode, valid_services: list[str], total_vram: int
) -> bool:
    """Return True if any mutable field differs between DB mode and YAML mode."""
    return (
        existing.name != new.name
        or existing.description != new.description
        or existing.services != valid_services
        or existing.total_vram_mb != total_vram
    )
