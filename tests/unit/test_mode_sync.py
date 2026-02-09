"""Tests for mode sync logic — AC2/AC3/AC4/AC5 of gpumod-652."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from typer.testing import CliRunner

from gpumod.db import Database
from gpumod.models import Mode
from gpumod.templates.modes import ModeLoader, sync_modes
from gpumod.templates.presets import PresetLoader

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_mode(directory: Path, filename: str, data: dict) -> Path:
    """Write a YAML mode file into a directory, return file path."""
    fp = directory / filename
    fp.write_text(yaml.safe_dump(data), encoding="utf-8")
    return fp


def _write_preset(directory: Path, filename: str, data: dict) -> Path:
    """Write a YAML preset file into a directory, return file path."""
    fp = directory / filename
    fp.write_text(yaml.safe_dump(data), encoding="utf-8")
    return fp


def _minimal_mode(
    id: str = "mode-test",
    name: str = "Test Mode",
    description: str | None = "A test mode",
    services: list[str] | None = None,
) -> dict:
    """Return a minimal valid mode dict."""
    base = {
        "id": id,
        "name": name,
    }
    if description is not None:
        base["description"] = description
    if services is not None:
        base["services"] = services
    return base


def _minimal_preset(
    id: str = "svc-test",
    name: str = "Test",
    driver: str = "vllm",
    port: int = 8000,
    vram_mb: int = 2000,
    **overrides: object,
) -> dict:
    """Return a minimal valid preset dict."""
    base = {
        "id": id,
        "name": name,
        "driver": driver,
        "port": port,
        "vram_mb": vram_mb,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# sync_modes tests — AC2
# ---------------------------------------------------------------------------


class TestSyncModes:
    """Tests for the sync_modes() function."""

    async def test_inserts_new_modes(self, tmp_path: Path) -> None:
        """sync_modes() should insert modes for mode YAMLs not yet in DB."""
        mode_dir = tmp_path / "modes"
        preset_dir = tmp_path / "presets"
        mode_dir.mkdir()
        preset_dir.mkdir()

        # Create presets for the services referenced by modes
        _write_preset(preset_dir, "svc-a.yaml", _minimal_preset(id="svc-a", vram_mb=1000))
        _write_preset(preset_dir, "svc-b.yaml", _minimal_preset(id="svc-b", vram_mb=2000))

        _write_mode(mode_dir, "a.yaml", _minimal_mode(id="mode-a", name="A", services=["svc-a"]))
        _write_mode(mode_dir, "b.yaml", _minimal_mode(id="mode-b", name="B", services=["svc-b"]))

        mode_loader = ModeLoader(mode_dirs=[mode_dir])
        preset_loader = PresetLoader(preset_dirs=[preset_dir])

        async with Database(tmp_path / "test.db") as db:
            # Insert services first (presets must be synced before modes)
            from gpumod.templates.presets import sync_presets

            await sync_presets(db, preset_loader)

            result = await sync_modes(db, mode_loader)

            assert result.inserted == 2
            assert result.updated == 0
            assert result.unchanged == 0

            # Verify in DB
            assert await db.get_mode("mode-a") is not None
            assert await db.get_mode("mode-b") is not None

    async def test_updates_changed_modes(self, tmp_path: Path) -> None:
        """sync_modes() should update modes whose YAML fields changed."""
        mode_dir = tmp_path / "modes"
        preset_dir = tmp_path / "presets"
        mode_dir.mkdir()
        preset_dir.mkdir()

        _write_preset(preset_dir, "svc-a.yaml", _minimal_preset(id="svc-a", vram_mb=1000))
        mode = _minimal_mode(id="test-mode", name="Original", services=["svc-a"])
        _write_mode(mode_dir, "test.yaml", mode)

        mode_loader = ModeLoader(mode_dirs=[mode_dir])
        preset_loader = PresetLoader(preset_dirs=[preset_dir])

        async with Database(tmp_path / "test.db") as db:
            from gpumod.templates.presets import sync_presets

            await sync_presets(db, preset_loader)
            await sync_modes(db, mode_loader)

            got = await db.get_mode("test-mode")
            assert got is not None
            assert got.name == "Original"

            # Change the YAML
            updated = _minimal_mode(id="test-mode", name="Updated", services=["svc-a"])
            _write_mode(mode_dir, "test.yaml", updated)

            mode_loader2 = ModeLoader(mode_dirs=[mode_dir])
            result = await sync_modes(db, mode_loader2)

            assert result.updated == 1
            assert result.inserted == 0
            assert result.unchanged == 0

            got = await db.get_mode("test-mode")
            assert got is not None
            assert got.name == "Updated"

    async def test_skips_unchanged_modes(self, tmp_path: Path) -> None:
        """sync_modes() should skip modes that match the DB exactly."""
        mode_dir = tmp_path / "modes"
        preset_dir = tmp_path / "presets"
        mode_dir.mkdir()
        preset_dir.mkdir()

        _write_preset(preset_dir, "svc-a.yaml", _minimal_preset(id="svc-a", vram_mb=1000))
        _write_mode(
            mode_dir, "test.yaml", _minimal_mode(id="test-mode", name="Test", services=["svc-a"])
        )

        mode_loader = ModeLoader(mode_dirs=[mode_dir])
        preset_loader = PresetLoader(preset_dirs=[preset_dir])

        async with Database(tmp_path / "test.db") as db:
            from gpumod.templates.presets import sync_presets

            await sync_presets(db, preset_loader)

            # First sync
            await sync_modes(db, mode_loader)

            # Second sync — no changes
            result = await sync_modes(db, mode_loader)

            assert result.inserted == 0
            assert result.updated == 0
            assert result.unchanged == 1

    async def test_idempotent_multiple_runs(self, tmp_path: Path) -> None:
        """Running sync_modes() multiple times should be idempotent (AC5)."""
        mode_dir = tmp_path / "modes"
        preset_dir = tmp_path / "presets"
        mode_dir.mkdir()
        preset_dir.mkdir()

        _write_preset(preset_dir, "svc.yaml", _minimal_preset(id="svc", vram_mb=1000))
        _write_mode(mode_dir, "x.yaml", _minimal_mode(id="mode-x", services=["svc"]))

        mode_loader = ModeLoader(mode_dirs=[mode_dir])
        preset_loader = PresetLoader(preset_dirs=[preset_dir])

        async with Database(tmp_path / "test.db") as db:
            from gpumod.templates.presets import sync_presets

            await sync_presets(db, preset_loader)

            r1 = await sync_modes(db, mode_loader)
            assert r1.inserted == 1

            r2 = await sync_modes(db, mode_loader)
            assert r2.inserted == 0
            assert r2.unchanged == 1

            r3 = await sync_modes(db, mode_loader)
            assert r3.inserted == 0
            assert r3.unchanged == 1

            # DB should have exactly one mode
            modes = await db.list_modes()
            assert len(modes) == 1

    async def test_no_modes_returns_all_zeros(self, tmp_path: Path) -> None:
        """sync_modes() with no mode files returns zeroed result."""
        mode_dir = tmp_path / "modes"
        mode_dir.mkdir()

        mode_loader = ModeLoader(mode_dirs=[mode_dir])

        async with Database(tmp_path / "test.db") as db:
            result = await sync_modes(db, mode_loader)

            assert result.inserted == 0
            assert result.updated == 0
            assert result.unchanged == 0
            assert result.deleted == 0


# ---------------------------------------------------------------------------
# Junction table tests — AC3
# ---------------------------------------------------------------------------


class TestSyncModesJunctionTable:
    """Tests for junction table (mode_services) updates — AC3."""

    async def test_updates_junction_table_when_services_change(self, tmp_path: Path) -> None:
        """When a mode YAML service list changes, the junction table should be updated."""
        mode_dir = tmp_path / "modes"
        preset_dir = tmp_path / "presets"
        mode_dir.mkdir()
        preset_dir.mkdir()

        _write_preset(preset_dir, "a.yaml", _minimal_preset(id="svc-a", vram_mb=1000))
        _write_preset(preset_dir, "b.yaml", _minimal_preset(id="svc-b", vram_mb=2000))
        _write_preset(preset_dir, "c.yaml", _minimal_preset(id="svc-c", vram_mb=3000))

        # Mode initially has [svc-a, svc-b]
        _write_mode(
            mode_dir, "test.yaml", _minimal_mode(id="test-mode", services=["svc-a", "svc-b"])
        )

        mode_loader = ModeLoader(mode_dirs=[mode_dir])
        preset_loader = PresetLoader(preset_dirs=[preset_dir])

        async with Database(tmp_path / "test.db") as db:
            from gpumod.templates.presets import sync_presets

            await sync_presets(db, preset_loader)
            await sync_modes(db, mode_loader)

            mode = await db.get_mode("test-mode")
            assert mode is not None
            assert mode.services == ["svc-a", "svc-b"]

            # Change to [svc-a, svc-c]
            _write_mode(
                mode_dir, "test.yaml", _minimal_mode(id="test-mode", services=["svc-a", "svc-c"])
            )

            mode_loader2 = ModeLoader(mode_dirs=[mode_dir])
            result = await sync_modes(db, mode_loader2)

            assert result.updated == 1

            mode = await db.get_mode("test-mode")
            assert mode is not None
            assert mode.services == ["svc-a", "svc-c"]

    async def test_updates_service_order_in_junction_table(self, tmp_path: Path) -> None:
        """When service order changes in YAML, junction table start_order should update."""
        mode_dir = tmp_path / "modes"
        preset_dir = tmp_path / "presets"
        mode_dir.mkdir()
        preset_dir.mkdir()

        _write_preset(preset_dir, "a.yaml", _minimal_preset(id="svc-a", vram_mb=1000))
        _write_preset(preset_dir, "b.yaml", _minimal_preset(id="svc-b", vram_mb=2000))
        _write_preset(preset_dir, "c.yaml", _minimal_preset(id="svc-c", vram_mb=3000))

        # Mode initially has [a, b, c]
        _write_mode(
            mode_dir,
            "test.yaml",
            _minimal_mode(id="test-mode", services=["svc-a", "svc-b", "svc-c"]),
        )

        mode_loader = ModeLoader(mode_dirs=[mode_dir])
        preset_loader = PresetLoader(preset_dirs=[preset_dir])

        async with Database(tmp_path / "test.db") as db:
            from gpumod.templates.presets import sync_presets

            await sync_presets(db, preset_loader)
            await sync_modes(db, mode_loader)

            mode = await db.get_mode("test-mode")
            assert mode is not None
            assert mode.services == ["svc-a", "svc-b", "svc-c"]

            # Change order to [c, a, b]
            _write_mode(
                mode_dir,
                "test.yaml",
                _minimal_mode(id="test-mode", services=["svc-c", "svc-a", "svc-b"]),
            )

            mode_loader2 = ModeLoader(mode_dirs=[mode_dir])
            result = await sync_modes(db, mode_loader2)

            assert result.updated == 1

            mode = await db.get_mode("test-mode")
            assert mode is not None
            assert mode.services == ["svc-c", "svc-a", "svc-b"]

    async def test_empty_services_list_syncs_correctly(self, tmp_path: Path) -> None:
        """A mode with services: [] should sync correctly with empty junction table."""
        mode_dir = tmp_path / "modes"
        mode_dir.mkdir()

        _write_mode(mode_dir, "blank.yaml", _minimal_mode(id="blank-mode", services=[]))

        mode_loader = ModeLoader(mode_dirs=[mode_dir])

        async with Database(tmp_path / "test.db") as db:
            result = await sync_modes(db, mode_loader)
            assert result.inserted == 1

            mode = await db.get_mode("blank-mode")
            assert mode is not None
            assert mode.services == []


# ---------------------------------------------------------------------------
# VRAM recalculation tests — AC4
# ---------------------------------------------------------------------------


class TestSyncModesVRAM:
    """Tests for VRAM recalculation — AC4."""

    async def test_calculates_total_vram_from_services(self, tmp_path: Path) -> None:
        """sync_modes() should calculate total_vram_mb from service VRAM values."""
        mode_dir = tmp_path / "modes"
        preset_dir = tmp_path / "presets"
        mode_dir.mkdir()
        preset_dir.mkdir()

        _write_preset(preset_dir, "a.yaml", _minimal_preset(id="svc-a", vram_mb=1000))
        _write_preset(preset_dir, "b.yaml", _minimal_preset(id="svc-b", vram_mb=2500))

        _write_mode(
            mode_dir, "test.yaml", _minimal_mode(id="test-mode", services=["svc-a", "svc-b"])
        )

        mode_loader = ModeLoader(mode_dirs=[mode_dir])
        preset_loader = PresetLoader(preset_dirs=[preset_dir])

        async with Database(tmp_path / "test.db") as db:
            from gpumod.templates.presets import sync_presets

            await sync_presets(db, preset_loader)
            await sync_modes(db, mode_loader)

            mode = await db.get_mode("test-mode")
            assert mode is not None
            assert mode.total_vram_mb == 3500  # 1000 + 2500

    async def test_recalculates_vram_on_update(self, tmp_path: Path) -> None:
        """When mode services change, total_vram_mb should be recalculated."""
        mode_dir = tmp_path / "modes"
        preset_dir = tmp_path / "presets"
        mode_dir.mkdir()
        preset_dir.mkdir()

        _write_preset(preset_dir, "a.yaml", _minimal_preset(id="svc-a", vram_mb=1000))
        _write_preset(preset_dir, "b.yaml", _minimal_preset(id="svc-b", vram_mb=2000))
        _write_preset(preset_dir, "c.yaml", _minimal_preset(id="svc-c", vram_mb=5000))

        _write_mode(
            mode_dir, "test.yaml", _minimal_mode(id="test-mode", services=["svc-a", "svc-b"])
        )

        mode_loader = ModeLoader(mode_dirs=[mode_dir])
        preset_loader = PresetLoader(preset_dirs=[preset_dir])

        async with Database(tmp_path / "test.db") as db:
            from gpumod.templates.presets import sync_presets

            await sync_presets(db, preset_loader)
            await sync_modes(db, mode_loader)

            mode = await db.get_mode("test-mode")
            assert mode is not None
            assert mode.total_vram_mb == 3000

            # Change to [svc-c] only
            _write_mode(mode_dir, "test.yaml", _minimal_mode(id="test-mode", services=["svc-c"]))

            mode_loader2 = ModeLoader(mode_dirs=[mode_dir])
            await sync_modes(db, mode_loader2)

            mode = await db.get_mode("test-mode")
            assert mode is not None
            assert mode.total_vram_mb == 5000

    async def test_empty_services_has_zero_vram(self, tmp_path: Path) -> None:
        """A mode with no services should have total_vram_mb = 0."""
        mode_dir = tmp_path / "modes"
        mode_dir.mkdir()

        _write_mode(mode_dir, "blank.yaml", _minimal_mode(id="blank-mode", services=[]))

        mode_loader = ModeLoader(mode_dirs=[mode_dir])

        async with Database(tmp_path / "test.db") as db:
            await sync_modes(db, mode_loader)

            mode = await db.get_mode("blank-mode")
            assert mode is not None
            assert mode.total_vram_mb == 0


# ---------------------------------------------------------------------------
# Edge case tests — deletion, orphans, etc.
# ---------------------------------------------------------------------------


class TestSyncModesEdgeCases:
    """Edge cases for sync_modes(): deletions, orphans, missing services, etc."""

    async def test_deleted_yaml_removes_mode_from_db(self, tmp_path: Path) -> None:
        """If a mode YAML file is removed, sync should delete the mode from DB."""
        mode_dir = tmp_path / "modes"
        preset_dir = tmp_path / "presets"
        mode_dir.mkdir()
        preset_dir.mkdir()

        _write_preset(preset_dir, "svc.yaml", _minimal_preset(id="svc", vram_mb=1000))
        fp = _write_mode(mode_dir, "old.yaml", _minimal_mode(id="old-mode", services=["svc"]))
        _write_mode(mode_dir, "keep.yaml", _minimal_mode(id="keep-mode", services=["svc"]))

        mode_loader = ModeLoader(mode_dirs=[mode_dir])
        preset_loader = PresetLoader(preset_dirs=[preset_dir])

        async with Database(tmp_path / "test.db") as db:
            from gpumod.templates.presets import sync_presets

            await sync_presets(db, preset_loader)
            await sync_modes(db, mode_loader)

            assert await db.get_mode("old-mode") is not None
            assert await db.get_mode("keep-mode") is not None

            # Remove old.yaml
            fp.unlink()
            mode_loader2 = ModeLoader(mode_dirs=[mode_dir])
            result = await sync_modes(db, mode_loader2)

            assert result.deleted == 1
            assert result.unchanged == 1
            assert await db.get_mode("old-mode") is None
            assert await db.get_mode("keep-mode") is not None

    async def test_mode_references_nonexistent_service_warns(self, tmp_path: Path) -> None:
        """If mode YAML references a nonexistent service, sync should warn and skip it."""
        mode_dir = tmp_path / "modes"
        preset_dir = tmp_path / "presets"
        mode_dir.mkdir()
        preset_dir.mkdir()

        _write_preset(preset_dir, "svc-a.yaml", _minimal_preset(id="svc-a", vram_mb=1000))
        # Mode references svc-a (exists) and svc-missing (doesn't exist)
        _write_mode(
            mode_dir, "test.yaml", _minimal_mode(id="test-mode", services=["svc-a", "svc-missing"])
        )

        mode_loader = ModeLoader(mode_dirs=[mode_dir])
        preset_loader = PresetLoader(preset_dirs=[preset_dir])

        async with Database(tmp_path / "test.db") as db:
            from gpumod.templates.presets import sync_presets

            await sync_presets(db, preset_loader)

            # Should succeed with a warning, only adding svc-a to junction table
            result = await sync_modes(db, mode_loader)

            assert result.inserted == 1
            assert len(result.warnings) > 0  # Should have a warning about missing service

            mode = await db.get_mode("test-mode")
            assert mode is not None
            # Only svc-a should be in the junction table
            assert mode.services == ["svc-a"]

    async def test_manual_db_mode_not_deleted(self, tmp_path: Path) -> None:
        """Modes added manually via CLI (not from YAML) should NOT be deleted by sync."""
        mode_dir = tmp_path / "modes"
        mode_dir.mkdir()

        # No mode YAMLs exist

        mode_loader = ModeLoader(mode_dirs=[mode_dir])

        async with Database(tmp_path / "test.db") as db:
            # Insert a mode manually (simulating CLI create)
            manual = Mode(
                id="manual-mode",
                name="Manual",
                description="Created via CLI",
                total_vram_mb=0,
            )
            await db.insert_mode(manual)

            result = await sync_modes(db, mode_loader)

            # Manual mode should not be deleted (no YAML = no deletion)
            assert result.deleted == 0
            assert await db.get_mode("manual-mode") is not None

    async def test_deleted_yaml_while_mode_active_does_not_clear_current(
        self, tmp_path: Path
    ) -> None:
        """If current_mode points to a deleted mode, sync should NOT clear current_mode."""
        mode_dir = tmp_path / "modes"
        preset_dir = tmp_path / "presets"
        mode_dir.mkdir()
        preset_dir.mkdir()

        _write_preset(preset_dir, "svc.yaml", _minimal_preset(id="svc", vram_mb=1000))
        _write_preset(preset_dir, "svc2.yaml", _minimal_preset(id="svc2", vram_mb=500))
        fp = _write_mode(
            mode_dir, "active.yaml", _minimal_mode(id="active-mode", services=["svc"])
        )
        # Keep at least one mode YAML so deletion logic triggers
        _write_mode(mode_dir, "other.yaml", _minimal_mode(id="other-mode", services=["svc2"]))

        mode_loader = ModeLoader(mode_dirs=[mode_dir])
        preset_loader = PresetLoader(preset_dirs=[preset_dir])

        async with Database(tmp_path / "test.db") as db:
            from gpumod.templates.presets import sync_presets

            await sync_presets(db, preset_loader)
            await sync_modes(db, mode_loader)

            # Set as current mode
            await db.set_current_mode("active-mode")

            # Remove active-mode YAML (but other.yaml still exists)
            fp.unlink()
            mode_loader2 = ModeLoader(mode_dirs=[mode_dir])
            await sync_modes(db, mode_loader2)

            # Mode should be deleted from DB
            assert await db.get_mode("active-mode") is None

            # But current_mode setting should NOT be cleared
            current = await db.get_current_mode()
            assert current == "active-mode"  # Stale but preserved


# ---------------------------------------------------------------------------
# CLI command tests — AC6
# ---------------------------------------------------------------------------


class TestModeSyncCLI:
    """Tests for the 'gpumod mode sync' CLI command — AC6."""

    def test_mode_sync_command_exists(self) -> None:
        """The mode sync subcommand should be registered."""
        from gpumod.cli import app

        result = runner.invoke(app, ["mode", "sync", "--help"])
        assert result.exit_code == 0
        assert "sync" in result.output.lower()

    def test_mode_sync_runs_and_reports(self, tmp_path: Path) -> None:
        """Running 'mode sync' should report inserted/updated/unchanged/deleted."""
        from gpumod.cli import app

        db_path = tmp_path / "test.db"
        result = runner.invoke(app, ["mode", "sync", "--db-path", str(db_path)])
        assert result.exit_code == 0
        assert "inserted" in result.output
        assert "updated" in result.output
        assert "unchanged" in result.output
