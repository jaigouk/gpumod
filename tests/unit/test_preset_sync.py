"""Tests for preset sync logic — AC2/AC3 of gpumod-7ug."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from typer.testing import CliRunner

from gpumod.db import Database
from gpumod.models import DriverType, Service, SleepMode
from gpumod.templates.presets import PresetLoader, sync_presets

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_preset(directory: Path, filename: str, data: dict) -> Path:
    """Write a YAML preset file into a directory, return file path."""
    fp = directory / filename
    fp.write_text(yaml.safe_dump(data), encoding="utf-8")
    return fp


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
# sync_presets tests
# ---------------------------------------------------------------------------


class TestSyncPresets:
    """Tests for the sync_presets() function."""

    async def test_inserts_new_presets(self, tmp_path: Path) -> None:
        """sync_presets() should insert services for presets not yet in DB."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        _write_preset(preset_dir, "a.yaml", _minimal_preset(id="svc-a", name="A"))
        _write_preset(preset_dir, "b.yaml", _minimal_preset(id="svc-b", name="B"))

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            result = await sync_presets(db, loader)

            assert result.inserted == 2
            assert result.updated == 0
            assert result.unchanged == 0

            # Verify in DB
            assert await db.get_service("svc-a") is not None
            assert await db.get_service("svc-b") is not None

    async def test_updates_changed_presets(self, tmp_path: Path) -> None:
        """sync_presets() should update services whose YAML fields changed."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        _write_preset(
            preset_dir,
            "embed.yaml",
            _minimal_preset(id="embed", name="Embedding", vram_mb=3000),
        )

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            # First sync — inserts
            await sync_presets(db, loader)
            got = await db.get_service("embed")
            assert got is not None
            assert got.vram_mb == 3000

            # Change the YAML
            _write_preset(
                preset_dir,
                "embed.yaml",
                _minimal_preset(id="embed", name="Embedding v2", vram_mb=5000),
            )
            loader2 = PresetLoader(preset_dirs=[preset_dir])
            result = await sync_presets(db, loader2)

            assert result.updated == 1
            assert result.inserted == 0
            assert result.unchanged == 0

            got = await db.get_service("embed")
            assert got is not None
            assert got.name == "Embedding v2"
            assert got.vram_mb == 5000

    async def test_skips_unchanged_presets(self, tmp_path: Path) -> None:
        """sync_presets() should skip presets that match the DB exactly."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        _write_preset(
            preset_dir,
            "chat.yaml",
            _minimal_preset(id="chat", name="Chat"),
        )

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            # First sync
            await sync_presets(db, loader)

            # Second sync — no changes
            result = await sync_presets(db, loader)

            assert result.inserted == 0
            assert result.updated == 0
            assert result.unchanged == 1

    async def test_idempotent_multiple_runs(self, tmp_path: Path) -> None:
        """Running sync_presets() multiple times should be idempotent (AC3)."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        _write_preset(preset_dir, "x.yaml", _minimal_preset(id="svc-x", name="X"))

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            r1 = await sync_presets(db, loader)
            assert r1.inserted == 1

            r2 = await sync_presets(db, loader)
            assert r2.inserted == 0
            assert r2.unchanged == 1

            r3 = await sync_presets(db, loader)
            assert r3.inserted == 0
            assert r3.unchanged == 1

            # DB should have exactly one service
            services = await db.list_services()
            assert len(services) == 1

    async def test_mixed_insert_update_unchanged(self, tmp_path: Path) -> None:
        """sync_presets() handles a mix of new, changed, and unchanged presets."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        _write_preset(preset_dir, "a.yaml", _minimal_preset(id="a", name="A", vram_mb=1000))
        _write_preset(preset_dir, "b.yaml", _minimal_preset(id="b", name="B", vram_mb=2000))

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            await sync_presets(db, loader)

            # Change B, keep A, add C
            _write_preset(preset_dir, "b.yaml", _minimal_preset(id="b", name="B-Updated", vram_mb=3000))
            _write_preset(preset_dir, "c.yaml", _minimal_preset(id="c", name="C", vram_mb=500))

            loader2 = PresetLoader(preset_dirs=[preset_dir])
            result = await sync_presets(db, loader2)

            assert result.inserted == 1   # c
            assert result.updated == 1    # b
            assert result.unchanged == 1  # a

    async def test_updates_unit_vars(self, tmp_path: Path) -> None:
        """sync_presets() should detect and update changed unit_vars."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        _write_preset(
            preset_dir,
            "embed.yaml",
            _minimal_preset(
                id="embed",
                name="Embed",
                unit_vars={"gpu_mem_util": 0.22},
            ),
        )

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            await sync_presets(db, loader)

            got = await db.get_service("embed")
            assert got is not None
            assert got.extra_config.get("unit_vars", {}).get("gpu_mem_util") == 0.22

            # Change gpu_mem_util
            _write_preset(
                preset_dir,
                "embed.yaml",
                _minimal_preset(
                    id="embed",
                    name="Embed",
                    unit_vars={"gpu_mem_util": 0.30},
                ),
            )

            loader2 = PresetLoader(preset_dirs=[preset_dir])
            result = await sync_presets(db, loader2)
            assert result.updated == 1

            got = await db.get_service("embed")
            assert got is not None
            assert got.extra_config["unit_vars"]["gpu_mem_util"] == 0.30

    async def test_no_presets_returns_all_zeros(self, tmp_path: Path) -> None:
        """sync_presets() with no preset files returns zeroed result."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            result = await sync_presets(db, loader)

            assert result.inserted == 0
            assert result.updated == 0
            assert result.unchanged == 0


# ---------------------------------------------------------------------------
# CLI command tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Edge case tests — YAML is source of truth
# ---------------------------------------------------------------------------


class TestSyncEdgeCases:
    """Edge cases for sync_presets(): deletions, orphans, driver changes, etc."""

    async def test_deleted_yaml_removes_service_from_db(self, tmp_path: Path) -> None:
        """If a YAML file is removed, sync should delete the service from DB."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        fp = _write_preset(preset_dir, "old.yaml", _minimal_preset(id="old-svc", name="Old"))
        _write_preset(preset_dir, "keep.yaml", _minimal_preset(id="keep-svc", name="Keep"))

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            await sync_presets(db, loader)
            assert await db.get_service("old-svc") is not None
            assert await db.get_service("keep-svc") is not None

            # Remove old.yaml
            fp.unlink()
            loader2 = PresetLoader(preset_dirs=[preset_dir])
            result = await sync_presets(db, loader2)

            assert result.deleted == 1
            assert result.unchanged == 1
            assert await db.get_service("old-svc") is None
            assert await db.get_service("keep-svc") is not None

    async def test_deleted_yaml_count_in_result(self, tmp_path: Path) -> None:
        """PresetSyncResult should include a deleted count."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        f1 = _write_preset(preset_dir, "a.yaml", _minimal_preset(id="a"))
        f2 = _write_preset(preset_dir, "b.yaml", _minimal_preset(id="b"))
        _write_preset(preset_dir, "c.yaml", _minimal_preset(id="c"))

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            await sync_presets(db, loader)

            # Remove a and b
            f1.unlink()
            f2.unlink()
            loader2 = PresetLoader(preset_dirs=[preset_dir])
            result = await sync_presets(db, loader2)

            assert result.deleted == 2
            assert result.unchanged == 1
            services = await db.list_services()
            assert len(services) == 1
            assert services[0].id == "c"

    async def test_all_yamls_deleted_empties_db(self, tmp_path: Path) -> None:
        """If all YAML files are removed, all services should be deleted."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        f1 = _write_preset(preset_dir, "x.yaml", _minimal_preset(id="x"))
        f2 = _write_preset(preset_dir, "y.yaml", _minimal_preset(id="y"))

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            await sync_presets(db, loader)
            assert len(await db.list_services()) == 2

            f1.unlink()
            f2.unlink()
            loader2 = PresetLoader(preset_dirs=[preset_dir])
            result = await sync_presets(db, loader2)

            assert result.deleted == 2
            assert result.inserted == 0
            assert len(await db.list_services()) == 0

    async def test_manual_db_service_not_deleted(self, tmp_path: Path) -> None:
        """Services added manually (not from presets) should NOT be deleted by sync."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        _write_preset(preset_dir, "yaml-svc.yaml", _minimal_preset(id="yaml-svc"))

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            # Insert a manual service that has no YAML
            manual = Service(
                id="manual-svc",
                name="Manual",
                driver=DriverType.FASTAPI,
                vram_mb=0,
                sleep_mode=SleepMode.NONE,
            )
            await db.insert_service(manual)

            result = await sync_presets(db, loader)

            # yaml-svc inserted, manual-svc untouched
            assert result.inserted == 1
            assert result.deleted == 0
            assert await db.get_service("manual-svc") is not None

    async def test_driver_type_change_updates(self, tmp_path: Path) -> None:
        """Changing driver type in YAML should update the DB service."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        _write_preset(
            preset_dir,
            "svc.yaml",
            _minimal_preset(id="svc", driver="vllm"),
        )

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            await sync_presets(db, loader)
            got = await db.get_service("svc")
            assert got is not None
            assert got.driver == DriverType.VLLM

            # Change driver to llamacpp
            _write_preset(
                preset_dir,
                "svc.yaml",
                _minimal_preset(id="svc", driver="llamacpp"),
            )
            loader2 = PresetLoader(preset_dirs=[preset_dir])
            result = await sync_presets(db, loader2)
            assert result.updated == 1

            got = await db.get_service("svc")
            assert got is not None
            assert got.driver == DriverType.LLAMACPP

    async def test_sleep_mode_change_updates(self, tmp_path: Path) -> None:
        """Changing sleep_mode in YAML should update the DB service."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        _write_preset(
            preset_dir,
            "svc.yaml",
            _minimal_preset(id="svc", sleep_mode="none"),
        )

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            await sync_presets(db, loader)

            _write_preset(
                preset_dir,
                "svc.yaml",
                _minimal_preset(id="svc", sleep_mode="router"),
            )
            loader2 = PresetLoader(preset_dirs=[preset_dir])
            result = await sync_presets(db, loader2)
            assert result.updated == 1

            got = await db.get_service("svc")
            assert got is not None
            assert got.sleep_mode == SleepMode.ROUTER

    async def test_sync_with_empty_db_and_no_presets(self, tmp_path: Path) -> None:
        """Sync with empty DB and no presets should be a no-op."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            result = await sync_presets(db, loader)
            assert result.inserted == 0
            assert result.updated == 0
            assert result.unchanged == 0
            assert result.deleted == 0

    async def test_delete_then_readd_yaml(self, tmp_path: Path) -> None:
        """Removing a YAML, syncing, then re-adding it should re-insert."""
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        fp = _write_preset(preset_dir, "svc.yaml", _minimal_preset(id="svc", name="V1"))

        loader = PresetLoader(preset_dirs=[preset_dir])
        async with Database(tmp_path / "test.db") as db:
            r1 = await sync_presets(db, loader)
            assert r1.inserted == 1

            # Remove
            fp.unlink()
            loader2 = PresetLoader(preset_dirs=[preset_dir])
            r2 = await sync_presets(db, loader2)
            assert r2.deleted == 1
            assert await db.get_service("svc") is None

            # Re-add with different name
            _write_preset(preset_dir, "svc.yaml", _minimal_preset(id="svc", name="V2"))
            loader3 = PresetLoader(preset_dirs=[preset_dir])
            r3 = await sync_presets(db, loader3)
            assert r3.inserted == 1

            got = await db.get_service("svc")
            assert got is not None
            assert got.name == "V2"


# ---------------------------------------------------------------------------
# CLI command tests
# ---------------------------------------------------------------------------


class TestPresetSyncCLI:
    """Tests for the 'gpumod preset sync' CLI command."""

    def test_preset_sync_command_exists(self) -> None:
        """The preset sync subcommand should be registered."""
        from gpumod.cli import app

        result = runner.invoke(app, ["preset", "sync", "--help"])
        assert result.exit_code == 0
        assert "sync" in result.output.lower()

    def test_preset_sync_runs_and_reports(self, tmp_path: Path) -> None:
        """Running 'preset sync' should report inserted/updated/unchanged/deleted."""
        from gpumod.cli import app

        db_path = tmp_path / "test.db"
        result = runner.invoke(app, ["preset", "sync", "--db-path", str(db_path)])
        assert result.exit_code == 0
        assert "inserted" in result.output
        assert "updated" in result.output
        assert "unchanged" in result.output
