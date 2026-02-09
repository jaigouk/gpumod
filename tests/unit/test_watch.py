"""Tests for filesystem watcher — gpumod-7nz.

TDD approach:
- RED: Write failing tests first
- GREEN: Implement to make tests pass
- REFACTOR: Clean up
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import yaml
from typer.testing import CliRunner

from gpumod.cli import app

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
# AC1: gpumod watch command exists and shows help
# ---------------------------------------------------------------------------


class TestWatchCommandExists:
    """Tests for watch command registration — AC1."""

    def test_watch_command_exists(self) -> None:
        """The watch subcommand should be registered."""
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "watch" in result.output.lower()

    def test_watch_has_timeout_option(self) -> None:
        """watch should have a --timeout option for testing."""
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--timeout" in result.output


# ---------------------------------------------------------------------------
# File filtering — AC6
# ---------------------------------------------------------------------------


class TestFileFiltering:
    """Tests for file filter logic — AC6."""

    def test_rejects_swp_files(self) -> None:
        """Filter should reject .swp files."""
        from gpumod.watcher import should_process_file

        assert should_process_file(Path("test.yaml")) is True
        assert should_process_file(Path("test.swp")) is False
        assert should_process_file(Path(".test.yaml.swp")) is False

    def test_rejects_backup_files(self) -> None:
        """Filter should reject backup files ending with ~."""
        from gpumod.watcher import should_process_file

        assert should_process_file(Path("test.yaml~")) is False
        assert should_process_file(Path("test~")) is False

    def test_rejects_tmp_files(self) -> None:
        """Filter should reject .tmp files."""
        from gpumod.watcher import should_process_file

        assert should_process_file(Path("test.tmp")) is False
        assert should_process_file(Path("test.yaml.tmp")) is False

    def test_accepts_yaml_files(self) -> None:
        """Filter should accept .yaml and .yml files."""
        from gpumod.watcher import should_process_file

        assert should_process_file(Path("test.yaml")) is True
        assert should_process_file(Path("test.yml")) is True
        assert should_process_file(Path("preset.yaml")) is True

    def test_rejects_non_yaml_files(self) -> None:
        """Filter should reject non-YAML files."""
        from gpumod.watcher import should_process_file

        assert should_process_file(Path("test.py")) is False
        assert should_process_file(Path("test.json")) is False
        assert should_process_file(Path("README.md")) is False


# ---------------------------------------------------------------------------
# Debounce logic — AC5
# ---------------------------------------------------------------------------


class TestDebounceLogic:
    """Tests for debounce logic — AC5."""

    async def test_debounce_coalesces_rapid_events(self) -> None:
        """Rapid events within debounce window should trigger single sync."""
        from gpumod.watcher import DebouncedSyncTrigger

        sync_fn = AsyncMock()
        trigger = DebouncedSyncTrigger(sync_fn, debounce_ms=100)

        # Fire 5 rapid events
        for _ in range(5):
            await trigger.schedule()
            await asyncio.sleep(0.01)  # 10ms between events

        # Wait for debounce to complete
        await asyncio.sleep(0.15)

        # Should have called sync only once
        assert sync_fn.await_count == 1

    async def test_debounce_separate_events_after_window(self) -> None:
        """Events after debounce window should trigger separate syncs."""
        from gpumod.watcher import DebouncedSyncTrigger

        sync_fn = AsyncMock()
        trigger = DebouncedSyncTrigger(sync_fn, debounce_ms=50)

        # First event
        await trigger.schedule()
        await asyncio.sleep(0.1)  # Wait for debounce to complete

        # Second event after debounce window
        await trigger.schedule()
        await asyncio.sleep(0.1)

        # Should have called sync twice
        assert sync_fn.await_count == 2

    async def test_debounce_cancel_pending(self) -> None:
        """Cancelling should stop pending sync."""
        from gpumod.watcher import DebouncedSyncTrigger

        sync_fn = AsyncMock()
        trigger = DebouncedSyncTrigger(sync_fn, debounce_ms=100)

        await trigger.schedule()
        trigger.cancel()
        await asyncio.sleep(0.15)

        # Sync should not have been called
        sync_fn.assert_not_awaited()


# ---------------------------------------------------------------------------
# Watcher integration — AC2, AC3, AC4
# ---------------------------------------------------------------------------


class TestWatcherIntegration:
    """Integration tests for watcher — AC2, AC3, AC4."""

    async def test_watcher_syncs_on_new_file(self, tmp_path: Path) -> None:
        """Creating a new YAML file should trigger sync — AC2."""
        from gpumod.watcher import FileWatcher

        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()

        sync_fn = AsyncMock()
        watcher = FileWatcher(
            directories=[preset_dir],
            on_change=sync_fn,
            debounce_ms=50,
        )

        # Start watcher in background
        task = asyncio.create_task(watcher.start())

        try:
            await asyncio.sleep(0.1)  # Let watcher start

            # Create a new YAML file
            _write_preset(preset_dir, "new.yaml", _minimal_preset(id="new"))

            # Wait for debounce + processing
            await asyncio.sleep(0.2)

            # Sync should have been called
            sync_fn.assert_awaited()
        finally:
            watcher.stop()
            await task

    async def test_watcher_syncs_on_file_edit(self, tmp_path: Path) -> None:
        """Editing an existing YAML file should trigger sync — AC3."""
        from gpumod.watcher import FileWatcher

        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        fp = _write_preset(preset_dir, "existing.yaml", _minimal_preset(id="existing"))

        sync_fn = AsyncMock()
        watcher = FileWatcher(
            directories=[preset_dir],
            on_change=sync_fn,
            debounce_ms=50,
        )

        task = asyncio.create_task(watcher.start())

        try:
            await asyncio.sleep(0.1)

            # Edit the file
            fp.write_text(yaml.safe_dump(_minimal_preset(id="existing", name="Updated")))

            await asyncio.sleep(0.2)
            sync_fn.assert_awaited()
        finally:
            watcher.stop()
            await task

    async def test_watcher_syncs_on_file_delete(self, tmp_path: Path) -> None:
        """Deleting a YAML file should trigger sync — AC4."""
        from gpumod.watcher import FileWatcher

        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        fp = _write_preset(preset_dir, "delete-me.yaml", _minimal_preset(id="delete-me"))

        sync_fn = AsyncMock()
        watcher = FileWatcher(
            directories=[preset_dir],
            on_change=sync_fn,
            debounce_ms=50,
        )

        task = asyncio.create_task(watcher.start())

        try:
            await asyncio.sleep(0.1)

            # Delete the file
            fp.unlink()

            await asyncio.sleep(0.2)
            sync_fn.assert_awaited()
        finally:
            watcher.stop()
            await task

    async def test_watcher_ignores_temp_files(self, tmp_path: Path) -> None:
        """Temp files should not trigger sync — AC6."""
        from gpumod.watcher import FileWatcher

        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()

        sync_fn = AsyncMock()
        watcher = FileWatcher(
            directories=[preset_dir],
            on_change=sync_fn,
            debounce_ms=50,
        )

        task = asyncio.create_task(watcher.start())

        try:
            await asyncio.sleep(0.1)

            # Create temp files
            (preset_dir / "test.swp").write_text("swap")
            (preset_dir / "test.yaml~").write_text("backup")
            (preset_dir / "test.tmp").write_text("temp")

            await asyncio.sleep(0.2)

            # Sync should NOT have been called
            sync_fn.assert_not_awaited()
        finally:
            watcher.stop()
            await task


# ---------------------------------------------------------------------------
# Error handling — AC7
# ---------------------------------------------------------------------------


class TestWatcherErrorHandling:
    """Tests for error handling — AC7."""

    async def test_malformed_yaml_does_not_crash(self, tmp_path: Path) -> None:
        """Invalid YAML should log warning but not crash — AC7."""
        from gpumod.watcher import FileWatcher

        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()

        sync_fn = AsyncMock()
        watcher = FileWatcher(
            directories=[preset_dir],
            on_change=sync_fn,
            debounce_ms=50,
        )

        task = asyncio.create_task(watcher.start())

        try:
            await asyncio.sleep(0.1)

            # Create malformed YAML
            (preset_dir / "bad.yaml").write_text("id: test\n  bad indent: here")

            await asyncio.sleep(0.2)

            # Sync should still have been called (watcher triggered)
            # The sync function handles the malformed YAML gracefully
            sync_fn.assert_awaited()

            # Watcher should still be running
            assert watcher.is_running
        finally:
            watcher.stop()
            await task

    async def test_sync_error_does_not_crash_watcher(self, tmp_path: Path) -> None:
        """If sync raises an error, watcher should continue."""
        from gpumod.watcher import FileWatcher

        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()

        call_count = 0

        async def failing_sync() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First sync fails")
            # Second call succeeds

        watcher = FileWatcher(
            directories=[preset_dir],
            on_change=failing_sync,
            debounce_ms=50,
        )

        task = asyncio.create_task(watcher.start())

        try:
            await asyncio.sleep(0.1)

            # First change — sync will fail
            _write_preset(preset_dir, "a.yaml", _minimal_preset(id="a"))
            await asyncio.sleep(0.2)

            # Watcher should still be running
            assert watcher.is_running

            # Second change — sync will succeed
            _write_preset(preset_dir, "b.yaml", _minimal_preset(id="b"))
            await asyncio.sleep(0.2)

            assert call_count == 2
        finally:
            watcher.stop()
            await task
