"""Filesystem watcher for preset/mode YAML hot-reload.

Implements gpumod-7nz: watches preset and mode YAML directories for changes
and automatically runs sync. Uses watchfiles (Rust-based) for efficient
cross-platform filesystem watching.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from watchfiles import Change, awatch

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File filtering — AC6
# ---------------------------------------------------------------------------

# Extensions to accept
_YAML_EXTENSIONS = {".yaml", ".yml"}

# Patterns to reject (editor temp files)
_IGNORED_SUFFIXES = {".swp", ".tmp"}
_IGNORED_PREFIXES = {"."}  # Hidden files like .test.yaml.swp


def should_process_file(path: Path) -> bool:
    """Check if a file should trigger sync.

    Accepts .yaml/.yml files, rejects:
    - .swp files (vim swap)
    - .tmp files (temporary)
    - ~ backup files (many editors)
    - Hidden files starting with .

    Parameters
    ----------
    path:
        Path to the file to check.

    Returns
    -------
    bool
        True if the file should trigger sync, False otherwise.
    """
    name = path.name
    suffix = path.suffix.lower()

    # Reject backup files ending with ~
    if name.endswith("~"):
        return False

    # Reject ignored suffixes
    if suffix in _IGNORED_SUFFIXES:
        return False

    # Reject hidden swap files like .test.yaml.swp
    if name.startswith(".") and suffix == ".swp":
        return False

    # Accept only YAML files
    return suffix in _YAML_EXTENSIONS


# ---------------------------------------------------------------------------
# Debounced sync trigger — AC5
# ---------------------------------------------------------------------------


class DebouncedSyncTrigger:
    """Debounces rapid filesystem events into a single sync call.

    Uses a sliding window approach: waits for `debounce_ms` after the last
    event before triggering sync. If new events arrive during the window,
    the timer resets.

    Parameters
    ----------
    sync_fn:
        Async function to call when debounce window expires.
    debounce_ms:
        Debounce window in milliseconds. Default 500ms.
    """

    def __init__(
        self,
        sync_fn: Callable[[], Awaitable[None]],
        debounce_ms: int = 500,
    ) -> None:
        self._sync_fn = sync_fn
        self._debounce_s = debounce_ms / 1000.0
        self._pending_task: asyncio.Task[None] | None = None

    async def schedule(self) -> None:
        """Schedule a sync after the debounce window.

        If a sync is already pending, it will be cancelled and rescheduled.
        """
        # Cancel any pending sync
        if self._pending_task is not None and not self._pending_task.done():
            self._pending_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pending_task

        # Schedule new sync after debounce window
        self._pending_task = asyncio.create_task(self._delayed_sync())

    async def _delayed_sync(self) -> None:
        """Wait for debounce window then call sync."""
        await asyncio.sleep(self._debounce_s)
        try:
            await self._sync_fn()
        except Exception:
            logger.exception("Sync failed during watch")

    def cancel(self) -> None:
        """Cancel any pending sync."""
        if self._pending_task is not None and not self._pending_task.done():
            self._pending_task.cancel()


# ---------------------------------------------------------------------------
# File watcher — AC1-AC4
# ---------------------------------------------------------------------------


class FileWatcher:
    """Watches directories for YAML file changes and triggers sync.

    Uses watchfiles for efficient cross-platform filesystem watching.
    Debounces rapid events and filters out temp files.

    Parameters
    ----------
    directories:
        List of directories to watch.
    on_change:
        Async callback to invoke when relevant files change.
    debounce_ms:
        Debounce window in milliseconds. Default 500ms.
    """

    def __init__(
        self,
        directories: list[Path],
        on_change: Callable[[], Awaitable[None]],
        debounce_ms: int = 500,
    ) -> None:
        self._directories = [d for d in directories if d.exists()]
        self._on_change = on_change
        self._debounce_ms = debounce_ms
        self._running = False
        self._stop_event = asyncio.Event()
        self._trigger = DebouncedSyncTrigger(on_change, debounce_ms)

    @property
    def is_running(self) -> bool:
        """Return True if the watcher is currently running."""
        return self._running

    async def start(self) -> None:
        """Start watching directories for changes.

        Blocks until stop() is called. Handles errors gracefully and
        continues watching.
        """
        if not self._directories:
            logger.warning("No directories to watch")
            return

        self._running = True
        self._stop_event.clear()

        logger.info("Starting watcher for: %s", [str(d) for d in self._directories])

        try:
            async for changes in awatch(
                *self._directories,
                stop_event=self._stop_event,
                recursive=True,
            ):
                await self._handle_changes(changes)
        except Exception:
            logger.exception("Watcher error")
        finally:
            self._running = False
            self._trigger.cancel()

    async def _handle_changes(self, changes: set[tuple[Change, str]]) -> None:
        """Process a batch of filesystem changes.

        Filters out temp files and triggers debounced sync for relevant changes.
        """
        relevant_changes = []

        for change_type, path_str in changes:
            path = Path(path_str)
            if should_process_file(path):
                relevant_changes.append((change_type, path))
                logger.debug("Detected %s: %s", change_type.name, path)

        if relevant_changes:
            await self._trigger.schedule()

    def stop(self) -> None:
        """Stop the watcher."""
        logger.info("Stopping watcher")
        self._stop_event.set()
        self._trigger.cancel()


# ---------------------------------------------------------------------------
# Watch command runner
# ---------------------------------------------------------------------------


async def run_watcher(
    preset_dirs: list[Path],
    mode_dirs: list[Path],
    sync_fn: Callable[[], Awaitable[None]],
    debounce_ms: int = 500,
    watch_timeout: float | None = None,
) -> None:
    """Run the filesystem watcher until interrupted or timeout.

    Parameters
    ----------
    preset_dirs:
        Directories containing preset YAML files.
    mode_dirs:
        Directories containing mode YAML files.
    sync_fn:
        Async function to call when files change.
    debounce_ms:
        Debounce window in milliseconds.
    watch_timeout:
        Optional timeout in seconds (for testing).
    """
    all_dirs = preset_dirs + mode_dirs
    watcher = FileWatcher(all_dirs, sync_fn, debounce_ms)

    if watch_timeout is not None:
        # For testing: run with timeout
        task = asyncio.create_task(watcher.start())
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=watch_timeout)
        except TimeoutError:
            watcher.stop()
            await task
    else:
        await watcher.start()
