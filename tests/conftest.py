"""Shared test fixtures for gpumod."""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
async def tmp_db(tmp_db_path: Path) -> AsyncGenerator[aiosqlite.Connection, None]:
    """Provide a temporary async SQLite connection."""
    async with aiosqlite.connect(tmp_db_path) as db:
        yield db
