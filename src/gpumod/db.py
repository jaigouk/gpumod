"""SQLite database layer for gpumod.

Provides async CRUD operations for services, modes, settings, and state.
Uses aiosqlite with WAL mode and foreign keys enabled.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import aiosqlite

from gpumod.models import DriverType, Mode, Service, SleepMode

if TYPE_CHECKING:
    from pathlib import Path

_SCHEMA_VERSION = 1

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS services (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    driver TEXT NOT NULL,
    port INTEGER,
    vram_mb INTEGER NOT NULL,
    sleep_mode TEXT NOT NULL DEFAULT 'none',
    health_endpoint TEXT DEFAULT '/health',
    model_id TEXT,
    unit_name TEXT,
    depends_on TEXT NOT NULL DEFAULT '[]',
    startup_timeout INTEGER NOT NULL DEFAULT 120,
    extra_config TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS modes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    total_vram_mb INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mode_services (
    mode_id TEXT NOT NULL REFERENCES modes(id) ON DELETE CASCADE,
    service_id TEXT NOT NULL REFERENCES services(id) ON DELETE CASCADE,
    start_order INTEGER DEFAULT 0,
    PRIMARY KEY (mode_id, service_id)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT DEFAULT '',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class Database:
    """Async SQLite database wrapper for gpumod."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the database connection and ensure the schema exists."""
        self._conn = await aiosqlite.connect(self._db_path)
        # Enable WAL mode for better concurrent read performance.
        await self._conn.execute("PRAGMA journal_mode=WAL")
        # Enable foreign key enforcement.
        await self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = aiosqlite.Row

        await self._conn.executescript(_CREATE_TABLES)

        # Set schema version if not already present.
        cursor = await self._conn.execute("SELECT COUNT(*) FROM schema_version")
        row = await cursor.fetchone()
        assert row is not None
        if row[0] == 0:
            await self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,)
            )
            await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> Database:
        await self.connect()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            msg = "Database is not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._conn

    @staticmethod
    def _row_to_service(row: Any) -> Service:
        return Service(
            id=row["id"],
            name=row["name"],
            driver=DriverType(row["driver"]),
            port=row["port"],
            vram_mb=row["vram_mb"],
            sleep_mode=SleepMode(row["sleep_mode"]),
            health_endpoint=row["health_endpoint"],
            model_id=row["model_id"],
            unit_name=row["unit_name"],
            depends_on=json.loads(row["depends_on"]),
            startup_timeout=row["startup_timeout"],
            extra_config=json.loads(row["extra_config"]),
        )

    @staticmethod
    def _row_to_mode(row: Any) -> Mode:
        return Mode(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            total_vram_mb=row["total_vram_mb"],
        )

    # ------------------------------------------------------------------
    # Services CRUD
    # ------------------------------------------------------------------

    async def list_services(self) -> list[Service]:
        """Return all services ordered by ID."""
        conn = self._ensure_conn()
        cursor = await conn.execute("SELECT * FROM services ORDER BY id")
        rows = await cursor.fetchall()
        return [self._row_to_service(r) for r in rows]

    async def get_service(self, service_id: str) -> Service | None:
        """Return a single service by ID, or None if not found."""
        conn = self._ensure_conn()
        cursor = await conn.execute("SELECT * FROM services WHERE id = ?", (service_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_service(row)

    async def insert_service(self, service: Service) -> None:
        """Insert a service into the database."""
        conn = self._ensure_conn()
        await conn.execute(
            """
            INSERT INTO services
                (id, name, driver, port, vram_mb, sleep_mode, health_endpoint,
                 model_id, unit_name, depends_on, startup_timeout, extra_config)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                service.id,
                service.name,
                service.driver.value,
                service.port,
                service.vram_mb,
                service.sleep_mode.value,
                service.health_endpoint,
                service.model_id,
                service.unit_name,
                json.dumps(service.depends_on),
                service.startup_timeout,
                json.dumps(service.extra_config),
            ),
        )
        await conn.commit()

    async def delete_service(self, service_id: str) -> None:
        """Delete a service by ID."""
        conn = self._ensure_conn()
        await conn.execute("DELETE FROM services WHERE id = ?", (service_id,))
        await conn.commit()

    # ------------------------------------------------------------------
    # Modes CRUD
    # ------------------------------------------------------------------

    async def list_modes(self) -> list[Mode]:
        """Return all modes ordered by ID."""
        conn = self._ensure_conn()
        cursor = await conn.execute("SELECT * FROM modes ORDER BY id")
        rows = await cursor.fetchall()
        return [self._row_to_mode(r) for r in rows]

    async def get_mode(self, mode_id: str) -> Mode | None:
        """Return a single mode by ID, or None if not found."""
        conn = self._ensure_conn()
        cursor = await conn.execute("SELECT * FROM modes WHERE id = ?", (mode_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_mode(row)

    async def insert_mode(self, mode: Mode) -> None:
        """Insert a mode into the database."""
        conn = self._ensure_conn()
        await conn.execute(
            """
            INSERT INTO modes (id, name, description, total_vram_mb)
            VALUES (?, ?, ?, ?)
            """,
            (mode.id, mode.name, mode.description, mode.total_vram_mb),
        )
        await conn.commit()

    async def get_mode_services(self, mode_id: str) -> list[Service]:
        """Return services for a mode, ordered by start_order."""
        conn = self._ensure_conn()
        cursor = await conn.execute(
            """
            SELECT s.*
            FROM services s
            JOIN mode_services ms ON s.id = ms.service_id
            WHERE ms.mode_id = ?
            ORDER BY ms.start_order
            """,
            (mode_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_service(r) for r in rows]

    async def set_mode_services(
        self,
        mode_id: str,
        service_ids: list[str],
        orders: list[int] | None = None,
    ) -> None:
        """Replace the mode_services junction rows for a mode."""
        conn = self._ensure_conn()
        # Remove existing rows.
        await conn.execute("DELETE FROM mode_services WHERE mode_id = ?", (mode_id,))
        # Insert new rows.
        if orders is None:
            orders = list(range(len(service_ids)))
        for sid, order in zip(service_ids, orders, strict=True):
            await conn.execute(
                "INSERT INTO mode_services (mode_id, service_id, start_order) VALUES (?, ?, ?)",
                (mode_id, sid, order),
            )
        await conn.commit()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    async def get_setting(self, key: str, default: str | None = None) -> str | None:
        """Return the value for a setting key, or *default* if not found."""
        conn = self._ensure_conn()
        cursor = await conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = await cursor.fetchone()
        if row is None:
            return default
        return str(row["value"])

    async def set_setting(self, key: str, value: str, description: str = "") -> None:
        """Upsert a setting."""
        conn = self._ensure_conn()
        await conn.execute(
            """
            INSERT INTO settings (key, value, description, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                description = excluded.description,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, value, description),
        )
        await conn.commit()

    # ------------------------------------------------------------------
    # State: current mode
    # ------------------------------------------------------------------

    async def get_current_mode(self) -> str | None:
        """Return the current mode ID, or None if not set."""
        return await self.get_setting("current_mode")

    async def set_current_mode(self, mode_id: str) -> None:
        """Set the current mode ID."""
        await self.set_setting("current_mode", mode_id, description="Currently active mode")
