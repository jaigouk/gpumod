"""SQLite database layer for gpumod.

Provides async CRUD operations for services, modes, settings, and state.
Uses aiosqlite with WAL mode and foreign keys enabled.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import aiosqlite

from gpumod.models import (
    DriverType,
    Mode,
    ModelInfo,
    ModelSource,
    Service,
    ServiceTemplate,
    SleepMode,
)
from gpumod.validation import validate_extra_config, validate_model_id, validate_vram_mb

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2

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

CREATE TABLE IF NOT EXISTS service_templates (
    service_id TEXT PRIMARY KEY REFERENCES services(id) ON DELETE CASCADE,
    unit_template TEXT NOT NULL,
    preset_template TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    parameters_b REAL,
    architecture TEXT,
    base_vram_mb INTEGER,
    kv_cache_per_1k_tokens_mb INTEGER,
    quantizations TEXT NOT NULL DEFAULT '[]',
    capabilities TEXT NOT NULL DEFAULT '[]',
    fetched_at TIMESTAMP,
    notes TEXT
);
"""


class Database:
    """Async SQLite database wrapper for gpumod."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the database connection and ensure the schema exists."""
        logger.debug("Connecting to database at %s", self._db_path)
        self._conn = await aiosqlite.connect(self._db_path)
        # Enable WAL mode for better concurrent read performance.
        await self._conn.execute("PRAGMA journal_mode=WAL")
        # Enable foreign key enforcement.
        await self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = aiosqlite.Row

        await self._conn.executescript(_CREATE_TABLES)

        # Manage schema version: insert if empty, update if outdated.
        cursor = await self._conn.execute("SELECT COUNT(*) FROM schema_version")
        row = await cursor.fetchone()
        assert row is not None
        if row[0] == 0:
            await self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,)
            )
        else:
            await self._conn.execute("UPDATE schema_version SET version = ?", (_SCHEMA_VERSION,))
        await self._conn.commit()
        logger.debug("Database connected (schema v%d)", _SCHEMA_VERSION)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")

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

    @staticmethod
    def _row_to_service_template(row: Any) -> ServiceTemplate:
        return ServiceTemplate(
            service_id=row["service_id"],
            unit_template=row["unit_template"],
            preset_template=row["preset_template"],
        )

    @staticmethod
    def _row_to_model_info(row: Any) -> ModelInfo:
        return ModelInfo(
            id=row["id"],
            source=ModelSource(row["source"]),
            parameters_b=row["parameters_b"],
            architecture=row["architecture"],
            base_vram_mb=row["base_vram_mb"],
            kv_cache_per_1k_tokens_mb=row["kv_cache_per_1k_tokens_mb"],
            quantizations=json.loads(row["quantizations"]),
            capabilities=json.loads(row["capabilities"]),
            fetched_at=row["fetched_at"],
            notes=row["notes"],
        )

    # ------------------------------------------------------------------
    # Services CRUD
    # ------------------------------------------------------------------

    async def list_services(self) -> list[Service]:
        """Return all services ordered by ID."""
        conn = self._ensure_conn()
        cursor = await conn.execute("SELECT * FROM services ORDER BY id")
        rows = await cursor.fetchall()
        result = [self._row_to_service(r) for r in rows]
        logger.debug("Listed %d services", len(result))
        return result

    async def get_service(self, service_id: str) -> Service | None:
        """Return a single service by ID, or None if not found."""
        conn = self._ensure_conn()
        cursor = await conn.execute("SELECT * FROM services WHERE id = ?", (service_id,))
        row = await cursor.fetchone()
        if row is None:
            logger.debug("Service %r not found", service_id)
            return None
        logger.debug("Retrieved service %r", service_id)
        return self._row_to_service(row)

    async def insert_service(self, service: Service) -> None:
        """Insert a service into the database."""
        validate_vram_mb(service.vram_mb)
        validate_extra_config(service.extra_config)
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
        logger.debug(
            "Inserted service %r (driver=%s, vram=%dMB)",
            service.id,
            service.driver.value,
            service.vram_mb,
        )

    async def delete_service(self, service_id: str) -> None:
        """Delete a service by ID."""
        conn = self._ensure_conn()
        await conn.execute("DELETE FROM services WHERE id = ?", (service_id,))
        await conn.commit()
        logger.debug("Deleted service %r", service_id)

    # ------------------------------------------------------------------
    # Modes CRUD
    # ------------------------------------------------------------------

    async def list_modes(self) -> list[Mode]:
        """Return all modes ordered by ID."""
        conn = self._ensure_conn()
        cursor = await conn.execute("SELECT * FROM modes ORDER BY id")
        rows = await cursor.fetchall()
        result = [self._row_to_mode(r) for r in rows]
        logger.debug("Listed %d modes", len(result))
        return result

    async def get_mode(self, mode_id: str) -> Mode | None:
        """Return a single mode by ID, or None if not found."""
        conn = self._ensure_conn()
        cursor = await conn.execute("SELECT * FROM modes WHERE id = ?", (mode_id,))
        row = await cursor.fetchone()
        if row is None:
            logger.debug("Mode %r not found", mode_id)
            return None
        logger.debug("Retrieved mode %r", mode_id)
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
        logger.debug("Inserted mode %r", mode.id)

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
        result = [self._row_to_service(r) for r in rows]
        logger.debug("Mode %r has %d services", mode_id, len(result))
        return result

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
        logger.debug("Set %d services for mode %r", len(service_ids), mode_id)

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

    # ------------------------------------------------------------------
    # Service Templates CRUD
    # ------------------------------------------------------------------

    async def get_template(self, service_id: str) -> ServiceTemplate | None:
        """Return a service template by service_id, or None if not found."""
        conn = self._ensure_conn()
        cursor = await conn.execute(
            "SELECT * FROM service_templates WHERE service_id = ?", (service_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_service_template(row)

    async def insert_template(self, template: ServiceTemplate) -> None:
        """Insert a service template into the database."""
        conn = self._ensure_conn()
        await conn.execute(
            """
            INSERT INTO service_templates (service_id, unit_template, preset_template)
            VALUES (?, ?, ?)
            """,
            (template.service_id, template.unit_template, template.preset_template),
        )
        await conn.commit()

    async def delete_template(self, service_id: str) -> None:
        """Delete a service template by service_id."""
        conn = self._ensure_conn()
        await conn.execute("DELETE FROM service_templates WHERE service_id = ?", (service_id,))
        await conn.commit()

    # ------------------------------------------------------------------
    # Models CRUD
    # ------------------------------------------------------------------

    async def get_model(self, model_id: str) -> ModelInfo | None:
        """Return a model by ID, or None if not found."""
        validate_model_id(model_id)
        conn = self._ensure_conn()
        cursor = await conn.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        row = await cursor.fetchone()
        if row is None:
            logger.debug("Model %r not found", model_id)
            return None
        logger.debug("Retrieved model %r", model_id)
        return self._row_to_model_info(row)

    async def list_models(self) -> list[ModelInfo]:
        """Return all models ordered by ID."""
        conn = self._ensure_conn()
        cursor = await conn.execute("SELECT * FROM models ORDER BY id")
        rows = await cursor.fetchall()
        result = [self._row_to_model_info(r) for r in rows]
        logger.debug("Listed %d models", len(result))
        return result

    async def insert_model(self, model: ModelInfo) -> None:
        """Insert a model into the database."""
        validate_model_id(model.id)
        conn = self._ensure_conn()
        await conn.execute(
            """
            INSERT INTO models
                (id, source, parameters_b, architecture, base_vram_mb,
                 kv_cache_per_1k_tokens_mb, quantizations, capabilities,
                 fetched_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model.id,
                model.source.value,
                model.parameters_b,
                model.architecture,
                model.base_vram_mb,
                model.kv_cache_per_1k_tokens_mb,
                json.dumps(model.quantizations),
                json.dumps(model.capabilities),
                model.fetched_at,
                model.notes,
            ),
        )
        await conn.commit()
        logger.debug("Inserted model %r", model.id)

    async def delete_model(self, model_id: str) -> None:
        """Delete a model by ID."""
        conn = self._ensure_conn()
        await conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
        await conn.commit()
        logger.debug("Deleted model %r", model_id)
