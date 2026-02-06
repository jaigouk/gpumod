"""Tests for gpumod.db — SQLite schema and CRUD operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpumod.db import Database
from gpumod.models import DriverType, Mode, Service, SleepMode

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(
    id: str = "svc-1",
    name: str = "Test Service",
    driver: DriverType = DriverType.VLLM,
    port: int = 8000,
    vram_mb: int = 2500,
    sleep_mode: SleepMode = SleepMode.NONE,
    health_endpoint: str = "/health",
    model_id: str | None = "org/model",
    unit_name: str = "vllm-test.service",
    depends_on: list[str] | None = None,
    startup_timeout: int = 60,
    extra_config: dict[str, object] | None = None,
) -> Service:
    return Service(
        id=id,
        name=name,
        driver=driver,
        port=port,
        vram_mb=vram_mb,
        sleep_mode=sleep_mode,
        health_endpoint=health_endpoint,
        model_id=model_id,
        unit_name=unit_name,
        depends_on=depends_on or [],
        startup_timeout=startup_timeout,
        extra_config=extra_config or {},
    )


def _make_mode(
    id: str = "mode-1",
    name: str = "Test Mode",
    description: str = "A test mode",
    services: list[str] | None = None,
    total_vram_mb: int = 5000,
) -> Mode:
    return Mode(
        id=id,
        name=name,
        description=description,
        services=services or [],
        total_vram_mb=total_vram_mb,
    )


# ---------------------------------------------------------------------------
# Schema / connect
# ---------------------------------------------------------------------------


class TestConnect:
    """Tests for Database.connect()."""

    async def test_connect_creates_all_tables(self, tmp_path: Path) -> None:
        """connect() should create all expected tables."""
        db = Database(tmp_path / "test.db")
        await db.connect()
        try:
            assert db._conn is not None
            cursor = await db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = {row[0] for row in await cursor.fetchall()}
            expected = {"services", "modes", "mode_services", "settings", "schema_version"}
            assert expected.issubset(tables), f"Missing tables: {expected - tables}"
        finally:
            await db.close()

    async def test_connect_sets_schema_version(self, tmp_path: Path) -> None:
        """connect() should record the schema version."""
        db = Database(tmp_path / "test.db")
        await db.connect()
        try:
            cursor = await db._conn.execute("SELECT version FROM schema_version")  # type: ignore[union-attr]
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] >= 1
        finally:
            await db.close()

    async def test_reconnect_idempotent(self, tmp_path: Path) -> None:
        """Re-connecting on an existing DB must not lose data."""
        db_path = tmp_path / "test.db"

        # First connection — insert a service
        db = Database(db_path)
        await db.connect()
        svc = _make_service(id="keep-me")
        await db.insert_service(svc)
        await db.close()

        # Second connection
        db2 = Database(db_path)
        await db2.connect()
        try:
            got = await db2.get_service("keep-me")
            assert got is not None
            assert got.id == "keep-me"
        finally:
            await db2.close()


# ---------------------------------------------------------------------------
# Services CRUD
# ---------------------------------------------------------------------------


class TestServicesCRUD:
    """Tests for service insert / get / list / delete."""

    async def test_insert_and_get_service(self, tmp_path: Path) -> None:
        """Round-trip: insert then get returns identical service."""
        async with Database(tmp_path / "test.db") as db:
            svc = _make_service(
                id="svc-roundtrip",
                depends_on=["dep-a", "dep-b"],
                extra_config={"key": "value"},
            )
            await db.insert_service(svc)
            got = await db.get_service("svc-roundtrip")

            assert got is not None
            assert got.id == svc.id
            assert got.name == svc.name
            assert got.driver == svc.driver
            assert got.port == svc.port
            assert got.vram_mb == svc.vram_mb
            assert got.sleep_mode == svc.sleep_mode
            assert got.health_endpoint == svc.health_endpoint
            assert got.model_id == svc.model_id
            assert got.unit_name == svc.unit_name
            assert got.depends_on == svc.depends_on
            assert got.startup_timeout == svc.startup_timeout
            assert got.extra_config == svc.extra_config

    async def test_list_services_ordered_by_id(self, tmp_path: Path) -> None:
        """list_services() must return all services ordered by ID."""
        async with Database(tmp_path / "test.db") as db:
            await db.insert_service(_make_service(id="z-service"))
            await db.insert_service(_make_service(id="a-service"))
            await db.insert_service(_make_service(id="m-service"))

            services = await db.list_services()

            assert len(services) == 3
            assert [s.id for s in services] == ["a-service", "m-service", "z-service"]

    async def test_get_service_returns_none_for_unknown(self, tmp_path: Path) -> None:
        """get_service() returns None for an ID not in the DB."""
        async with Database(tmp_path / "test.db") as db:
            assert await db.get_service("nonexistent") is None

    async def test_delete_service(self, tmp_path: Path) -> None:
        """delete_service() removes the service from the DB."""
        async with Database(tmp_path / "test.db") as db:
            svc = _make_service(id="to-delete")
            await db.insert_service(svc)
            assert await db.get_service("to-delete") is not None

            await db.delete_service("to-delete")
            assert await db.get_service("to-delete") is None


# ---------------------------------------------------------------------------
# Modes CRUD
# ---------------------------------------------------------------------------


class TestModesCRUD:
    """Tests for mode insert / get and junction table operations."""

    async def test_insert_and_get_mode(self, tmp_path: Path) -> None:
        """Round-trip: insert then get returns identical mode."""
        async with Database(tmp_path / "test.db") as db:
            mode = _make_mode(id="mode-rt", services=["svc-1", "svc-2"])
            await db.insert_mode(mode)
            got = await db.get_mode("mode-rt")

            assert got is not None
            assert got.id == mode.id
            assert got.name == mode.name
            assert got.description == mode.description
            assert got.total_vram_mb == mode.total_vram_mb

    async def test_get_mode_services_in_start_order(self, tmp_path: Path) -> None:
        """get_mode_services() returns services ordered by start_order."""
        async with Database(tmp_path / "test.db") as db:
            svc_a = _make_service(id="svc-a", name="A")
            svc_b = _make_service(id="svc-b", name="B")
            svc_c = _make_service(id="svc-c", name="C")
            for s in (svc_a, svc_b, svc_c):
                await db.insert_service(s)

            mode = _make_mode(id="ordered-mode")
            await db.insert_mode(mode)
            # B first (order 1), then C (order 2), then A (order 3)
            await db.set_mode_services(
                "ordered-mode",
                ["svc-b", "svc-c", "svc-a"],
                orders=[1, 2, 3],
            )

            result = await db.get_mode_services("ordered-mode")
            assert [s.id for s in result] == ["svc-b", "svc-c", "svc-a"]

    async def test_set_mode_services_creates_junction(self, tmp_path: Path) -> None:
        """set_mode_services() creates rows in the mode_services table."""
        async with Database(tmp_path / "test.db") as db:
            svc = _make_service(id="svc-junct")
            await db.insert_service(svc)
            mode = _make_mode(id="junct-mode")
            await db.insert_mode(mode)

            await db.set_mode_services("junct-mode", ["svc-junct"])

            services = await db.get_mode_services("junct-mode")
            assert len(services) == 1
            assert services[0].id == "svc-junct"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    """Tests for key-value settings."""

    async def test_get_set_setting_roundtrip(self, tmp_path: Path) -> None:
        """set_setting / get_setting round-trip."""
        async with Database(tmp_path / "test.db") as db:
            await db.set_setting("theme", "dark", description="UI theme")
            assert await db.get_setting("theme") == "dark"

    async def test_get_setting_returns_default_for_missing(self, tmp_path: Path) -> None:
        """get_setting() returns the supplied default when key is missing."""
        async with Database(tmp_path / "test.db") as db:
            assert await db.get_setting("missing-key") is None
            assert await db.get_setting("missing-key", default="fallback") == "fallback"


# ---------------------------------------------------------------------------
# Current-mode state
# ---------------------------------------------------------------------------


class TestCurrentMode:
    """Tests for get/set current mode."""

    async def test_get_set_current_mode(self, tmp_path: Path) -> None:
        """set_current_mode / get_current_mode round-trip."""
        async with Database(tmp_path / "test.db") as db:
            await db.set_current_mode("code")
            assert await db.get_current_mode() == "code"

            await db.set_current_mode("rag")
            assert await db.get_current_mode() == "rag"


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestAsyncContextManager:
    """Tests for ``async with Database(...) as db``."""

    async def test_context_manager(self, tmp_path: Path) -> None:
        """async with Database() should connect on enter and close on exit."""
        db_path = tmp_path / "ctx.db"
        async with Database(db_path) as db:
            # Must be usable inside the block
            await db.set_setting("alive", "yes")
            assert await db.get_setting("alive") == "yes"

        # After exiting, connection should be closed
        assert db._conn is None


# ---------------------------------------------------------------------------
# Foreign-key cascade
# ---------------------------------------------------------------------------


class TestForeignKeyCascade:
    """Deleting a mode should cascade-delete mode_services entries."""

    async def test_delete_mode_cascades_to_mode_services(self, tmp_path: Path) -> None:
        """Deleting a mode removes its mode_services junction rows."""
        async with Database(tmp_path / "test.db") as db:
            svc = _make_service(id="svc-cascade")
            await db.insert_service(svc)

            mode = _make_mode(id="cascade-mode")
            await db.insert_mode(mode)
            await db.set_mode_services("cascade-mode", ["svc-cascade"])

            # Verify junction row exists
            services = await db.get_mode_services("cascade-mode")
            assert len(services) == 1

            # Delete the mode directly
            assert db._conn is not None
            await db._conn.execute("DELETE FROM modes WHERE id = ?", ("cascade-mode",))
            await db._conn.commit()

            # Junction rows must be gone
            cursor = await db._conn.execute(
                "SELECT COUNT(*) FROM mode_services WHERE mode_id = ?",
                ("cascade-mode",),
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 0
