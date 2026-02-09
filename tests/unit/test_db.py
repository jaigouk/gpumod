"""Tests for gpumod.db — SQLite schema and CRUD operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gpumod.db import Database
from gpumod.models import (
    DriverType,
    Mode,
    ModelInfo,
    ModelSource,
    Service,
    ServiceTemplate,
    SleepMode,
)

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
            expected = {
                "services",
                "modes",
                "mode_services",
                "settings",
                "schema_version",
                "service_templates",
                "models",
            }
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
                extra_config={"context_size": "4096"},
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


# ---------------------------------------------------------------------------
# Helpers — Phase 2 models
# ---------------------------------------------------------------------------


def _make_template(
    service_id: str = "vllm-chat",
    unit_template: str = "[Unit]\nDescription=vLLM Chat\n",
    preset_template: str | None = None,
) -> ServiceTemplate:
    return ServiceTemplate(
        service_id=service_id,
        unit_template=unit_template,
        preset_template=preset_template,
    )


def _make_model_info(
    id: str = "meta-llama/Llama-3-8B",
    source: ModelSource = ModelSource.HUGGINGFACE,
    parameters_b: float | None = 8.0,
    architecture: str | None = "llama",
    base_vram_mb: int | None = 16000,
    kv_cache_per_1k_tokens_mb: int | None = 64,
    quantizations: list[str] | None = None,
    capabilities: list[str] | None = None,
    fetched_at: str | None = "2025-01-15T10:00:00Z",
    notes: str | None = None,
) -> ModelInfo:
    return ModelInfo(
        id=id,
        source=source,
        parameters_b=parameters_b,
        architecture=architecture,
        base_vram_mb=base_vram_mb,
        kv_cache_per_1k_tokens_mb=kv_cache_per_1k_tokens_mb,
        quantizations=quantizations or [],
        capabilities=capabilities or [],
        fetched_at=fetched_at,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Schema v2 verification
# ---------------------------------------------------------------------------


class TestSchemaV2:
    """Tests for schema version 2 tables."""

    async def test_schema_version_is_2(self, tmp_path: Path) -> None:
        """Schema version should be 2 after connect."""
        async with Database(tmp_path / "test.db") as db:
            conn = db._ensure_conn()
            cursor = await conn.execute("SELECT version FROM schema_version")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 2

    async def test_service_templates_table_exists(self, tmp_path: Path) -> None:
        """service_templates table should be created on connect."""
        async with Database(tmp_path / "test.db") as db:
            conn = db._ensure_conn()
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='service_templates'"
            )
            row = await cursor.fetchone()
            assert row is not None

    async def test_models_table_exists(self, tmp_path: Path) -> None:
        """models table should be created on connect."""
        async with Database(tmp_path / "test.db") as db:
            conn = db._ensure_conn()
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='models'"
            )
            row = await cursor.fetchone()
            assert row is not None


# ---------------------------------------------------------------------------
# Service Templates CRUD
# ---------------------------------------------------------------------------


class TestServiceTemplatesCRUD:
    """Tests for service template insert / get / delete."""

    async def test_insert_and_get_template(self, tmp_path: Path) -> None:
        """Round-trip: insert then get returns identical template."""
        async with Database(tmp_path / "test.db") as db:
            # Need a service first for FK
            svc = _make_service(id="vllm-chat")
            await db.insert_service(svc)

            tpl = _make_template(
                service_id="vllm-chat",
                unit_template="[Unit]\nDescription=vLLM Chat\n[Service]\nExecStart=/usr/bin/vllm",
                preset_template="id: vllm-chat\ndriver: vllm\n",
            )
            await db.insert_template(tpl)
            got = await db.get_template("vllm-chat")

            assert got is not None
            assert got.service_id == tpl.service_id
            assert got.unit_template == tpl.unit_template
            assert got.preset_template == tpl.preset_template

    async def test_get_template_returns_none_for_unknown(self, tmp_path: Path) -> None:
        """get_template() returns None for a service_id not in the DB."""
        async with Database(tmp_path / "test.db") as db:
            assert await db.get_template("nonexistent") is None

    async def test_delete_template(self, tmp_path: Path) -> None:
        """delete_template() removes the template from the DB."""
        async with Database(tmp_path / "test.db") as db:
            svc = _make_service(id="to-del-tpl")
            await db.insert_service(svc)

            tpl = _make_template(service_id="to-del-tpl")
            await db.insert_template(tpl)
            assert await db.get_template("to-del-tpl") is not None

            await db.delete_template("to-del-tpl")
            assert await db.get_template("to-del-tpl") is None

    async def test_template_without_preset(self, tmp_path: Path) -> None:
        """Template can have no preset_template (NULL)."""
        async with Database(tmp_path / "test.db") as db:
            svc = _make_service(id="no-preset")
            await db.insert_service(svc)

            tpl = _make_template(service_id="no-preset", preset_template=None)
            await db.insert_template(tpl)
            got = await db.get_template("no-preset")

            assert got is not None
            assert got.preset_template is None

    async def test_template_cascade_on_service_delete(self, tmp_path: Path) -> None:
        """Deleting a service should cascade-delete its template."""
        async with Database(tmp_path / "test.db") as db:
            svc = _make_service(id="cascade-svc")
            await db.insert_service(svc)

            tpl = _make_template(service_id="cascade-svc")
            await db.insert_template(tpl)
            assert await db.get_template("cascade-svc") is not None

            # Delete the service
            await db.delete_service("cascade-svc")
            # Template should be gone
            assert await db.get_template("cascade-svc") is None


# ---------------------------------------------------------------------------
# Models CRUD
# ---------------------------------------------------------------------------


class TestModelsCRUD:
    """Tests for model insert / get / list / delete."""

    async def test_insert_and_get_model(self, tmp_path: Path) -> None:
        """Round-trip: insert then get returns identical model info."""
        async with Database(tmp_path / "test.db") as db:
            model = _make_model_info(
                id="meta-llama/Llama-3-8B",
                quantizations=["fp16", "q4_k_m"],
                capabilities=["chat", "code"],
                notes="Test model",
            )
            await db.insert_model(model)
            got = await db.get_model("meta-llama/Llama-3-8B")

            assert got is not None
            assert got.id == model.id
            assert got.source == model.source
            assert got.parameters_b == model.parameters_b
            assert got.architecture == model.architecture
            assert got.base_vram_mb == model.base_vram_mb
            assert got.kv_cache_per_1k_tokens_mb == model.kv_cache_per_1k_tokens_mb
            assert got.quantizations == model.quantizations
            assert got.capabilities == model.capabilities
            assert got.fetched_at == model.fetched_at
            assert got.notes == model.notes

    async def test_get_model_returns_none_for_unknown(self, tmp_path: Path) -> None:
        """get_model() returns None for an ID not in the DB."""
        async with Database(tmp_path / "test.db") as db:
            assert await db.get_model("nonexistent") is None

    async def test_list_models_empty(self, tmp_path: Path) -> None:
        """list_models() returns empty list when no models exist."""
        async with Database(tmp_path / "test.db") as db:
            models = await db.list_models()
            assert models == []

    async def test_list_models_ordered_by_id(self, tmp_path: Path) -> None:
        """list_models() returns all models ordered by ID."""
        async with Database(tmp_path / "test.db") as db:
            await db.insert_model(_make_model_info(id="z-model"))
            await db.insert_model(_make_model_info(id="a-model"))
            await db.insert_model(_make_model_info(id="m-model"))

            models = await db.list_models()
            assert len(models) == 3
            assert [m.id for m in models] == ["a-model", "m-model", "z-model"]

    async def test_delete_model(self, tmp_path: Path) -> None:
        """delete_model() removes the model from the DB."""
        async with Database(tmp_path / "test.db") as db:
            model = _make_model_info(id="to-delete")
            await db.insert_model(model)
            assert await db.get_model("to-delete") is not None

            await db.delete_model("to-delete")
            assert await db.get_model("to-delete") is None

    async def test_model_with_null_optional_fields(self, tmp_path: Path) -> None:
        """Model can have NULL optional fields (parameters_b, architecture, etc)."""
        async with Database(tmp_path / "test.db") as db:
            model = _make_model_info(
                id="minimal-model",
                source=ModelSource.LOCAL,
                parameters_b=None,
                architecture=None,
                base_vram_mb=None,
                kv_cache_per_1k_tokens_mb=None,
                fetched_at=None,
                notes=None,
            )
            await db.insert_model(model)
            got = await db.get_model("minimal-model")

            assert got is not None
            assert got.parameters_b is None
            assert got.architecture is None
            assert got.base_vram_mb is None
            assert got.kv_cache_per_1k_tokens_mb is None
            assert got.fetched_at is None
            assert got.notes is None

    async def test_model_gguf_source(self, tmp_path: Path) -> None:
        """Model from GGUF source round-trips correctly."""
        async with Database(tmp_path / "test.db") as db:
            model = _make_model_info(
                id="local/codellama-7b.Q4_K_M.gguf",
                source=ModelSource.GGUF,
                parameters_b=7.0,
                base_vram_mb=5000,
            )
            await db.insert_model(model)
            got = await db.get_model("local/codellama-7b.Q4_K_M.gguf")

            assert got is not None
            assert got.source == ModelSource.GGUF

    async def test_model_with_empty_json_lists(self, tmp_path: Path) -> None:
        """Models with empty quantizations/capabilities round-trip as empty lists."""
        async with Database(tmp_path / "test.db") as db:
            model = _make_model_info(
                id="empty-lists",
                quantizations=[],
                capabilities=[],
            )
            await db.insert_model(model)
            got = await db.get_model("empty-lists")

            assert got is not None
            assert got.quantizations == []
            assert got.capabilities == []


# ---------------------------------------------------------------------------
# DB-level validation (SEC-D6, SEC-V5)
# ---------------------------------------------------------------------------


class TestDBValidation:
    """Tests for DB-level validation (SEC-D6, SEC-V5)."""

    async def test_insert_service_rejects_unknown_extra_config_keys(self, tmp_path: Path) -> None:
        """Extra config with unknown keys should be rejected."""
        async with Database(tmp_path / "test.db") as db:
            service = Service(
                id="test-svc",
                name="Test",
                driver=DriverType.VLLM,
                vram_mb=1000,
                sleep_mode=SleepMode.NONE,
                extra_config={"unknown_key": "value"},
            )
            with pytest.raises(ValueError, match="Unknown extra_config keys"):
                await db.insert_service(service)

    async def test_insert_service_rejects_oversized_vram(self, tmp_path: Path) -> None:
        """VRAM exceeding 1TB should be rejected."""
        async with Database(tmp_path / "test.db") as db:
            service = Service(
                id="test-svc",
                name="Test",
                driver=DriverType.VLLM,
                vram_mb=2_000_000,
                sleep_mode=SleepMode.NONE,
            )
            with pytest.raises(ValueError, match="exceeds maximum"):
                await db.insert_service(service)

    async def test_insert_service_rejects_negative_vram(self, tmp_path: Path) -> None:
        """Negative VRAM should be rejected."""
        async with Database(tmp_path / "test.db") as db:
            service = Service(
                id="test-svc",
                name="Test",
                driver=DriverType.VLLM,
                vram_mb=-1,
                sleep_mode=SleepMode.NONE,
            )
            with pytest.raises(ValueError, match="non-negative"):
                await db.insert_service(service)

    async def test_insert_service_accepts_valid_extra_config(self, tmp_path: Path) -> None:
        """Known extra_config keys should be accepted."""
        async with Database(tmp_path / "test.db") as db:
            service = Service(
                id="test-svc",
                name="Test",
                driver=DriverType.VLLM,
                vram_mb=1000,
                sleep_mode=SleepMode.NONE,
                extra_config={"context_size": "4096", "quantization": "q4"},
            )
            await db.insert_service(service)
            result = await db.get_service("test-svc")
            assert result is not None
            assert result.extra_config["context_size"] == "4096"

    async def test_get_model_rejects_invalid_id(self, tmp_path: Path) -> None:
        """Model IDs with path traversal should be rejected."""
        async with Database(tmp_path / "test.db") as db:
            with pytest.raises(ValueError, match="Invalid model_id"):
                await db.get_model("../passwd")

    async def test_insert_model_rejects_invalid_id(self, tmp_path: Path) -> None:
        """Model IDs with invalid chars should be rejected at insert."""
        async with Database(tmp_path / "test.db") as db:
            model = ModelInfo(
                id="../evil",
                source=ModelSource.LOCAL,
            )
            with pytest.raises(ValueError, match="Invalid model_id"):
                await db.insert_model(model)

    async def test_get_model_accepts_valid_id(self, tmp_path: Path) -> None:
        """Valid model IDs should work."""
        async with Database(tmp_path / "test.db") as db:
            result = await db.get_model("meta-llama/Llama-3.1-8B")
            assert result is None  # Not found but no validation error

    async def test_insert_service_accepts_empty_extra_config(self, tmp_path: Path) -> None:
        """Empty extra_config should be accepted."""
        async with Database(tmp_path / "test.db") as db:
            service = Service(
                id="test-empty",
                name="Test",
                driver=DriverType.VLLM,
                vram_mb=1000,
                sleep_mode=SleepMode.NONE,
                extra_config={},
            )
            await db.insert_service(service)
            result = await db.get_service("test-empty")
            assert result is not None

    async def test_validate_vram_zero_accepted(self, tmp_path: Path) -> None:
        """Zero VRAM should be accepted (for utility services)."""
        async with Database(tmp_path / "test.db") as db:
            service = Service(
                id="test-zero",
                name="Test",
                driver=DriverType.FASTAPI,
                vram_mb=0,
                sleep_mode=SleepMode.NONE,
            )
            await db.insert_service(service)
            result = await db.get_service("test-zero")
            assert result is not None


# ---------------------------------------------------------------------------
# Mode services population (_populate_mode_services)
# ---------------------------------------------------------------------------


class TestPopulateModeServices:
    """Tests for _populate_mode_services() — junction table → Mode.services."""

    async def test_list_modes_populates_services(self, tmp_path: Path) -> None:
        """list_modes() should return modes with populated services lists."""
        async with Database(tmp_path / "test.db") as db:
            svc_a = _make_service(id="svc-a", name="A")
            svc_b = _make_service(id="svc-b", name="B")
            for s in (svc_a, svc_b):
                await db.insert_service(s)

            mode = _make_mode(id="test-mode", services=["svc-a", "svc-b"])
            await db.insert_mode(mode)
            await db.set_mode_services("test-mode", ["svc-a", "svc-b"])

            modes = await db.list_modes()
            assert len(modes) == 1
            assert modes[0].services == ["svc-a", "svc-b"]

    async def test_get_mode_populates_services(self, tmp_path: Path) -> None:
        """get_mode() should return a mode with populated services list."""
        async with Database(tmp_path / "test.db") as db:
            svc = _make_service(id="svc-x")
            await db.insert_service(svc)

            mode = _make_mode(id="pop-mode")
            await db.insert_mode(mode)
            await db.set_mode_services("pop-mode", ["svc-x"])

            got = await db.get_mode("pop-mode")
            assert got is not None
            assert got.services == ["svc-x"]

    async def test_mode_with_no_services_has_empty_list(self, tmp_path: Path) -> None:
        """A mode with no junction rows should have services == []."""
        async with Database(tmp_path / "test.db") as db:
            mode = _make_mode(id="empty-mode")
            await db.insert_mode(mode)

            got = await db.get_mode("empty-mode")
            assert got is not None
            assert got.services == []

    async def test_mode_services_respect_start_order(self, tmp_path: Path) -> None:
        """Services should be ordered by start_order from the junction table."""
        async with Database(tmp_path / "test.db") as db:
            for sid in ("z-svc", "a-svc", "m-svc"):
                await db.insert_service(_make_service(id=sid))

            mode = _make_mode(id="ordered-mode")
            await db.insert_mode(mode)
            # z first (order 1), a second (order 2), m third (order 3)
            await db.set_mode_services(
                "ordered-mode",
                ["z-svc", "a-svc", "m-svc"],
                orders=[1, 2, 3],
            )

            got = await db.get_mode("ordered-mode")
            assert got is not None
            assert got.services == ["z-svc", "a-svc", "m-svc"]

    async def test_list_modes_multiple_modes_each_with_services(self, tmp_path: Path) -> None:
        """list_modes() must populate services for ALL modes, not just the first."""
        async with Database(tmp_path / "test.db") as db:
            for sid in ("embed", "chat", "code-llm"):
                await db.insert_service(_make_service(id=sid))

            for mid, svcs in [
                ("mode-a", ["embed"]),
                ("mode-b", ["embed", "chat"]),
                ("mode-c", ["code-llm"]),
            ]:
                await db.insert_mode(_make_mode(id=mid))
                await db.set_mode_services(mid, svcs)

            modes = await db.list_modes()
            by_id = {m.id: m for m in modes}

            assert by_id["mode-a"].services == ["embed"]
            assert by_id["mode-b"].services == ["embed", "chat"]
            assert by_id["mode-c"].services == ["code-llm"]

    async def test_services_survive_model_dump_json(self, tmp_path: Path) -> None:
        """Mode.model_dump(mode='json') must include populated services."""
        async with Database(tmp_path / "test.db") as db:
            await db.insert_service(_make_service(id="svc-1"))
            await db.insert_mode(_make_mode(id="test-mode"))
            await db.set_mode_services("test-mode", ["svc-1"])

            modes = await db.list_modes()
            dumped = modes[0].model_dump(mode="json")
            assert dumped["services"] == ["svc-1"], (
                "model_dump must preserve populated services"
            )

    async def test_services_survive_sanitize_roundtrip(self, tmp_path: Path) -> None:
        """Services must survive the full MCP serialization path."""
        from gpumod.mcp_tools import _sanitize_dict_names

        async with Database(tmp_path / "test.db") as db:
            await db.insert_service(_make_service(id="svc-a"))
            await db.insert_service(_make_service(id="svc-b", name="B"))
            await db.insert_mode(_make_mode(id="test-mode"))
            await db.set_mode_services("test-mode", ["svc-a", "svc-b"])

            modes = await db.list_modes()
            # Exact MCP tool code path
            result = [
                _sanitize_dict_names(m.model_dump(mode="json")) for m in modes
            ]
            assert result[0]["services"] == ["svc-a", "svc-b"]

    async def test_get_mode_services_consistent_with_list_modes(
        self, tmp_path: Path
    ) -> None:
        """get_mode_services() and list_modes() must return the same service IDs."""
        async with Database(tmp_path / "test.db") as db:
            for sid in ("svc-x", "svc-y"):
                await db.insert_service(_make_service(id=sid))

            await db.insert_mode(_make_mode(id="consistency-test"))
            await db.set_mode_services("consistency-test", ["svc-x", "svc-y"])

            # Via list_modes → _populate_mode_services
            modes = await db.list_modes()
            list_mode_services = [m for m in modes if m.id == "consistency-test"][0].services

            # Via get_mode_services (JOIN query)
            join_services = await db.get_mode_services("consistency-test")
            join_service_ids = [s.id for s in join_services]

            assert list_mode_services == join_service_ids

    async def test_populate_mode_services_after_adding_services(
        self, tmp_path: Path
    ) -> None:
        """Services added AFTER mode creation must still appear."""
        async with Database(tmp_path / "test.db") as db:
            await db.insert_service(_make_service(id="late-svc"))
            await db.insert_mode(_make_mode(id="dynamic-mode"))

            # Initially no services
            mode = await db.get_mode("dynamic-mode")
            assert mode is not None
            assert mode.services == []

            # Add service to mode
            await db.set_mode_services("dynamic-mode", ["late-svc"])

            # Must see the new service
            mode = await db.get_mode("dynamic-mode")
            assert mode is not None
            assert mode.services == ["late-svc"]

            # list_modes must also see it
            modes = await db.list_modes()
            found = [m for m in modes if m.id == "dynamic-mode"][0]
            assert found.services == ["late-svc"]
