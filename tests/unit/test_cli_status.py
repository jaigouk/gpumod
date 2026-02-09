"""Tests for gpumod.cli — status and init top-level commands."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer.testing

from gpumod.cli import app
from gpumod.models import (
    DriverType,
    GPUInfo,
    Service,
    ServiceInfo,
    ServiceState,
    ServiceStatus,
    SystemStatus,
    VRAMUsage,
)
from gpumod.templates.modes import ModeSyncResult
from gpumod.templates.presets import PresetSyncResult

runner = typer.testing.CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_cli_auto_sync():
    """Disable auto-sync in cli_context to preserve test fixtures."""
    mock_preset_result = PresetSyncResult(inserted=0, updated=0, unchanged=0, deleted=0)
    mock_mode_result = ModeSyncResult(inserted=0, updated=0, unchanged=0, deleted=0)

    with (
        patch("gpumod.cli.sync_presets", new=AsyncMock(return_value=mock_preset_result)),
        patch("gpumod.cli.sync_modes", new=AsyncMock(return_value=mock_mode_result)),
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(
    *,
    id: str = "svc-1",
    name: str = "Test Service",
    driver: DriverType = DriverType.VLLM,
    port: int | None = 8000,
    vram_mb: int = 4096,
) -> Service:
    return Service(id=id, name=name, driver=driver, port=port, vram_mb=vram_mb)


def _make_service_info(
    *,
    id: str = "svc-1",
    name: str = "Test Service",
    driver: DriverType = DriverType.VLLM,
    port: int | None = 8000,
    vram_mb: int = 4096,
    state: ServiceState = ServiceState.RUNNING,
    status_vram_mb: int | None = None,
) -> ServiceInfo:
    svc = _make_service(id=id, name=name, driver=driver, port=port, vram_mb=vram_mb)
    resolved_vram = status_vram_mb if status_vram_mb is not None else vram_mb
    status = ServiceStatus(state=state, vram_mb=resolved_vram)
    return ServiceInfo(service=svc, status=status)


def _make_gpu_info(
    *,
    name: str = "NVIDIA RTX 4090",
    vram_total_mb: int = 24576,
) -> GPUInfo:
    return GPUInfo(name=name, vram_total_mb=vram_total_mb)


def _make_vram_usage(
    *,
    total_mb: int = 24576,
    used_mb: int = 8192,
    free_mb: int = 16384,
) -> VRAMUsage:
    return VRAMUsage(total_mb=total_mb, used_mb=used_mb, free_mb=free_mb)


def _make_full_system_status(
    *,
    gpu: GPUInfo | None = None,
    vram: VRAMUsage | None = None,
    current_mode: str | None = "inference",
    services: list[ServiceInfo] | None = None,
) -> SystemStatus:
    if gpu is None:
        gpu = _make_gpu_info()
    if vram is None:
        vram = _make_vram_usage()
    if services is None:
        services = [
            _make_service_info(
                id="vllm-7b",
                name="vLLM 7B",
                vram_mb=4096,
                state=ServiceState.RUNNING,
            ),
            _make_service_info(
                id="llama-13b",
                name="LlamaCpp 13B",
                driver=DriverType.LLAMACPP,
                vram_mb=8192,
                state=ServiceState.SLEEPING,
            ),
        ]
    return SystemStatus(gpu=gpu, vram=vram, current_mode=current_mode, services=services)


def _make_mock_context() -> MagicMock:
    """Create a mock AppContext with all backend services mocked."""
    ctx = MagicMock()
    ctx.manager = MagicMock()
    ctx.manager.get_status = AsyncMock(return_value=_make_full_system_status())
    ctx.db = MagicMock()
    ctx.db.connect = AsyncMock()
    ctx.db.close = AsyncMock()
    ctx.db.insert_service = AsyncMock()
    ctx.preset_loader = MagicMock()
    ctx.preset_loader.discover_presets = MagicMock(return_value=[])
    ctx.preset_loader.to_service = MagicMock()
    return ctx


# ---------------------------------------------------------------------------
# status command tests
# ---------------------------------------------------------------------------


class TestStatusCommand:
    """Tests for `gpumod status` top-level command."""

    def test_status_shows_system_overview(self) -> None:
        """Status command with full SystemStatus shows GPU and service overview."""
        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        # Should show GPU name
        assert "RTX 4090" in result.output
        # Should show service names
        assert "vLLM 7B" in result.output
        assert "LlamaCpp 13B" in result.output

    def test_status_shows_gpu_info(self) -> None:
        """Status command shows GPU name and VRAM information."""
        gpu = _make_gpu_info(name="NVIDIA A100", vram_total_mb=81920)
        system_status = _make_full_system_status(gpu=gpu)
        mock_ctx = _make_mock_context()
        mock_ctx.manager.get_status = AsyncMock(return_value=system_status)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "A100" in result.output
        assert "81920" in result.output

    def test_status_shows_current_mode(self) -> None:
        """Status command shows the current mode name."""
        system_status = _make_full_system_status(current_mode="inference")
        mock_ctx = _make_mock_context()
        mock_ctx.manager.get_status = AsyncMock(return_value=system_status)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "inference" in result.output.lower()

    def test_status_shows_service_list(self) -> None:
        """Status command shows service names and their states."""
        services = [
            _make_service_info(id="svc-a", name="Alpha", state=ServiceState.RUNNING),
            _make_service_info(id="svc-b", name="Beta", state=ServiceState.STOPPED),
        ]
        system_status = _make_full_system_status(services=services)
        mock_ctx = _make_mock_context()
        mock_ctx.manager.get_status = AsyncMock(return_value=system_status)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Alpha" in result.output
        assert "Beta" in result.output
        assert "running" in result.output.lower()
        assert "stopped" in result.output.lower()

    def test_status_visual_flag_renders_vram_bar(self) -> None:
        """Status command with --visual flag uses the visualization module."""
        mock_ctx = _make_mock_context()

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            patch("gpumod.cli.StatusPanel") as mock_panel_cls,
        ):
            mock_panel_instance = MagicMock()
            mock_panel_instance.render = MagicMock(return_value=MagicMock())
            mock_panel_cls.return_value = mock_panel_instance

            result = runner.invoke(app, ["status", "--visual"])

        assert result.exit_code == 0
        mock_panel_instance.render.assert_called_once()

    def test_status_json_flag_outputs_json(self) -> None:
        """Status command with --json outputs valid JSON."""
        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["status", "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)
        assert "gpu" in parsed
        assert "services" in parsed
        assert parsed["gpu"]["name"] == "NVIDIA RTX 4090"

    def test_status_no_gpu_shows_warning(self) -> None:
        """Status command with gpu=None shows a warning message."""
        system_status = SystemStatus(gpu=None, vram=None, current_mode=None, services=[])
        mock_ctx = _make_mock_context()
        mock_ctx.manager.get_status = AsyncMock(return_value=system_status)

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "no gpu" in result.output.lower()


# ---------------------------------------------------------------------------
# init command tests
# ---------------------------------------------------------------------------


class TestInitCommand:
    """Tests for `gpumod init` top-level command."""

    def test_init_creates_database(self) -> None:
        """Init command connects to the database."""
        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        # create_context was called (which connects the db)

    def test_init_discovers_presets(self) -> None:
        """Init command calls discover_presets on the preset loader."""
        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        mock_ctx.preset_loader.discover_presets.assert_called_once()

    def test_init_loads_presets_into_db(self) -> None:
        """Init command inserts each preset as a service into the database."""
        from gpumod.models import PresetConfig

        preset1 = PresetConfig(
            id="preset-1", name="Preset One", driver=DriverType.VLLM, vram_mb=4096
        )
        preset2 = PresetConfig(
            id="preset-2", name="Preset Two", driver=DriverType.LLAMACPP, vram_mb=2048
        )
        svc1 = _make_service(id="preset-1", name="Preset One")
        svc2 = _make_service(id="preset-2", name="Preset Two")

        mock_ctx = _make_mock_context()
        mock_ctx.preset_loader.discover_presets = MagicMock(return_value=[preset1, preset2])
        mock_ctx.preset_loader.to_service = MagicMock(side_effect=[svc1, svc2])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert mock_ctx.db.insert_service.await_count == 2

    def test_init_shows_summary(self) -> None:
        """Init command shows the number of presets found."""
        from gpumod.models import PresetConfig

        preset1 = PresetConfig(
            id="preset-1", name="Preset One", driver=DriverType.VLLM, vram_mb=4096
        )
        preset2 = PresetConfig(
            id="preset-2", name="Preset Two", driver=DriverType.LLAMACPP, vram_mb=2048
        )
        svc1 = _make_service(id="preset-1", name="Preset One")
        svc2 = _make_service(id="preset-2", name="Preset Two")

        mock_ctx = _make_mock_context()
        mock_ctx.preset_loader.discover_presets = MagicMock(return_value=[preset1, preset2])
        mock_ctx.preset_loader.to_service = MagicMock(side_effect=[svc1, svc2])

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "2" in result.output

    def test_init_custom_db_path(self) -> None:
        """Init command with --db-path option passes custom path to create_context."""
        mock_ctx = _make_mock_context()
        mock_create = AsyncMock(return_value=mock_ctx)

        with patch("gpumod.cli.create_context", new=mock_create):
            result = runner.invoke(app, ["init", "--db-path", "/tmp/custom.db"])

        assert result.exit_code == 0
        # Verify create_context was called with the custom db_path
        call_kwargs = mock_create.call_args
        assert call_kwargs is not None
        # The db_path should be a Path object matching our input
        from pathlib import Path

        called_path = call_kwargs[1].get("db_path") or call_kwargs[0][0]
        assert str(called_path) == "/tmp/custom.db" or called_path == Path("/tmp/custom.db")

    def test_init_idempotent(self) -> None:
        """Init command handles IntegrityError on duplicate insert gracefully."""
        import sqlite3

        from gpumod.models import PresetConfig

        preset1 = PresetConfig(
            id="preset-1", name="Preset One", driver=DriverType.VLLM, vram_mb=4096
        )
        svc1 = _make_service(id="preset-1", name="Preset One")

        mock_ctx = _make_mock_context()
        mock_ctx.preset_loader.discover_presets = MagicMock(return_value=[preset1])
        mock_ctx.preset_loader.to_service = MagicMock(return_value=svc1)
        # Simulate IntegrityError on insert (duplicate key)
        mock_ctx.db.insert_service = AsyncMock(
            side_effect=sqlite3.IntegrityError("UNIQUE constraint failed"),
        )

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["init"])

        # Should not crash — handles duplicate gracefully
        assert result.exit_code == 0
        # Should mention skipping or already existing
        output_lower = result.output.lower()
        assert "skip" in output_lower or "already" in output_lower or "1" in result.output

    def test_init_closes_db(self) -> None:
        """Init command closes the database connection on completion."""
        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        mock_ctx.db.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestRunAsyncTypeSafety
# ---------------------------------------------------------------------------


class TestRunAsyncTypeSafety:
    """Tests that run_async is properly typed and preserves return type."""

    def test_run_async_returns_typed_value(self) -> None:
        """run_async should return the correct type from the coroutine."""
        from gpumod.cli import run_async

        async def return_int() -> int:
            return 42

        result = run_async(return_int())
        assert result == 42
        assert isinstance(result, int)

    def test_run_async_returns_string(self) -> None:
        """run_async preserves string return type."""
        from gpumod.cli import run_async

        async def return_str() -> str:
            return "hello"

        result = run_async(return_str())
        assert result == "hello"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# cli_context tests
# ---------------------------------------------------------------------------


class TestCliContext:
    """Tests for the cli_context() async context manager."""

    async def test_cli_context_yields_app_context(self) -> None:
        """cli_context() yields a valid AppContext."""
        from gpumod.cli import cli_context

        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            async with cli_context() as ctx:
                assert ctx is mock_ctx

    async def test_cli_context_closes_db(self) -> None:
        """cli_context() closes the database on normal exit."""
        from gpumod.cli import cli_context

        mock_ctx = _make_mock_context()

        with patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)):
            async with cli_context() as ctx:
                pass  # Normal exit

        ctx.db.close.assert_awaited_once()

    async def test_cli_context_closes_db_on_exception(self) -> None:
        """cli_context() closes the database even when an exception occurs."""
        from gpumod.cli import cli_context

        mock_ctx = _make_mock_context()

        with (
            patch("gpumod.cli.create_context", new=AsyncMock(return_value=mock_ctx)),
            pytest.raises(RuntimeError, match="test error"),
        ):
            async with cli_context() as ctx:
                raise RuntimeError("test error")

        ctx.db.close.assert_awaited_once()

    async def test_cli_context_forwards_db_path(self) -> None:
        """cli_context() forwards the db_path argument to create_context."""
        from pathlib import Path

        from gpumod.cli import cli_context

        mock_ctx = _make_mock_context()
        mock_create = AsyncMock(return_value=mock_ctx)

        with patch("gpumod.cli.create_context", new=mock_create):
            async with cli_context(db_path=Path("/tmp/test.db")):
                pass

        mock_create.assert_awaited_once_with(db_path=Path("/tmp/test.db"))


# ---------------------------------------------------------------------------
# Structural tests — cli_context() usage enforcement
# ---------------------------------------------------------------------------


class TestCliContextUsageEnforcement:
    """Verify that status() and init() use cli_context(), not create_context() directly."""

    _CLI_PY = Path(__file__).resolve().parents[2] / "src" / "gpumod" / "cli.py"

    def _get_function_body_source(self, func_name: str) -> str:
        """Extract the source of a top-level function from cli.py using AST."""
        source = self._CLI_PY.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                start = node.body[0].lineno - 1
                end = node.end_lineno
                lines = source.splitlines()[start:end]
                return "\n".join(lines)
        raise ValueError(f"Function {func_name!r} not found in cli.py")

    def test_status_does_not_call_create_context_directly(self) -> None:
        """status() must use cli_context(), not call create_context() directly."""
        body = self._get_function_body_source("status")
        assert "create_context" not in body, (
            "status() should use cli_context() instead of create_context()"
        )

    def test_init_does_not_call_create_context_directly(self) -> None:
        """init() must use cli_context(), not call create_context() directly."""
        body = self._get_function_body_source("init")
        assert "create_context" not in body, (
            "init() should use cli_context() instead of create_context()"
        )

    def test_status_uses_cli_context(self) -> None:
        """status() must contain an 'async with cli_context' call."""
        body = self._get_function_body_source("status")
        assert "cli_context" in body, "status() must use cli_context()"

    def test_init_uses_cli_context(self) -> None:
        """init() must contain an 'async with cli_context' call."""
        body = self._get_function_body_source("init")
        assert "cli_context" in body, "init() must use cli_context()"


class TestAllCliModulesUseCliContext:
    """Verify all 6 CLI sub-modules use cli_context() and never call create_context() directly."""

    _SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "gpumod"
    _CLI_MODULES = [
        "cli_service.py",
        "cli_mode.py",
        "cli_model.py",
        "cli_template.py",
        "cli_plan.py",
        "cli_simulate.py",
    ]

    @pytest.mark.parametrize("module", _CLI_MODULES)
    def test_module_does_not_import_create_context(self, module: str) -> None:
        """CLI sub-module should not import create_context directly."""
        source = (self._SRC_DIR / module).read_text()
        assert "create_context" not in source, (
            f"{module} should import cli_context, not create_context"
        )

    @pytest.mark.parametrize("module", _CLI_MODULES)
    def test_module_uses_cli_context(self, module: str) -> None:
        """CLI sub-module should use cli_context()."""
        source = (self._SRC_DIR / module).read_text()
        assert "cli_context" in source, f"{module} must use cli_context()"


class TestDependencyUpperBounds:
    """Verify all dependencies in pyproject.toml have upper bounds."""

    _PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
    _UPPER_BOUND_RE = re.compile(r"<\d")

    def _parse_deps(self, section_marker: str) -> list[str]:
        """Extract dependency strings from a section of pyproject.toml."""
        content = self._PYPROJECT.read_text()
        in_section = False
        deps: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if section_marker in line:
                in_section = True
                continue
            if in_section:
                if stripped == "]":
                    break
                if stripped.startswith('"') and ">=" in stripped:
                    deps.append(stripped.strip('",'))
        return deps

    def test_runtime_deps_have_upper_bounds(self) -> None:
        """All runtime dependencies must have an upper version bound (<X.0)."""
        deps = self._parse_deps("dependencies = [")
        for dep in deps:
            assert self._UPPER_BOUND_RE.search(dep), (
                f"Runtime dependency missing upper bound: {dep}"
            )

    def test_dev_deps_have_upper_bounds(self) -> None:
        """All dev dependencies must have an upper version bound (<X.0)."""
        deps = self._parse_deps("dev = [")
        for dep in deps:
            assert self._UPPER_BOUND_RE.search(dep), f"Dev dependency missing upper bound: {dep}"
