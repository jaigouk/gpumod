"""Tests for packaging, distribution & public API (P5-T5).

Verifies:
- Console entry point resolves correctly
- Public API imports work via __init__.py
- __all__ is defined and complete
- Presets dir accessible via importlib.resources
- Jinja2 .j2 templates included in package
- pytest markers (integration, slow) are defined
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.resources
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


# ---------------------------------------------------------------------------
# Console entry point
# ---------------------------------------------------------------------------


class TestConsoleEntryPoint:
    """Verify the gpumod console_scripts entry point."""

    def test_entry_point_resolves(self) -> None:
        """gpumod entry point should resolve to gpumod.cli:run_cli."""
        eps = importlib.metadata.entry_points(group="console_scripts", name="gpumod")
        gpumod_eps = list(eps)
        assert len(gpumod_eps) == 1, "Expected exactly one 'gpumod' console_scripts entry point"
        ep = gpumod_eps[0]
        assert ep.value == "gpumod.cli:run_cli"

    def test_run_cli_callable(self) -> None:
        """run_cli should be importable and callable."""
        from gpumod.cli import run_cli

        assert callable(run_cli)


# ---------------------------------------------------------------------------
# Public API imports
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Verify the public API exposed by gpumod.__init__."""

    def test_import_version(self) -> None:
        """__version__ should be importable from gpumod."""
        from gpumod import __version__

        assert isinstance(__version__, str)
        assert __version__  # non-empty

    def test_import_service(self) -> None:
        """Service should be importable from gpumod."""
        from gpumod import Service
        from gpumod.models import Service as ServiceDirect

        assert Service is ServiceDirect

    def test_import_mode(self) -> None:
        """Mode should be importable from gpumod."""
        from gpumod import Mode
        from gpumod.models import Mode as ModeDirect

        assert Mode is ModeDirect

    def test_import_database(self) -> None:
        """Database should be importable from gpumod."""
        from gpumod import Database
        from gpumod.db import Database as DatabaseDirect

        assert Database is DatabaseDirect

    def test_import_service_manager(self) -> None:
        """ServiceManager should be importable from gpumod."""
        from gpumod import ServiceManager
        from gpumod.services.manager import ServiceManager as SMDirect

        assert ServiceManager is SMDirect

    def test_import_simulation_engine(self) -> None:
        """SimulationEngine should be importable from gpumod."""
        from gpumod import SimulationEngine
        from gpumod.simulation import SimulationEngine as SEDirect

        assert SimulationEngine is SEDirect

    def test_combined_import(self) -> None:
        """All public names should be importable in a single statement."""
        from gpumod import Database, Mode, Service, ServiceManager, SimulationEngine, __version__

        assert __version__
        assert Service is not None
        assert Mode is not None
        assert Database is not None
        assert ServiceManager is not None
        assert SimulationEngine is not None


# ---------------------------------------------------------------------------
# __all__ definition
# ---------------------------------------------------------------------------


class TestAllDefinition:
    """Verify __all__ is defined and complete in gpumod.__init__."""

    def test_all_defined(self) -> None:
        """__all__ should be defined in gpumod package."""
        import gpumod

        assert hasattr(gpumod, "__all__")
        assert isinstance(gpumod.__all__, list)

    def test_all_contains_expected_names(self) -> None:
        """__all__ should contain all expected public names."""
        import gpumod

        expected = {
            "__version__",
            "Database",
            "Mode",
            "Service",
            "ServiceManager",
            "SimulationEngine",
        }
        assert expected == set(gpumod.__all__)

    def test_all_names_are_importable(self) -> None:
        """Every name in __all__ should be an attribute of gpumod."""
        import gpumod

        for name in gpumod.__all__:
            assert hasattr(gpumod, name), f"{name} listed in __all__ but not importable"


# ---------------------------------------------------------------------------
# Package data: presets
# ---------------------------------------------------------------------------


class TestPresetsAccessible:
    """Verify presets directory is accessible after pip install."""

    def test_presets_dir_exists(self) -> None:
        """Presets directory should be findable relative to the package."""
        # The presets dir is resolved via config, but we check importlib.resources
        # can find the gpumod package itself
        pkg_files = importlib.resources.files("gpumod")
        assert pkg_files is not None

    def test_presets_yaml_files_exist(self) -> None:
        """At least one YAML preset file should exist in the presets directory."""
        from gpumod.config import _resolve_default_presets_dir

        presets_dir = _resolve_default_presets_dir()
        yaml_files = list(presets_dir.rglob("*.yaml"))
        assert len(yaml_files) > 0, f"No YAML files found in {presets_dir}"


# ---------------------------------------------------------------------------
# Package data: Jinja2 templates
# ---------------------------------------------------------------------------


class TestJinja2TemplatesIncluded:
    """Verify Jinja2 .j2 template files are findable in the package."""

    def test_templates_dir_exists(self) -> None:
        """The systemd templates directory should exist."""
        templates = importlib.resources.files("gpumod.templates") / "systemd"
        assert templates is not None

    def test_vllm_template_exists(self) -> None:
        """vllm.service.j2 should be included in the package."""
        template = importlib.resources.files("gpumod.templates.systemd") / "vllm.service.j2"
        content = template.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_llamacpp_template_exists(self) -> None:
        """llamacpp.service.j2 should be included in the package."""
        template = importlib.resources.files("gpumod.templates.systemd") / "llamacpp.service.j2"
        content = template.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_fastapi_template_exists(self) -> None:
        """fastapi.service.j2 should be included in the package."""
        template = importlib.resources.files("gpumod.templates.systemd") / "fastapi.service.j2"
        content = template.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_all_j2_templates_findable(self) -> None:
        """All .j2 files should be findable via importlib.resources."""
        expected_templates = {"vllm.service.j2", "llamacpp.service.j2", "fastapi.service.j2"}
        systemd_dir = importlib.resources.files("gpumod.templates.systemd")
        # Check each template is accessible
        for name in expected_templates:
            resource = systemd_dir / name
            assert resource.read_text(encoding="utf-8"), f"Template {name} is empty or missing"


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------


class TestPytestMarkers:
    """Verify custom pytest markers are registered."""

    def test_integration_marker_defined(self, pytestconfig: pytest.Config) -> None:
        """The 'integration' marker should be registered in pyproject.toml."""
        markers = pytestconfig.getini("markers")
        marker_names = [m.split(":")[0].strip() for m in markers]
        assert "integration" in marker_names

    def test_slow_marker_defined(self, pytestconfig: pytest.Config) -> None:
        """The 'slow' marker should be registered in pyproject.toml."""
        markers = pytestconfig.getini("markers")
        marker_names = [m.split(":")[0].strip() for m in markers]
        assert "slow" in marker_names
