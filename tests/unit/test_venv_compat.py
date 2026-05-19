"""Tests for gpumod.services.venv_compat (gpumod-ng7).

Pure-library tests — no real venv needed. We build a synthetic
site-packages tree with hand-written dist-info METADATA files
under tmp_path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpumod.models import DriverType, Service, SleepMode
from gpumod.services.venv_compat import (
    CompatStatus,
    check_compat,
    find_venv_root,
    read_installed_versions,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dist_info(
    site_packages: Path, name: str, version: str, normalised_name: str | None = None
) -> None:
    """Create a synthetic dist-info dir with the minimal METADATA pip needs."""
    dist_name = (normalised_name or name).replace("-", "_")
    dist = site_packages / f"{dist_name}-{version}.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


def _make_synthetic_venv(
    tmp_path: Path, packages: dict[str, str], py_version: str = "3.13"
) -> Path:
    """Build <tmp>/venv/{bin/, lib/python{X}/site-packages/...} layout."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "vllm").write_text("#!/bin/sh\nexec true\n")
    site_packages = venv / "lib" / f"python{py_version}" / "site-packages"
    site_packages.mkdir(parents=True)
    for name, version in packages.items():
        _make_dist_info(site_packages, name, version)
    return venv


def _service_with_bin(bin_path: Path) -> Service:
    """Build a Service whose unit_vars.vllm_bin points at the synthetic venv."""
    return Service(
        id="vllm-test",
        name="test",
        driver=DriverType.VLLM,
        port=8000,
        vram_mb=2500,
        sleep_mode=SleepMode.NONE,
        extra_config={"unit_vars": {"vllm_bin": str(bin_path)}},
    )


# ---------------------------------------------------------------------------
# read_installed_versions
# ---------------------------------------------------------------------------


class TestReadInstalledVersions:
    def test_parses_dist_info_metadata(self, tmp_path: Path) -> None:
        venv = _make_synthetic_venv(
            tmp_path,
            {
                "vllm": "0.11.0",
                "transformers": "4.57.6",
                "huggingface-hub": "0.36.2",
            },
        )
        installed = read_installed_versions(venv)
        assert installed["vllm"] == "0.11.0"
        assert installed["transformers"] == "4.57.6"
        assert installed["huggingface-hub"] == "0.36.2"

    def test_normalises_package_names_to_lower_hyphen(self, tmp_path: Path) -> None:
        """Per PEP 503, 'huggingface_hub' and 'huggingface-hub' should match."""
        venv = _make_synthetic_venv(tmp_path, {})
        site = venv / "lib" / "python3.13" / "site-packages"
        # dist-info uses underscore; METADATA Name uses hyphen
        _make_dist_info(site, "huggingface-hub", "1.15.0")
        installed = read_installed_versions(venv)
        assert "huggingface-hub" in installed

    def test_returns_empty_when_no_site_packages(self, tmp_path: Path) -> None:
        venv = tmp_path / "empty"
        venv.mkdir()
        assert read_installed_versions(venv) == {}

    def test_handles_multiple_python_versions(self, tmp_path: Path) -> None:
        """If a venv has lib/python3.12 AND python3.13, prefer the highest."""
        venv = tmp_path / "venv"
        for v in ("3.12", "3.13"):
            sp = venv / "lib" / f"python{v}" / "site-packages"
            sp.mkdir(parents=True)
            _make_dist_info(sp, "vllm", "0.11.0" if v == "3.13" else "0.10.0")
        installed = read_installed_versions(venv)
        assert installed["vllm"] == "0.11.0"


# ---------------------------------------------------------------------------
# check_compat
# ---------------------------------------------------------------------------


class TestCheckCompat:
    def test_all_satisfied_is_ok(self) -> None:
        result = check_compat(
            declared={
                "vllm": ">=0.11.0,<0.12",
                "transformers": ">=4.55.2,<5.0",
            },
            installed={
                "vllm": "0.11.0",
                "transformers": "4.57.6",
            },
        )
        assert result.ok is True
        assert result.violations == []

    def test_drift_one_package(self) -> None:
        result = check_compat(
            declared={"transformers": "<5.0"},
            installed={"transformers": "5.8.1"},
        )
        assert result.ok is False
        assert len(result.violations) == 1
        v = result.violations[0]
        assert v.package == "transformers"
        assert v.installed == "5.8.1"
        assert v.constraint == "<5.0"
        assert v.status == CompatStatus.DRIFT

    def test_missing_declared_package(self) -> None:
        result = check_compat(
            declared={"vllm": ">=0.11.0"},
            installed={},  # vllm not installed at all
        )
        assert result.ok is False
        assert len(result.violations) == 1
        v = result.violations[0]
        assert v.package == "vllm"
        assert v.installed is None
        assert v.status == CompatStatus.MISSING

    def test_name_normalisation_avoids_false_missing(self) -> None:
        """huggingface_hub installed should satisfy huggingface-hub declared."""
        result = check_compat(
            declared={"huggingface-hub": "<1.0"},
            installed={"huggingface_hub": "0.36.2"},  # underscore variant
        )
        assert result.ok is True

    def test_ignores_installed_packages_not_in_contract(self) -> None:
        """Extra installed packages don't trigger violations."""
        result = check_compat(
            declared={"vllm": ">=0.11.0"},
            installed={
                "vllm": "0.11.0",
                "numpy": "2.0.0",  # not declared — ignored
            },
        )
        assert result.ok is True


# ---------------------------------------------------------------------------
# find_venv_root
# ---------------------------------------------------------------------------


class TestFindVenvRoot:
    def test_uses_service_unit_vars_vllm_bin(self, tmp_path: Path) -> None:
        venv = _make_synthetic_venv(tmp_path, {})
        service = _service_with_bin(venv / "bin" / "vllm")
        root = find_venv_root(service, settings={})
        assert root == venv

    def test_falls_back_to_settings_vllm_bin(self, tmp_path: Path) -> None:
        venv = _make_synthetic_venv(tmp_path, {})
        # Service has no unit_vars.vllm_bin override
        service = Service(
            id="vllm-test",
            name="test",
            driver=DriverType.VLLM,
            port=8000,
            vram_mb=2500,
        )
        settings = {"vllm_bin": str(venv / "bin" / "vllm")}
        assert find_venv_root(service, settings) == venv

    def test_returns_none_when_no_binary_path_known(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _name: None)
        service = Service(
            id="vllm-test",
            name="test",
            driver=DriverType.VLLM,
            port=8000,
            vram_mb=2500,
        )
        assert find_venv_root(service, settings={}) is None
