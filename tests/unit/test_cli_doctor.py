"""Tests for `gpumod doctor venv` CLI (gpumod-ng7)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from gpumod.cli import app
from gpumod.models import DriverType, Service, SleepMode

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPUMOD_DATA_DIR", str(tmp_path / "data"))


def _make_synthetic_venv(tmp_path: Path, packages: dict[str, str]) -> Path:
    """Build <tmp>/venv/{bin/vllm, lib/python3.13/site-packages/...}."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "vllm").write_text("#!/bin/sh\n")
    site = venv / "lib" / "python3.13" / "site-packages"
    site.mkdir(parents=True)
    for name, version in packages.items():
        dist_name = name.replace("-", "_")
        dist = site / f"{dist_name}-{version}.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    return venv


def _make_service(*, compat: dict[str, str] | None, vllm_bin: str | None = None) -> Service:
    extra: dict[str, dict[str, str]] = {}
    if vllm_bin:
        extra["unit_vars"] = {"vllm_bin": vllm_bin}
    return Service(
        id="vllm-test",
        name="vLLM test",
        driver=DriverType.VLLM,
        port=8000,
        vram_mb=2500,
        sleep_mode=SleepMode.NONE,
        compat=compat,
        extra_config=extra,
    )


@pytest.fixture
def stub_registry_factory(monkeypatch: pytest.MonkeyPatch):
    """Return a callable that registers ``{service_id: Service}`` in the stub."""

    def install(services: dict[str, Service]) -> None:
        async def fake_get(_self: object, service_id: str):
            if service_id in services:
                return services[service_id]
            msg = f"Service not found: {service_id!r}"
            raise KeyError(msg)

        monkeypatch.setattr(
            "gpumod.services.registry.ServiceRegistry.get",
            fake_get,
        )

    return install


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDoctorVenvOK:
    def test_exits_zero_when_compat_ok(
        self,
        runner: CliRunner,
        tmp_path: Path,
        stub_registry_factory,
    ) -> None:
        venv = _make_synthetic_venv(tmp_path, {"vllm": "0.11.0", "transformers": "4.57.6"})
        service = _make_service(
            compat={"vllm": ">=0.11.0,<0.12", "transformers": "<5.0"},
            vllm_bin=str(venv / "bin" / "vllm"),
        )
        stub_registry_factory({"vllm-test": service})

        result = runner.invoke(app, ["doctor", "venv", "--service-id", "vllm-test"])
        assert result.exit_code == 0, result.stdout + (result.stderr or "")

    def test_exits_zero_when_no_compat_declared(
        self,
        runner: CliRunner,
        tmp_path: Path,
        stub_registry_factory,
    ) -> None:
        """No contract = nothing to validate = no-op success."""
        venv = _make_synthetic_venv(tmp_path, {})
        service = _make_service(compat=None, vllm_bin=str(venv / "bin" / "vllm"))
        stub_registry_factory({"vllm-test": service})

        result = runner.invoke(app, ["doctor", "venv", "--service-id", "vllm-test"])
        assert result.exit_code == 0


class TestDoctorVenvDrift:
    def test_exits_one_on_drift(
        self,
        runner: CliRunner,
        tmp_path: Path,
        stub_registry_factory,
    ) -> None:
        # Installed transformers 5.x; contract requires <5.0
        venv = _make_synthetic_venv(tmp_path, {"transformers": "5.8.1"})
        service = _make_service(
            compat={"transformers": "<5.0"},
            vllm_bin=str(venv / "bin" / "vllm"),
        )
        stub_registry_factory({"vllm-test": service})

        result = runner.invoke(app, ["doctor", "venv", "--service-id", "vllm-test"])
        assert result.exit_code == 1
        stderr = result.stderr or ""
        assert "transformers" in stderr
        assert "5.8.1" in stderr or "drift" in stderr.lower()

    def test_exits_one_on_missing(
        self,
        runner: CliRunner,
        tmp_path: Path,
        stub_registry_factory,
    ) -> None:
        venv = _make_synthetic_venv(tmp_path, {})  # vllm not installed
        service = _make_service(
            compat={"vllm": ">=0.11.0"},
            vllm_bin=str(venv / "bin" / "vllm"),
        )
        stub_registry_factory({"vllm-test": service})

        result = runner.invoke(app, ["doctor", "venv", "--service-id", "vllm-test"])
        assert result.exit_code == 1
        stderr = result.stderr or ""
        assert "vllm" in stderr
        assert "missing" in stderr.lower()


class TestDoctorVenvUnknownService:
    def test_exits_two_on_unknown_service(
        self,
        runner: CliRunner,
        stub_registry_factory,
    ) -> None:
        stub_registry_factory({})
        result = runner.invoke(app, ["doctor", "venv", "--service-id", "ghost"])
        assert result.exit_code == 2
        assert "ghost" in (result.stderr or "")


class TestDoctorVenvProfileOverride:
    def test_profile_flag_uses_shipped_profile_when_no_inline_compat(
        self,
        runner: CliRunner,
        tmp_path: Path,
        stub_registry_factory,
    ) -> None:
        """--profile <name> loads the shipped profile when service.compat is None."""
        # Profile vllm-0.11 declares transformers <5.0; install 5.x to trigger DRIFT.
        venv = _make_synthetic_venv(tmp_path, {"transformers": "5.8.1"})
        service = _make_service(compat=None, vllm_bin=str(venv / "bin" / "vllm"))
        stub_registry_factory({"vllm-test": service})

        result = runner.invoke(
            app,
            ["doctor", "venv", "--service-id", "vllm-test", "--profile", "vllm-0.11"],
        )
        assert result.exit_code == 1
        assert "transformers" in (result.stderr or "")


class TestDoctorVenvLoadsSettingsFromDB:
    """Regression for gpumod-032: cli_doctor previously passed settings={}.

    That made `find_venv_root` skip the DB-stored ``vllm_bin`` and fall
    through to ``shutil.which("vllm")``, which on a misconfigured host
    returns a different (broken) interpreter than the one the systemd
    unit actually launches — turning the safeguard into a false alarm.
    """

    def test_finds_venv_via_db_setting_when_no_unit_var(
        self,
        runner: CliRunner,
        tmp_path: Path,
        stub_registry_factory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Synthetic venv with vllm 0.11 satisfying the compat block.
        venv = _make_synthetic_venv(tmp_path, {"vllm": "0.11.0"})
        # Service has NO unit_vars.vllm_bin — resolution must use settings.
        service = _make_service(compat={"vllm": ">=0.11.0,<0.12"}, vllm_bin=None)
        stub_registry_factory({"vllm-test": service})

        # Stub the DB-settings loader so it returns our synthetic venv path.
        async def fake_build_settings(_db: object) -> dict[str, str]:
            return {"vllm_bin": str(venv / "bin" / "vllm")}

        monkeypatch.setattr(
            "gpumod.services.unit_installer._build_settings",
            fake_build_settings,
        )

        result = runner.invoke(app, ["doctor", "venv", "--service-id", "vllm-test"])
        assert result.exit_code == 0, result.stdout + (result.stderr or "")
