"""Tests for the `gpumod preflight` CLI subcommands (gpumod-ecr)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from gpumod.cli import app

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Typer CLI runner; stderr is captured separately in click >=8.2."""
    return CliRunner()


def _meminfo_with_avail_mb(tmp_path: Path, avail_mb: int) -> Path:
    """Write a /proc/meminfo-like file reporting the given MemAvailable in MB."""
    f = tmp_path / "meminfo"
    avail_kb = avail_mb * 1024
    f.write_text(
        f"MemTotal:       30000000 kB\nMemFree:           20000 kB\nMemAvailable:  {avail_kb} kB\n"
    )
    return f


@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Force each CLI invocation to use a fresh tmp DB so tests are independent."""
    monkeypatch.setenv("GPUMOD_DATA_DIR", str(tmp_path / "data"))


@pytest.fixture
def patch_meminfo_comfortable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Default-state fixture: 28 GB MemAvailable, plenty of headroom."""
    meminfo_path = _meminfo_with_avail_mb(tmp_path, 28_000)
    monkeypatch.setattr("gpumod.services.ram._MEMINFO_PATH", meminfo_path)
    return meminfo_path


def _make_fake_service(service_id: str = "qwen36", vram_mb: int = 22_000):
    """Build a minimal Service object for tests that bypass DB loading."""
    from gpumod.models import DriverType, Service, SleepMode

    return Service(
        id=service_id,
        name=f"Test {service_id}",
        driver=DriverType.LLAMACPP,
        port=7099,
        vram_mb=vram_mb,
        sleep_mode=SleepMode.NONE,
        health_endpoint="/health",
    )


@pytest.fixture
def stub_registry_get(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ServiceRegistry.get so tests don't need a real DB row.

    Returns a service for known IDs; raises KeyError for unknown ones —
    matches the real method's behaviour.
    """

    async def fake_get(_self: object, service_id: str):
        if service_id == "qwen36":
            return _make_fake_service("qwen36", vram_mb=22_000)
        if service_id == "small-service":
            return _make_fake_service("small-service", vram_mb=2_500)
        msg = f"Service not found: {service_id!r}"
        raise KeyError(msg)

    monkeypatch.setattr(
        "gpumod.services.registry.ServiceRegistry.get",
        fake_get,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_meminfo_comfortable", "stub_registry_get")
class TestPreflightRamSafe:
    """`gpumod preflight ram` exits 0 when MemAvailable is comfortable."""

    def test_exits_zero_when_ram_is_comfortable(self, runner: CliRunner) -> None:
        # 28 GB available; 22 GB-VRAM service needs ~11.6 GB
        result = runner.invoke(app, ["preflight", "ram", "--service-id", "qwen36"])
        assert result.exit_code == 0, result.stdout + (result.stderr or "")


@pytest.mark.usefixtures("stub_registry_get")
class TestPreflightRamUnsafe:
    """`gpumod preflight ram` exits 1 with stderr diagnostic when unsafe."""

    def test_exits_one_when_under_hard_floor(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        meminfo = _meminfo_with_avail_mb(tmp_path, 3_000)  # below hard floor
        monkeypatch.setattr("gpumod.services.ram._MEMINFO_PATH", meminfo)
        result = runner.invoke(app, ["preflight", "ram", "--service-id", "qwen36"])
        assert result.exit_code == 1
        assert "hard floor" in (result.stderr or "").lower()
        assert "3000" in (result.stderr or "")

    def test_exits_one_when_under_headroom(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Above hard floor but below 22 GB * 0.3 + 5 GB = 11.6 GB headroom
        meminfo = _meminfo_with_avail_mb(tmp_path, 8_000)
        monkeypatch.setattr("gpumod.services.ram._MEMINFO_PATH", meminfo)
        result = runner.invoke(app, ["preflight", "ram", "--service-id", "qwen36"])
        assert result.exit_code == 1
        assert "8000" in (result.stderr or "")


@pytest.mark.usefixtures("stub_registry_get")
class TestPreflightRamHardFloorOverride:
    """`--hard-floor` CLI option overrides the module default."""

    def test_hard_floor_override_raises_threshold(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # 8 GB available; default 6 GB floor would allow it (vram=2500 small)
        meminfo = _meminfo_with_avail_mb(tmp_path, 8_000)
        monkeypatch.setattr("gpumod.services.ram._MEMINFO_PATH", meminfo)
        # Without override: passes (8 GB > 6 GB floor; 2.5 GB headroom)
        ok = runner.invoke(app, ["preflight", "ram", "--service-id", "small-service"])
        assert ok.exit_code == 0
        # With --hard-floor 10000: now 8 GB < 10 GB, must fail
        fail = runner.invoke(
            app,
            [
                "preflight",
                "ram",
                "--service-id",
                "small-service",
                "--hard-floor",
                "10000",
            ],
        )
        assert fail.exit_code == 1
        assert "10000" in (fail.stderr or "")


@pytest.mark.usefixtures("patch_meminfo_comfortable", "stub_registry_get")
class TestPreflightRamUnknownService:
    """`gpumod preflight ram` exits 2 when the service ID isn't registered."""

    def test_exits_two_on_unknown_service(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["preflight", "ram", "--service-id", "does-not-exist"])
        assert result.exit_code == 2
        assert "does-not-exist" in (result.stderr or "")
