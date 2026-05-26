"""Tests for `gpumod doctor sysctl` CLI (gpumod-ej0)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from gpumod.cli import app

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _patch_proc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str | None) -> None:
    """Inject a synthetic /proc/sys/vm/min_free_kbytes via the module path."""
    if value is None:
        # Make read_min_free_kbytes return None (file missing)
        bad = tmp_path / "missing"
        monkeypatch.setattr(
            "gpumod.services.sysctl_check._DEFAULT_PROC_PATH",
            bad,
        )
        return
    f = tmp_path / "min_free_kbytes"
    f.write_text(value)
    monkeypatch.setattr(
        "gpumod.services.sysctl_check._DEFAULT_PROC_PATH",
        f,
    )


class TestDoctorSysctlCommand:
    def test_passes_when_above_threshold(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _patch_proc(monkeypatch, tmp_path, "1048576")  # exactly at threshold
        result = runner.invoke(app, ["doctor", "sysctl"])
        assert result.exit_code == 0
        assert "OK" in result.stdout
        assert "1048576" in result.stdout

    def test_fails_when_below_threshold(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _patch_proc(monkeypatch, tmp_path, "67584")  # default on 30 GiB reference host
        result = runner.invoke(app, ["doctor", "sysctl"])
        assert result.exit_code == 1
        # Remediation goes to stderr — CliRunner mixes by default
        output = result.stderr or result.output
        assert "67584" in output
        assert "/etc/sysctl.d/" in output

    def test_fails_when_unreadable(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _patch_proc(monkeypatch, tmp_path, None)
        result = runner.invoke(app, ["doctor", "sysctl"])
        assert result.exit_code == 1
        output = result.stderr or result.output
        assert "min_free_kbytes" in output.lower() or "Could not" in output

    def test_custom_threshold_via_flag(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _patch_proc(monkeypatch, tmp_path, "524288")
        # With default threshold (1 GiB) this would fail. Pass --threshold so it passes.
        result = runner.invoke(app, ["doctor", "sysctl", "--threshold", "524288"])
        assert result.exit_code == 0

    def test_help_documents_threshold_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["doctor", "sysctl", "--help"])
        assert result.exit_code == 0
        assert "threshold" in result.output.lower()
