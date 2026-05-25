"""Tests for `gpumod doctor oom-protection` CLI (gpumod-1lpe)."""

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


def _patch_dropins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    code_server_content: str | None = None,
    oomd_content: str | None = None,
) -> None:
    """Inject synthetic drop-in paths via module-level defaults."""
    cs_path = tmp_path / "cs" / "10-oom-protect.conf"
    oomd_path = tmp_path / "oomd" / "gpumod.conf"

    if code_server_content is not None:
        cs_path.parent.mkdir(parents=True, exist_ok=True)
        cs_path.write_text(code_server_content)

    if oomd_content is not None:
        oomd_path.parent.mkdir(parents=True, exist_ok=True)
        oomd_path.write_text(oomd_content)

    monkeypatch.setattr(
        "gpumod.services.oom_protection_check._DEFAULT_CODE_SERVER_DROPIN",
        cs_path,
    )
    monkeypatch.setattr(
        "gpumod.services.oom_protection_check._DEFAULT_OOMD_DROPIN",
        oomd_path,
    )


_CORRECT_CS = (
    "[Service]\n"
    "MemoryMin=1G\n"
    "MemoryLow=2G\n"
    "OOMScoreAdjust=-900\n"
    "ManagedOOMMemoryPressure=avoid\n"
    "ManagedOOMSwap=avoid\n"
    "Restart=always\n"
    "RestartSec=2s\n"
)

_CORRECT_OOMD = (
    "[OOM]\n"
    "DefaultMemoryPressureLimit=60%\n"
    "DefaultMemoryPressureDurationSec=20s\n"
    "SwapUsedLimit=90%\n"
)


class TestDoctorOomProtectionCommand:
    def test_passes_when_both_dropins_correct(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _patch_dropins(
            monkeypatch,
            tmp_path,
            code_server_content=_CORRECT_CS,
            oomd_content=_CORRECT_OOMD,
        )
        result = runner.invoke(app, ["doctor", "oom-protection"])
        assert result.exit_code == 0
        assert "OK" in result.stdout

    def test_fails_when_dropin_missing(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Only oomd present, code-server missing
        _patch_dropins(
            monkeypatch,
            tmp_path,
            oomd_content=_CORRECT_OOMD,
        )
        result = runner.invoke(app, ["doctor", "oom-protection"])
        assert result.exit_code == 1
        output = result.stderr or result.output
        assert "missing" in output.lower() or "Missing" in output

    def test_fails_when_values_wrong(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        wrong_cs = (
            "[Service]\n"
            "MemoryMin=512M\n"  # wrong
            "MemoryLow=2G\n"
            "OOMScoreAdjust=-900\n"
            "ManagedOOMMemoryPressure=avoid\n"
            "ManagedOOMSwap=avoid\n"
        )
        _patch_dropins(
            monkeypatch,
            tmp_path,
            code_server_content=wrong_cs,
            oomd_content=_CORRECT_OOMD,
        )
        result = runner.invoke(app, ["doctor", "oom-protection"])
        assert result.exit_code == 1
        output = result.stderr or result.output
        assert "MemoryMin" in output
