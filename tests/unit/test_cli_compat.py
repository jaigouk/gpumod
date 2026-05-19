"""Tests for `gpumod compat list|show` CLI (gpumod-ng7)."""

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


@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPUMOD_DATA_DIR", str(tmp_path / "data"))


class TestCompatList:
    def test_lists_shipped_profiles(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["compat", "list"])
        assert result.exit_code == 0
        # vllm-0.11 ships with ng7
        assert "vllm-0.11" in result.stdout


class TestCompatShow:
    def test_shows_named_profile(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["compat", "show", "vllm-0.11"])
        assert result.exit_code == 0
        # Profile YAML content should be visible
        assert "vllm" in result.stdout
        assert "transformers" in result.stdout

    def test_unknown_profile_exits_one(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["compat", "show", "does-not-exist"])
        assert result.exit_code == 1
        assert "does-not-exist" in (result.stderr or "")
