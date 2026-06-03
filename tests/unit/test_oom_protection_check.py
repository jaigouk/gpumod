"""Tests for OOM protection drop-in check (gpumod-1lpe).

Verifies that the code-server and systemd-oomd drop-in configs are
present and contain the expected directives for host OOM protection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpumod.services.oom_protection_check import (
    OOMProtectionCheckResult,
    check_oom_protection,
    parse_directives,
    read_dropin,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestReadDropin:
    def test_reads_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "dropin.conf"
        f.write_text("[Service]\nMemoryMin=1G\n")
        assert read_dropin(f) == "[Service]\nMemoryMin=1G\n"

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        f = tmp_path / "does_not_exist"
        assert read_dropin(f) is None

    def test_returns_none_when_unreadable(self, tmp_path: Path) -> None:
        f = tmp_path / "unreadable.conf"
        f.write_text("[Service]\nMemoryMin=1G\n")
        f.chmod(0o000)
        try:
            assert read_dropin(f) is None
        finally:
            # Restore permissions so tmp_path cleanup works
            f.chmod(0o644)


class TestParseDirectives:
    def test_normal_content(self) -> None:
        content = "[Service]\nMemoryMin=1G\nMemoryLow=2G\nRestart=always\n"
        result = parse_directives(content)
        assert result["MemoryMin"] == "1G"
        assert result["MemoryLow"] == "2G"
        assert result["Restart"] == "always"

    def test_empty_content(self) -> None:
        result = parse_directives("")
        assert result == {}

    def test_mixed_section_headers(self) -> None:
        content = (
            "[Unit]\nDescription=test\n"
            "[Service]\nMemoryMin=1G\n"
            "[OOM]\nDefaultMemoryPressureDurationSec=20s\n"
        )
        result = parse_directives(content)
        # parse_directives should extract all Key=Value lines regardless of section
        assert result["Description"] == "test"
        assert result["MemoryMin"] == "1G"
        assert result["DefaultMemoryPressureDurationSec"] == "20s"

    def test_skips_comments_and_blank_lines(self) -> None:
        content = "# comment\n\n[Service]\nMemoryMin=1G\n"
        result = parse_directives(content)
        assert result == {"MemoryMin": "1G"}

    def test_handles_values_with_equals(self) -> None:
        # Edge case: value contains '='
        content = "[Service]\nEnvironment=FOO=bar\n"
        result = parse_directives(content)
        assert result["Environment"] == "FOO=bar"


class TestCheckOomProtection:
    def _write_code_server_dropin(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "[Service]\n"
            "MemoryMin=1G\n"
            "MemoryLow=2G\n"
            "MemorySwapMax=8G\n"
            "OOMScoreAdjust=-900\n"
            "ManagedOOMMemoryPressure=avoid\n"
            "ManagedOOMSwap=avoid\n"
            "Restart=always\n"
            "RestartSec=2s\n"
        )

    def _write_oomd_dropin(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "[OOM]\n"
            "DefaultMemoryPressureLimit=60%\n"
            "DefaultMemoryPressureDurationSec=20s\n"
            "SwapUsedLimit=90%\n"
        )

    def test_both_present_and_correct(self, tmp_path: Path) -> None:
        cs = tmp_path / "cs" / "10-oom-protect.conf"
        oomd = tmp_path / "oomd" / "gpumod.conf"
        self._write_code_server_dropin(cs)
        self._write_oomd_dropin(oomd)

        result = check_oom_protection(code_server_dropin=cs, oomd_dropin=oomd)

        assert result.ok is True
        assert result.missing_dropins == []
        assert result.wrong_values == []
        assert result.remediation is None

    def test_code_server_missing(self, tmp_path: Path) -> None:
        cs = tmp_path / "cs" / "10-oom-protect.conf"
        oomd = tmp_path / "oomd" / "gpumod.conf"
        self._write_oomd_dropin(oomd)

        result = check_oom_protection(code_server_dropin=cs, oomd_dropin=oomd)

        assert result.ok is False
        assert len(result.missing_dropins) == 1
        assert "code-server" in result.missing_dropins[0].lower()
        assert result.remediation is not None

    def test_oomd_missing(self, tmp_path: Path) -> None:
        cs = tmp_path / "cs" / "10-oom-protect.conf"
        oomd = tmp_path / "oomd" / "gpumod.conf"
        self._write_code_server_dropin(cs)

        result = check_oom_protection(code_server_dropin=cs, oomd_dropin=oomd)

        assert result.ok is False
        assert len(result.missing_dropins) == 1
        assert result.remediation is not None

    def test_both_missing(self, tmp_path: Path) -> None:
        cs = tmp_path / "cs" / "10-oom-protect.conf"
        oomd = tmp_path / "oomd" / "gpumod.conf"

        result = check_oom_protection(code_server_dropin=cs, oomd_dropin=oomd)

        assert result.ok is False
        assert len(result.missing_dropins) == 2
        assert result.remediation is not None

    def test_wrong_values_in_code_server(self, tmp_path: Path) -> None:
        cs = tmp_path / "cs" / "10-oom-protect.conf"
        oomd = tmp_path / "oomd" / "gpumod.conf"
        cs.parent.mkdir(parents=True, exist_ok=True)
        cs.write_text(
            "[Service]\n"
            "MemoryMin=512M\n"  # wrong
            "MemoryLow=2G\n"
            "OOMScoreAdjust=-900\n"
            "ManagedOOMMemoryPressure=avoid\n"
            "ManagedOOMSwap=avoid\n"
        )
        self._write_oomd_dropin(oomd)

        result = check_oom_protection(code_server_dropin=cs, oomd_dropin=oomd)

        assert result.ok is False
        assert len(result.wrong_values) >= 1
        assert any("MemoryMin" in v for v in result.wrong_values)
        assert result.remediation is not None

    def test_memoryswapmax_missing_is_flagged(self, tmp_path: Path) -> None:
        """Regression: drop-in predating the 2026-06-03 swap cap addition.

        Pre-existing installations have every directive except MemorySwapMax.
        The check must flag this so operators re-run install.sh and pick up
        the new bound (claude-code#19223 blast-radius cap).
        """
        cs = tmp_path / "cs" / "10-oom-protect.conf"
        oomd = tmp_path / "oomd" / "gpumod.conf"
        cs.parent.mkdir(parents=True, exist_ok=True)
        cs.write_text(
            "[Service]\n"
            "MemoryMin=1G\n"
            "MemoryLow=2G\n"
            # MemorySwapMax intentionally absent
            "OOMScoreAdjust=-900\n"
            "ManagedOOMMemoryPressure=avoid\n"
            "ManagedOOMSwap=avoid\n"
        )
        self._write_oomd_dropin(oomd)

        result = check_oom_protection(code_server_dropin=cs, oomd_dropin=oomd)

        assert result.ok is False
        assert any("MemorySwapMax" in v for v in result.wrong_values)
        assert result.remediation is not None

    def test_wrong_values_in_oomd(self, tmp_path: Path) -> None:
        cs = tmp_path / "cs" / "10-oom-protect.conf"
        oomd = tmp_path / "oomd" / "gpumod.conf"
        self._write_code_server_dropin(cs)
        oomd.parent.mkdir(parents=True, exist_ok=True)
        oomd.write_text(
            "[OOM]\nDefaultMemoryPressureDurationSec=30s\n"  # wrong
        )

        result = check_oom_protection(code_server_dropin=cs, oomd_dropin=oomd)

        assert result.ok is False
        assert len(result.wrong_values) >= 1
        assert any("DefaultMemoryPressureDurationSec" in v for v in result.wrong_values)
        assert result.remediation is not None

    def test_result_type(self, tmp_path: Path) -> None:
        cs = tmp_path / "cs" / "10-oom-protect.conf"
        oomd = tmp_path / "oomd" / "gpumod.conf"
        self._write_code_server_dropin(cs)
        self._write_oomd_dropin(oomd)

        result = check_oom_protection(code_server_dropin=cs, oomd_dropin=oomd)

        assert isinstance(result, OOMProtectionCheckResult)
        assert hasattr(result, "ok")
        assert hasattr(result, "missing_dropins")
        assert hasattr(result, "wrong_values")
        assert hasattr(result, "remediation")
