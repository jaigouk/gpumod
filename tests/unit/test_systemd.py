"""Tests for gpumod.services.systemd — systemctl wrapper with security validation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpumod.services.systemd import (
    ALLOWED_COMMANDS,
    SystemctlResult,
    SystemdError,
    _validate_command,
    _validate_unit_name,
    get_unit_state,
    is_active,
    restart,
    start,
    stop,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_process_mock(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> AsyncMock:
    """Create a mock that mimics asyncio.subprocess.Process."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout.encode(), stderr.encode()))
    proc.returncode = returncode
    return proc


# ---------------------------------------------------------------------------
# Unit name validation
# ---------------------------------------------------------------------------


class TestUnitNameValidation:
    """Tests for _validate_unit_name."""

    @pytest.mark.parametrize(
        "bad_name",
        [
            "foo;bar",
            "foo|bar",
            "foo$bar",
            "foo`bar",
            "foo&bar",
            "foo(bar",
            "foo)bar",
            "foo{bar",
            "foo}bar",
            "foo bar",
            "",
        ],
    )
    def test_reject_shell_metacharacters(self, bad_name: str) -> None:
        """Unit names containing shell metacharacters must be rejected."""
        with pytest.raises(ValueError, match="Invalid unit name"):
            _validate_unit_name(bad_name)

    @pytest.mark.parametrize(
        "good_name",
        [
            "vllm-embedding.service",
            "glm-code.service",
            "qwen3-asr@gpu0.service",
            "some_unit.service",
            "unit:scope.service",
            "a.b.c",
        ],
    )
    def test_accept_valid_names(self, good_name: str) -> None:
        """Valid unit names must be accepted without error."""
        _validate_unit_name(good_name)  # Should not raise


# ---------------------------------------------------------------------------
# Command validation
# ---------------------------------------------------------------------------


class TestCommandValidation:
    """Tests for _validate_command."""

    @pytest.mark.parametrize(
        "bad_cmd",
        [
            "daemon-reload",
            "mask",
            "unmask",
            "edit",
            "cat",
            "reset-failed",
        ],
    )
    def test_reject_unknown_commands(self, bad_cmd: str) -> None:
        """Commands not in the allowlist must be rejected."""
        with pytest.raises(ValueError, match="not allowed"):
            _validate_command(bad_cmd)

    @pytest.mark.parametrize("good_cmd", list(ALLOWED_COMMANDS))
    def test_accept_allowlisted_commands(self, good_cmd: str) -> None:
        """Allowlisted commands must be accepted."""
        _validate_command(good_cmd)  # Should not raise


# ---------------------------------------------------------------------------
# systemctl() core function
# ---------------------------------------------------------------------------


class TestSystemctl:
    """Tests for the core systemctl() async function."""

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_successful_start(self, mock_exec: MagicMock) -> None:
        """start() should complete without error on success."""
        mock_exec.return_value = _make_process_mock(returncode=0)
        await start("vllm-embedding.service")
        mock_exec.assert_called_once()

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_successful_stop(self, mock_exec: MagicMock) -> None:
        """stop() should complete without error on success."""
        mock_exec.return_value = _make_process_mock(returncode=0)
        await stop("vllm-embedding.service")
        mock_exec.assert_called_once()

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_successful_restart(self, mock_exec: MagicMock) -> None:
        """restart() should complete without error on success."""
        mock_exec.return_value = _make_process_mock(returncode=0)
        await restart("vllm-embedding.service")
        mock_exec.assert_called_once()

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_failed_command_raises_systemd_error(self, mock_exec: MagicMock) -> None:
        """A non-zero return code from systemctl must raise SystemdError."""
        mock_exec.return_value = _make_process_mock(returncode=1, stderr="Failed to start unit")
        with pytest.raises(SystemdError):
            await start("broken.service")


# ---------------------------------------------------------------------------
# is_active()
# ---------------------------------------------------------------------------


class TestIsActive:
    """Tests for is_active()."""

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_returns_true_for_active(self, mock_exec: MagicMock) -> None:
        """is_active() returns True when systemctl reports 'active'."""
        mock_exec.return_value = _make_process_mock(stdout="active\n", returncode=0)
        assert await is_active("vllm-embedding.service") is True

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_returns_false_for_inactive(self, mock_exec: MagicMock) -> None:
        """is_active() returns False when systemctl reports 'inactive'."""
        mock_exec.return_value = _make_process_mock(stdout="inactive\n", returncode=3)
        assert await is_active("vllm-embedding.service") is False

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_returns_false_for_failed_subprocess(self, mock_exec: MagicMock) -> None:
        """is_active() returns False when subprocess fails entirely."""
        mock_exec.side_effect = OSError("command not found")
        assert await is_active("vllm-embedding.service") is False

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_never_raises(self, mock_exec: MagicMock) -> None:
        """is_active() must never raise — returns False on any error."""
        mock_exec.side_effect = Exception("unexpected")
        result = await is_active("vllm-embedding.service")
        assert result is False


# ---------------------------------------------------------------------------
# get_unit_state()
# ---------------------------------------------------------------------------


class TestGetUnitState:
    """Tests for get_unit_state()."""

    @pytest.mark.parametrize("state", ["active", "inactive", "failed"])
    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_parses_known_states(self, mock_exec: MagicMock, state: str) -> None:
        """get_unit_state() correctly parses known states."""
        mock_exec.return_value = _make_process_mock(stdout=f"{state}\n", returncode=0)
        assert await get_unit_state("some.service") == state

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_returns_unknown_on_error(self, mock_exec: MagicMock) -> None:
        """get_unit_state() returns 'unknown' on subprocess error."""
        mock_exec.side_effect = OSError("nope")
        assert await get_unit_state("some.service") == "unknown"


# ---------------------------------------------------------------------------
# SystemdError
# ---------------------------------------------------------------------------


class TestSystemdError:
    """Tests for SystemdError exception class."""

    def test_error_contains_command_and_stderr(self) -> None:
        """SystemdError message should include command name and stderr."""
        result = SystemctlResult(return_code=1, stdout="", stderr="unit not found")
        err = SystemdError("start", result)

        assert "start" in str(err)
        assert "unit not found" in str(err)
        assert err.command == "start"
        assert err.result is result
