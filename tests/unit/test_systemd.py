"""Tests for gpumod.services.systemd — systemctl wrapper with security validation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpumod.services.systemd import (
    ALLOWED_COMMANDS,
    SystemctlResult,
    SystemdError,
    _validate_unit_name,
    _validate_command,
    get_unit_state,
    is_active,
    journal_logs,
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

    @pytest.mark.parametrize(
        ("func", "expected_cmd"),
        [
            (start, "start"),
            (stop, "stop"),
            (restart, "restart"),
        ],
    )
    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_all_commands_use_user_scope_not_sudo(
        self, mock_exec: MagicMock, func, expected_cmd: str,
    ) -> None:
        """start/stop/restart must use 'systemctl --user', never 'sudo'."""
        mock_exec.return_value = _make_process_mock(returncode=0)
        await func("test.service")
        args = mock_exec.call_args[0]
        assert args[0] == "systemctl", f"{func.__name__} must call systemctl"
        assert args[1] == "--user", f"{func.__name__} must use --user flag"
        assert args[2] == expected_cmd
        assert "sudo" not in args, f"{func.__name__} must never use sudo"

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
    async def test_uses_user_scope_not_sudo(self, mock_exec: MagicMock) -> None:
        """is_active() must use 'systemctl --user', never 'sudo'."""
        mock_exec.return_value = _make_process_mock(stdout="active\n", returncode=0)
        await is_active("test.service")
        args = mock_exec.call_args[0]
        assert args[:3] == ("systemctl", "--user", "is-active")
        assert "sudo" not in args

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

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_uses_user_scope_not_sudo(self, mock_exec: MagicMock) -> None:
        """get_unit_state() must use 'systemctl --user', never 'sudo'."""
        mock_exec.return_value = _make_process_mock(stdout="active\n", returncode=0)
        await get_unit_state("test.service")
        args = mock_exec.call_args[0]
        assert args[:3] == ("systemctl", "--user", "is-active")
        assert "sudo" not in args

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


# ---------------------------------------------------------------------------
# Source-level guard: prevent sudo from ever returning
# ---------------------------------------------------------------------------


class TestNoSudoInSource:
    """Defence-in-depth: scan systemd.py source to prevent sudo regression."""

    def test_source_code_never_contains_sudo(self) -> None:
        """The systemd module source must never reference 'sudo'."""
        import inspect

        import gpumod.services.systemd as mod

        source = inspect.getsource(mod)
        # Allow the word "sudo" only in comments/docstrings that explicitly
        # say "no sudo" or "never sudo". Strip those before checking.
        lines = source.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments/docstrings that explicitly say sudo is not used
            if "no ``sudo``" in stripped or "No ``sudo``" in stripped:
                continue
            if "never" in stripped.lower() and "sudo" in stripped.lower():
                continue
            assert "sudo" not in stripped.lower() or stripped.startswith("#"), (
                f"Line {i} references 'sudo' — all systemctl calls must use "
                f"'systemctl --user', never 'sudo': {stripped!r}"
            )


# ---------------------------------------------------------------------------
# journal_logs()
# ---------------------------------------------------------------------------


class TestJournalLogs:
    """Tests for journal_logs() — fetch recent journal entries for a unit."""

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_returns_log_lines(self, mock_exec: MagicMock) -> None:
        """journal_logs() returns a list of log line strings."""
        log_output = "line one\nline two\nline three\n"
        mock_exec.return_value = _make_process_mock(stdout=log_output, returncode=0)

        result = await journal_logs("vllm-chat.service")

        assert result == ["line one", "line two", "line three"]

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_calls_journalctl_with_user_flag(self, mock_exec: MagicMock) -> None:
        """journal_logs() must call journalctl --user -u <unit>."""
        mock_exec.return_value = _make_process_mock(stdout="", returncode=0)

        await journal_logs("test.service")

        args = mock_exec.call_args[0]
        assert args[0] == "journalctl"
        assert "--user" in args
        assert "-u" in args
        assert "test.service" in args

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_default_lines_is_20(self, mock_exec: MagicMock) -> None:
        """journal_logs() defaults to 20 lines."""
        mock_exec.return_value = _make_process_mock(stdout="", returncode=0)

        await journal_logs("test.service")

        args = mock_exec.call_args[0]
        n_idx = list(args).index("-n")
        assert args[n_idx + 1] == "20"

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_custom_lines_count(self, mock_exec: MagicMock) -> None:
        """journal_logs() passes custom line count."""
        mock_exec.return_value = _make_process_mock(stdout="", returncode=0)

        await journal_logs("test.service", lines=50)

        args = mock_exec.call_args[0]
        n_idx = list(args).index("-n")
        assert args[n_idx + 1] == "50"

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_lines_clamped_to_min_1(self, mock_exec: MagicMock) -> None:
        """journal_logs() clamps lines to minimum of 1."""
        mock_exec.return_value = _make_process_mock(stdout="", returncode=0)

        await journal_logs("test.service", lines=0)

        args = mock_exec.call_args[0]
        n_idx = list(args).index("-n")
        assert args[n_idx + 1] == "1"

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_lines_clamped_to_max_200(self, mock_exec: MagicMock) -> None:
        """journal_logs() clamps lines to maximum of 200."""
        mock_exec.return_value = _make_process_mock(stdout="", returncode=0)

        await journal_logs("test.service", lines=999)

        args = mock_exec.call_args[0]
        n_idx = list(args).index("-n")
        assert args[n_idx + 1] == "200"

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_returns_empty_list_on_subprocess_error(
        self, mock_exec: MagicMock,
    ) -> None:
        """journal_logs() returns [] on subprocess failure, never raises."""
        mock_exec.side_effect = OSError("command not found")

        result = await journal_logs("test.service")

        assert result == []

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_returns_empty_list_on_nonzero_exit(
        self, mock_exec: MagicMock,
    ) -> None:
        """journal_logs() returns [] when journalctl exits non-zero."""
        mock_exec.return_value = _make_process_mock(
            stdout="", stderr="No journal files", returncode=1,
        )

        result = await journal_logs("test.service")

        assert result == []

    async def test_rejects_invalid_unit_name(self) -> None:
        """journal_logs() returns [] for invalid unit names (no injection)."""
        result = await journal_logs("foo;rm -rf /")

        assert result == []

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_strips_trailing_empty_lines(self, mock_exec: MagicMock) -> None:
        """journal_logs() strips trailing empty lines from output."""
        mock_exec.return_value = _make_process_mock(
            stdout="line one\nline two\n\n\n", returncode=0,
        )

        result = await journal_logs("test.service")

        assert result == ["line one", "line two"]

    @patch("gpumod.services.systemd.asyncio.create_subprocess_exec")
    async def test_includes_no_pager_flag(self, mock_exec: MagicMock) -> None:
        """journal_logs() must pass --no-pager to prevent blocking."""
        mock_exec.return_value = _make_process_mock(stdout="", returncode=0)

        await journal_logs("test.service")

        args = mock_exec.call_args[0]
        assert "--no-pager" in args
