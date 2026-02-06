"""Systemd helper — safe, async wrapper around systemctl.

All subprocess calls use ``asyncio.create_subprocess_exec`` (never ``shell=True``).
Unit names are validated against a strict regex rejecting shell metacharacters.
Commands are restricted to an explicit allowlist.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        "start",
        "stop",
        "restart",
        "is-active",
        "status",
        "enable",
        "disable",
        "show",
    }
)

UNIT_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_@:.\-]+$")


@dataclass
class SystemctlResult:
    """Result of a ``systemctl`` invocation."""

    return_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """Return True when the command exited with code 0."""
        return self.return_code == 0


class SystemdError(Exception):
    """Raised when a ``systemctl`` command fails."""

    def __init__(self, command: str, result: SystemctlResult) -> None:
        self.command = command
        self.result = result
        super().__init__(f"systemctl {command} failed (rc={result.return_code}): {result.stderr}")


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def _validate_unit_name(unit: str) -> None:
    """Raise ``ValueError`` if *unit* contains shell metacharacters or is empty."""
    if not unit or not UNIT_NAME_PATTERN.match(unit):
        msg = f"Invalid unit name: {unit!r}"
        raise ValueError(msg)


def _validate_command(command: str) -> None:
    """Raise ``ValueError`` if *command* is not in the allowlist."""
    if command not in ALLOWED_COMMANDS:
        msg = f"Command {command!r} not allowed. Must be one of {sorted(ALLOWED_COMMANDS)}"
        raise ValueError(msg)


# ------------------------------------------------------------------
# Core async functions
# ------------------------------------------------------------------


async def systemctl(
    command: str,
    unit: str,
    *,
    timeout: float = 30.0,
) -> SystemctlResult:
    """Run ``systemctl <command> <unit>`` and return the result.

    Raises
    ------
    SystemdError
        If the command exits with a non-zero return code.
    ValueError
        If *command* or *unit* fails validation.
    """
    _validate_command(command)
    _validate_unit_name(unit)

    proc = await asyncio.create_subprocess_exec(
        "systemctl",
        command,
        unit,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout_bytes, stderr_bytes = await asyncio.wait_for(
        proc.communicate(),
        timeout=timeout,
    )

    result = SystemctlResult(
        return_code=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout_bytes.decode() if stdout_bytes else "",
        stderr=stderr_bytes.decode() if stderr_bytes else "",
    )

    if not result.success:
        raise SystemdError(command, result)

    return result


async def is_active(unit: str) -> bool:
    """Return ``True`` if *unit* is ``active``. Never raises."""
    try:
        _validate_unit_name(unit)
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "is-active",
            unit,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, _ = await proc.communicate()
        return stdout_bytes.decode().strip() == "active"
    except Exception:  # noqa: BLE001
        return False


async def get_unit_state(unit: str) -> str:
    """Return the unit state string (e.g. ``active``, ``inactive``, ``failed``).

    Returns ``"unknown"`` on any error.
    """
    try:
        _validate_unit_name(unit)
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "is-active",
            unit,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, _ = await proc.communicate()
        return stdout_bytes.decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


async def start(unit: str, *, timeout: float = 30.0) -> None:
    """Start a systemd unit."""
    await systemctl("start", unit, timeout=timeout)


async def stop(unit: str, *, timeout: float = 30.0) -> None:
    """Stop a systemd unit."""
    await systemctl("stop", unit, timeout=timeout)


async def restart(unit: str, *, timeout: float = 30.0) -> None:
    """Restart a systemd unit."""
    await systemctl("restart", unit, timeout=timeout)
