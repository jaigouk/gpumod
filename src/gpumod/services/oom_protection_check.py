"""OOM protection drop-in health check (gpumod-1lpe).

Verifies that the code-server and systemd-oomd drop-in configs are
present and contain the expected directives for host OOM protection.

Background: when gpumod loads large LLM models, the resulting memory
pressure can cause systemd-oomd to kill code-server (the operator's
IDE). Drop-in overrides in ``/etc/systemd/system/code-server@.service.d/``
and ``/etc/systemd/oomd.conf.d/`` protect code-server by:
  - Guaranteeing it a minimum resident set (MemoryMin/MemoryLow)
  - Lowering its OOM score (OOMScoreAdjust=-900)
  - Opting out of oomd PSI-based killing (ManagedOOMMemoryPressure=avoid)
  - Tuning oomd's reaction time (DefaultMemoryPressureDurationSec=20s)

This module:
- Reads drop-in files from injectable paths
- Parses ``Key=Value`` directives from systemd unit content
- Compares against expected values
- Returns a structured result with remediation guidance

History:
- 2026-06-03: Added ``MemorySwapMax=8G`` to the expected code-server
  directives. Bounds swap blast-radius from a leaky VS Code extension host
  (claude-code#19223 — Anthropic's Claude Code extension grew RSS to 15.7
  GiB over hours of idle in code-server). See
  ``docs/research/20260525_oom_protection_findings/FINDINGS.md`` for the
  cgroup memory-protection rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_CODE_SERVER_DROPIN = Path(
    "/etc/systemd/system/code-server@.service.d/10-oom-protect.conf"
)
_DEFAULT_OOMD_DROPIN = Path("/etc/systemd/oomd.conf.d/gpumod.conf")

# Expected content snippets to verify (key directives only — order-independent)
EXPECTED_CODE_SERVER_DIRECTIVES: dict[str, str] = {
    "MemoryMin": "1G",
    "MemoryLow": "2G",
    "MemorySwapMax": "8G",
    "OOMScoreAdjust": "-900",
    "ManagedOOMMemoryPressure": "avoid",
    "ManagedOOMSwap": "avoid",
}

EXPECTED_OOMD_DIRECTIVES: dict[str, str] = {
    "DefaultMemoryPressureDurationSec": "20s",
}


@dataclass(frozen=True)
class OOMProtectionCheckResult:
    """Outcome of the OOM protection drop-in check.

    Attributes
    ----------
    ok:
        True when both drop-ins are present with correct directive values.
    missing_dropins:
        Human-readable list of drop-in files that are missing.
    wrong_values:
        Human-readable list of directives with unexpected values.
    remediation:
        Operator-facing fix-it instruction when ``ok`` is False;
        ``None`` when ``ok`` is True.
    """

    ok: bool
    missing_dropins: list[str] = field(default_factory=list)
    wrong_values: list[str] = field(default_factory=list)
    remediation: str | None = None


def read_dropin(path: Path) -> str | None:
    """Read a drop-in config file.

    Returns the file text, or ``None`` on any OS-level error (missing,
    permissions, etc.).
    """
    try:
        return path.read_text()
    except OSError:
        return None


def parse_directives(content: str) -> dict[str, str]:
    """Parse ``Key=Value`` lines from systemd unit content.

    Skips section headers (``[Service]``), comments (``#``), and blank
    lines. Returns a dict mapping directive names to their values.
    When a value contains ``=``, only the first ``=`` is used as the
    split point.
    """
    result: dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "[")):
            continue
        if "=" in stripped:
            key, _, value = stripped.partition("=")
            result[key] = value
    return result


def _check_directives(
    content: str,
    expected: dict[str, str],
    label: str,
) -> list[str]:
    """Compare parsed directives against expected values.

    Returns a list of human-readable mismatch descriptions.
    """
    parsed = parse_directives(content)
    problems: list[str] = []
    for key, expected_val in expected.items():
        actual_val = parsed.get(key)
        if actual_val is None:
            problems.append(f"{label}: {key} missing (expected {expected_val})")
        elif actual_val != expected_val:
            problems.append(f"{label}: {key}={actual_val} (expected {expected_val})")
    return problems


def check_oom_protection(
    code_server_dropin: Path | None = None,
    oomd_dropin: Path | None = None,
) -> OOMProtectionCheckResult:
    """Check that OOM protection drop-ins are installed and correct.

    Uses the ``_DEFAULT_*`` paths when args are ``None``.
    """
    cs_path = code_server_dropin if code_server_dropin is not None else _DEFAULT_CODE_SERVER_DROPIN
    oomd_path = oomd_dropin if oomd_dropin is not None else _DEFAULT_OOMD_DROPIN

    missing: list[str] = []
    wrong: list[str] = []

    cs_content = read_dropin(cs_path)
    if cs_content is None:
        missing.append(f"code-server drop-in: {cs_path}")
    else:
        wrong.extend(_check_directives(cs_content, EXPECTED_CODE_SERVER_DIRECTIVES, "code-server"))

    oomd_content = read_dropin(oomd_path)
    if oomd_content is None:
        missing.append(f"oomd drop-in: {oomd_path}")
    else:
        wrong.extend(_check_directives(oomd_content, EXPECTED_OOMD_DIRECTIVES, "oomd"))

    if not missing and not wrong:
        return OOMProtectionCheckResult(ok=True)

    remediation = _build_remediation(missing, wrong)
    return OOMProtectionCheckResult(
        ok=False,
        missing_dropins=missing,
        wrong_values=wrong,
        remediation=remediation,
    )


def _build_remediation(missing: list[str], wrong: list[str]) -> str:
    """Build operator-facing remediation text."""
    lines: list[str] = []
    if missing:
        lines.append("Missing drop-in files:")
        lines.extend(f"  - {m}" for m in missing)
    if wrong:
        lines.append("Incorrect directive values:")
        lines.extend(f"  - {w}" for w in wrong)
    lines.append("")
    lines.append("Install the drop-ins:")
    lines.append("  sudo scripts/oom-protection/install.sh")
    lines.append("")
    lines.append("Or manually copy the configs and run:")
    lines.append("  sudo systemctl daemon-reload")
    return "\n".join(lines)
