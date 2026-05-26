"""Kernel sysctl health check for GPU stability (gpumod-ej0).

Verifies that ``vm.min_free_kbytes`` is high enough to keep contiguous
high-order pages available for CUDA pinned-memory allocations on this
host.

Background: ``cudaHostAlloc`` requires physically contiguous, page-locked
RAM. When fragmentation builds up over hours (sustained CI / mmap /
crashlooping services), the next pinned allocation can fail — and the
NVIDIA driver hangs silently instead of returning an error. The reference
30 GiB host has documented 9+ such freezes (see
``docs/research/20260525_oom_protection_findings/FINDINGS.md``).

The default ``vm.min_free_kbytes`` on this host is ~66 MiB (scales from
the kernel formula). Bumping it to ~1 GiB tells the kernel to keep that
much free at all times — which translates to many more free high-order
pages available, dramatically reducing the failure probability.

This module:
- Reads ``/proc/sys/vm/min_free_kbytes`` from an injectable path
- Compares against a recommended threshold (default 1 GiB)
- Returns a structured result with remediation pointing at the durable
  ``/etc/sysctl.d/`` fix
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# 1 GiB. Conservative starting point that meaningfully reduces freeze
# probability without significantly cutting MemAvailable. Bump to 2 GiB
# only if freezes still occur with this value.
RECOMMENDED_MIN_FREE_KBYTES = 1048576

_DEFAULT_PROC_PATH = Path("/proc/sys/vm/min_free_kbytes")


@dataclass(frozen=True)
class SysctlCheckResult:
    """Outcome of the vm.min_free_kbytes check.

    Attributes
    ----------
    ok:
        True when ``current >= threshold``.
    current:
        The value read from ``/proc/sys/vm/min_free_kbytes``, or ``None``
        when the read failed.
    threshold:
        The configured threshold being compared against.
    remediation:
        Operator-facing fix-it instruction when ``ok`` is False;
        ``None`` when ``ok`` is True.
    """

    ok: bool
    current: int | None
    threshold: int
    remediation: str | None


def read_min_free_kbytes(path: Path | None = None) -> int | None:
    """Read the current ``vm.min_free_kbytes`` value.

    Returns ``None`` when the path is missing or contents are malformed.
    Looks up ``_DEFAULT_PROC_PATH`` at call time when ``path`` is None so
    monkey-patching the module attribute works in tests.
    """
    resolved = path if path is not None else _DEFAULT_PROC_PATH
    try:
        return int(resolved.read_text().strip())
    except (OSError, ValueError):
        return None


def check_min_free_kbytes(
    current: int | None,
    threshold: int = RECOMMENDED_MIN_FREE_KBYTES,
) -> SysctlCheckResult:
    """Compare a current ``vm.min_free_kbytes`` value against the threshold."""
    if current is None:
        return SysctlCheckResult(
            ok=False,
            current=None,
            threshold=threshold,
            remediation=(
                "Could not read /proc/sys/vm/min_free_kbytes — unable to "
                "verify kernel fragmentation safeguard. Ensure /proc is "
                "mounted and the path is readable."
            ),
        )

    if current >= threshold:
        return SysctlCheckResult(
            ok=True,
            current=current,
            threshold=threshold,
            remediation=None,
        )

    deficit_mib = (threshold - current) // 1024
    return SysctlCheckResult(
        ok=False,
        current=current,
        threshold=threshold,
        remediation=(
            f"vm.min_free_kbytes={current} kB is below the recommended "
            f"{threshold} kB (~{threshold // 1024} MiB). Short by "
            f"~{deficit_mib} MiB.\n"
            f"\n"
            f"Apply the persistent fix:\n"
            f"  sudo tee /etc/sysctl.d/99-gpumod-stability.conf <<'EOF'\n"
            f"  vm.min_free_kbytes={threshold}\n"
            f"  EOF\n"
            f"  sudo sysctl --system\n"
            f"\n"
            f"Or use the installer:\n"
            f"  sudo scripts/install-gpumod-sysctl.sh\n"
            f"\n"
            f"Background: low vm.min_free_kbytes lets contiguous "
            f"high-order pages fragment, causing CUDA pinned-memory "
            f"allocations (cudaHostAlloc) to hang the NVIDIA driver. See "
            f"docs/research/20260525_oom_protection_findings/FINDINGS.md "
            f"for the broader freeze class."
        ),
    )
