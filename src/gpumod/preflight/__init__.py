"""Preflight validation system for gpumod services.

Provides extensible preflight checks that run before service startup
to validate configuration and prevent runtime failures.

Usage:
    from gpumod.preflight import run_preflight

    results, has_errors = await run_preflight(service)
    if has_errors:
        print("Preflight failed!")

Or with custom checks:
    from gpumod.preflight import PreflightRunner, TokenizerCheck

    runner = PreflightRunner(checks=[TokenizerCheck()])
    results = await runner.run_all(service)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpumod.preflight.base import CheckResult, PreflightCheck
from gpumod.preflight.model_file import (
    DiskSpaceChecker,
    DiskSpaceResult,
    DownloadAbortedError,
    DownloadInfo,
    InsufficientDiskSpaceError,
    ModelDownloader,
    ModelFileCheck,
)
from gpumod.preflight.tokenizer import TokenizerCheck

if TYPE_CHECKING:
    from gpumod.models import Service

__all__ = [
    "CheckResult",
    "DiskSpaceChecker",
    "DiskSpaceResult",
    "DownloadAbortedError",
    "DownloadInfo",
    "InsufficientDiskSpaceError",
    "ModelDownloader",
    "ModelFileCheck",
    "PreflightCheck",
    "PreflightRunner",
    "TokenizerCheck",
    "run_preflight",
]


class PreflightRunner:
    """Orchestrates running multiple preflight checks.

    Follows Single Responsibility - only coordinates check execution,
    not the checks themselves.

    Parameters
    ----------
    checks:
        List of preflight checks to run.
    """

    def __init__(self, checks: list[PreflightCheck]) -> None:
        self.checks = checks

    @classmethod
    def default(cls) -> PreflightRunner:
        """Create a runner with all default checks.

        Returns
        -------
        PreflightRunner:
            Runner configured with standard checks.
        """
        return cls(checks=[ModelFileCheck(), TokenizerCheck()])

    async def run_all(self, service: Service) -> dict[str, CheckResult]:
        """Run all registered checks on a service.

        Parameters
        ----------
        service:
            The service to validate.

        Returns
        -------
        dict[str, CheckResult]:
            Mapping of check name to result.
        """
        results: dict[str, CheckResult] = {}
        for check in self.checks:
            results[check.name] = await check.check(service)
        return results

    def has_errors(self, results: dict[str, CheckResult]) -> bool:
        """Check if any results have error severity.

        Parameters
        ----------
        results:
            Results from run_all().

        Returns
        -------
        bool:
            True if any check has error severity and failed.
        """
        return any(r.severity == "error" and not r.passed for r in results.values())

    def format_errors(self, results: dict[str, CheckResult]) -> str:
        """Format error results as a human-readable message.

        Parameters
        ----------
        results:
            Results from run_all().

        Returns
        -------
        str:
            Formatted error message with remediation steps.
        """
        lines = []
        for name, result in results.items():
            if result.severity == "error" and not result.passed:
                lines.append(f"[{name}] {result.message}")
                if result.remediation:
                    lines.append(f"  Remediation: {result.remediation}")
        return "\n".join(lines)


async def run_preflight(service: Service) -> tuple[dict[str, CheckResult], bool]:
    """Convenience function to run all default preflight checks.

    Parameters
    ----------
    service:
        The service to validate.

    Returns
    -------
    tuple[dict[str, CheckResult], bool]:
        Tuple of (results dict, has_errors flag).
    """
    runner = PreflightRunner.default()
    results = await runner.run_all(service)
    return results, runner.has_errors(results)
