"""Base types for preflight validation system.

Provides the Protocol for preflight checks and the CheckResult dataclass.
Following Interface Segregation and Dependency Inversion principles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from gpumod.models import Service


@dataclass
class CheckResult:
    """Result of a preflight check.

    Attributes
    ----------
    passed:
        Whether the check passed.
    severity:
        Severity level: "info", "warning", or "error".
    message:
        Human-readable description of the result.
    remediation:
        Optional remediation steps if the check failed.
    """

    passed: bool
    severity: Literal["info", "warning", "error"]
    message: str
    remediation: str | None = None


class PreflightCheck(Protocol):
    """Protocol for preflight checks.

    Each check validates a specific aspect of service configuration
    before the service is started. Follows Open/Closed principle -
    new checks can be added without modifying existing code.
    """

    @property
    def name(self) -> str:
        """Return the name of this check."""
        ...

    async def check(self, service: Service) -> CheckResult:
        """Run the check on the given service.

        Parameters
        ----------
        service:
            The service to validate.

        Returns
        -------
        CheckResult:
            The result of the validation.
        """
        ...
