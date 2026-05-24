"""Quiesce gate for heavy (GPU-bound) service starts."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gpumod.db import Database
    from gpumod.models import Service


@dataclass(frozen=True)
class QuiesceVerdict:
    """Result of a quiesce check."""

    allowed: bool
    remaining_seconds: float
    remediation: str


QUIESCE_LAST_HEAVY_STOP_KEY = "quiesce_last_heavy_stop"


def is_heavy(service: Service) -> bool:
    """Return True if service is GPU-heavy (vram_mb > 0)."""
    return service.vram_mb > 0


async def record_heavy_stop(db: Database) -> None:
    """Record current time as last heavy-service stop.

    Stores epoch float in settings table.
    """
    await db.set_setting(
        QUIESCE_LAST_HEAVY_STOP_KEY,
        str(time.time()),
        description="Epoch when last heavy service stopped (quiesce gate)",
    )


async def check_quiesce(db: Database, quiesce_seconds: float) -> QuiesceVerdict:
    """Check if quiesce period has elapsed since last heavy stop.

    Clock-skew safe: if elapsed < 0, treat as no prior stop.
    Corrupted value: treat as no prior stop (don't block on bad data).
    """
    raw = await db.get_setting(QUIESCE_LAST_HEAVY_STOP_KEY)
    if raw is None:
        return QuiesceVerdict(allowed=True, remaining_seconds=0, remediation="")

    try:
        last_stop = float(raw)
    except (ValueError, TypeError):
        return QuiesceVerdict(allowed=True, remaining_seconds=0, remediation="")

    elapsed = time.time() - last_stop
    if elapsed < 0:
        logger.warning("Clock skew: last heavy stop appears %.0fs in the future", -elapsed)
        return QuiesceVerdict(allowed=True, remaining_seconds=0, remediation="")

    remaining = quiesce_seconds - elapsed
    if remaining <= 0:
        return QuiesceVerdict(allowed=True, remaining_seconds=0, remediation="")

    return QuiesceVerdict(
        allowed=False,
        remaining_seconds=remaining,
        remediation=(
            f"Quiesce period active: {remaining:.0f}s remaining "
            f"(configured: {quiesce_seconds:.0f}s). "
            f"A heavy GPU service was stopped {elapsed:.0f}s ago. "
            f"Wait for the GPU driver to fully reclaim memory, or use "
            f"--no-quiesce to bypass."
        ),
    )
