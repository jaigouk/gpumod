"""Running-services context formatter (gpumod-lgt).

When preflight refuses to start a service, callers should append the
output of ``format_running_services`` to the error message so the
operator knows which OTHER gpumod services are currently holding
GPU/RAM and what to stop. The block lists services sorted by VRAM
budget descending so the heaviest stop-candidate appears first.

Designed to be a "best-effort" enricher — if the registry call raises,
we return ``""`` rather than swallowing the original preflight failure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpumod.services.registry import ServiceRegistry

logger = logging.getLogger(__name__)


async def format_running_services(
    registry: ServiceRegistry,
    *,
    exclude_id: str | None,
) -> str:
    """Return a formatted block listing currently running gpumod services.

    Parameters
    ----------
    registry:
        A ``ServiceRegistry`` (or anything with an async ``list_running()``).
    exclude_id:
        Service ID to omit from the output — typically the service
        currently being attempted, since suggesting its own stop is
        useless.

    Returns
    -------
    str:
        A multi-line block listing each running service with its
        ``vram_mb`` and a ``gpumod service stop <id>`` hint. Empty
        string when no other services are running, or when the registry
        call fails (best-effort).
    """
    try:
        services = await registry.list_running()
    except Exception as exc:
        # Best-effort enricher — registry failure must NOT mask the
        # underlying preflight error message we're trying to enrich.
        logger.warning("Could not list running services for context: %s", exc)
        return ""

    others = [s for s in services if exclude_id is None or s.id != exclude_id]
    if not others:
        return ""

    others_sorted = sorted(
        others,
        key=lambda s: getattr(s, "vram_mb", 0) or 0,
        reverse=True,
    )

    lines = ["Currently running gpumod services:"]
    for svc in others_sorted:
        vram = getattr(svc, "vram_mb", 0) or 0
        lines.append(f"  - {svc.id}: {vram} MB VRAM declared")
    lines.append("")
    lines.append("Recommended (free the most VRAM first):")
    lines.extend(f"  gpumod service stop {svc.id}" for svc in others_sorted)
    return "\n".join(lines)
