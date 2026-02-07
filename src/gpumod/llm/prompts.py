"""Prompt templates for LLM-based GPU resource planning.

SEC-L2: Prompt templates maintain clear instruction/data boundaries.
SEC-L5: Only service IDs, names, VRAM numbers, and GPU capacity are sent.
"""

from __future__ import annotations

import json
from typing import Any

from gpumod.validation import sanitize_name

PLANNING_SYSTEM_PROMPT = """You are a GPU resource planner for ML workloads.
You analyze GPU state and suggest optimal service allocation plans.

RULES:
- Only use service IDs from the PROVIDED list below
- VRAM allocations must be positive integers in MB
- Total VRAM must not exceed GPU capacity
- Respond ONLY with valid JSON matching the required schema
- Do NOT invent service IDs that are not in the provided data
- Do NOT include any text outside the JSON response

REQUIRED RESPONSE SCHEMA:
{
  "services": [
    {"service_id": "<id from provided list>", "vram_mb": <positive integer>}
  ],
  "reasoning": "<brief explanation of the allocation strategy>"
}
"""


def build_planning_prompt(
    services: list[dict[str, Any]],
    gpu_total_mb: int,
    current_mode: str | None = None,
    budget_mb: int | None = None,
) -> str:
    """Build the user prompt with GPU state data.

    SEC-L5: Only sends service IDs, names, VRAM, and GPU capacity.
    SEC-L2: Data placed in clearly delimited section, not in instructions.

    Parameters
    ----------
    services:
        List of service dicts with 'id', 'name', and 'vram_mb' fields.
    gpu_total_mb:
        Total GPU VRAM capacity in MB.
    current_mode:
        Currently active mode name, if any.
    budget_mb:
        Optional VRAM budget constraint in MB.

    Returns
    -------
    str
        The formatted user prompt string.
    """
    parts: list[str] = []

    parts.append("Plan optimal GPU VRAM allocation for the following services.")
    parts.append("")
    parts.append(f"GPU total VRAM: {gpu_total_mb} MB")

    if current_mode is not None:
        parts.append(f"Current mode: {sanitize_name(current_mode)}")

    if budget_mb is not None:
        parts.append(f"VRAM budget: {budget_mb} MB")

    parts.append("")

    # SEC-L2: Clear data boundary
    parts.append("--- BEGIN SERVICE DATA ---")

    # SEC-L5: Only send minimal fields (id, name, vram_mb)
    minimal_services = [
        {"id": s["id"], "name": sanitize_name(s["name"]), "vram_mb": s["vram_mb"]}
        for s in services
    ]
    parts.append(json.dumps(minimal_services, indent=2, ensure_ascii=True))

    parts.append("--- END SERVICE DATA ---")

    return "\n".join(parts)
