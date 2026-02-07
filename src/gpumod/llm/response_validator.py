"""Response validation for LLM planning output.

SEC-L1: All service IDs in LLM responses are validated via
:func:`gpumod.validation.validate_service_id` before use.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gpumod.llm.base import LLMResponseError
from gpumod.validation import validate_service_id


class ServiceAllocation(BaseModel):
    """A single service VRAM allocation from the LLM.

    Attributes
    ----------
    service_id:
        The service identifier (validated against SEC-V1 regex).
    vram_mb:
        The VRAM allocation in megabytes (must be positive).
    """

    model_config = ConfigDict(extra="forbid")

    service_id: str
    vram_mb: int

    @field_validator("vram_mb")
    @classmethod
    def _vram_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            msg = f"VRAM must be a positive integer, got {v}"
            raise ValueError(msg)
        return v


class PlanSuggestion(BaseModel):
    """Schema for an LLM planning response.

    Attributes
    ----------
    services:
        List of service VRAM allocations.
    reasoning:
        The LLM's explanation for the allocation strategy.
    """

    model_config = ConfigDict(extra="forbid")

    services: list[ServiceAllocation]
    reasoning: str = Field(max_length=10_000)

    @field_validator("reasoning")
    @classmethod
    def _sanitize_reasoning(cls, v: str) -> str:
        """Strip terminal escape sequences from reasoning text."""
        # Strip ANSI escape sequences
        v = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", v)
        # Remove control characters (except newline/tab)
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)
        return v


def validate_plan_response(raw: dict[str, Any]) -> PlanSuggestion:
    """Validate and parse an LLM planning response.

    SEC-L1: All ``service_id`` values are validated via
    :func:`gpumod.validation.validate_service_id` to prevent injection
    of invalid or malicious identifiers.

    Parameters
    ----------
    raw:
        The raw dict parsed from the LLM JSON output.

    Returns
    -------
    PlanSuggestion
        The validated plan suggestion.

    Raises
    ------
    LLMResponseError
        If the response does not match the expected schema, contains
        invalid service IDs, or has invalid VRAM values.
    """
    try:
        plan = PlanSuggestion.model_validate(raw)
    except Exception as exc:
        msg = f"Invalid LLM response structure: {exc}"
        raise LLMResponseError(msg) from exc

    # SEC-L1: Validate every service_id against the allowed pattern
    for alloc in plan.services:
        try:
            validate_service_id(alloc.service_id)
        except ValueError as exc:
            msg = f"LLM returned invalid service_id {alloc.service_id!r}: {exc}"
            raise LLMResponseError(msg) from exc

    return plan
