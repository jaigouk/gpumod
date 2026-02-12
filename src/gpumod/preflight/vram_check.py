"""VRAM preflight check for gpumod services (gpumod-89z).

Validates that a service's VRAM requirements fit within available GPU
memory BEFORE attempting to start the service.

Key Features:
- Compares configured vram_mb against free GPU memory
- Includes configurable safety margin (default 512 MB)
- Generates actionable suggestions when VRAM doesn't fit
- llama.cpp-specific: suggests reduced n_gpu_layers or ctx_size
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gpumod.preflight.base import CheckResult

if TYPE_CHECKING:
    from gpumod.models import Service
    from gpumod.services.vram import VRAMTracker

logger = logging.getLogger(__name__)

# Default safety margin in MB
DEFAULT_SAFETY_MARGIN_MB = 512


# ---------------------------------------------------------------------------
# VRAMSuggestion
# ---------------------------------------------------------------------------


@dataclass
class VRAMSuggestion:
    """Suggestion for reducing VRAM usage.

    Attributes:
        message: Human-readable suggestion text.
        suggested_layers: Suggested n_gpu_layers value (llama.cpp).
        suggested_ctx_size: Suggested ctx_size value.
        estimated_vram_mb: Estimated VRAM after applying suggestion.
    """

    message: str
    suggested_layers: int | None = None
    suggested_ctx_size: int | None = None
    estimated_vram_mb: int | None = None

    @classmethod
    def for_llamacpp(
        cls,
        required_mb: int,
        available_mb: int,
        current_layers: int,
        total_layers: int,
        ctx_size: int,
    ) -> VRAMSuggestion | None:
        """Generate suggestion for llama.cpp service.

        Parameters:
            required_mb: Required VRAM in MB.
            available_mb: Available VRAM in MB.
            current_layers: Current n_gpu_layers setting.
            total_layers: Total layers in the model.
            ctx_size: Current context size.

        Returns:
            VRAMSuggestion or None if no reasonable suggestion.
        """
        if required_mb <= 0 or available_mb <= 0:
            return None

        overage = required_mb - available_mb
        if overage <= 0:
            return None

        # Strategy 1: Reduce n_gpu_layers
        # Rough estimate: each layer uses (required_mb / total_layers) MB
        if current_layers > 0 and total_layers > 0:
            mb_per_layer = required_mb / total_layers
            layers_to_remove = int(overage / mb_per_layer) + 1
            suggested_layers = max(0, current_layers - layers_to_remove)

            if suggested_layers > 0:
                estimated = required_mb - (layers_to_remove * mb_per_layer)
                return cls(
                    message=(
                        f"Reduce n_gpu_layers from {current_layers} to {suggested_layers} "
                        f"to save ~{layers_to_remove * mb_per_layer:.0f} MB"
                    ),
                    suggested_layers=suggested_layers,
                    estimated_vram_mb=int(estimated),
                )

        # Strategy 2: Reduce context size
        # KV cache roughly scales with context size
        if ctx_size > 4096:
            # Halving context roughly halves KV cache
            suggested_ctx = ctx_size // 2
            kv_savings = overage // 2  # Conservative estimate

            return cls(
                message=(
                    f"Reduce ctx_size from {ctx_size} to {suggested_ctx} "
                    f"to save ~{kv_savings} MB of KV cache"
                ),
                suggested_ctx_size=suggested_ctx,
                estimated_vram_mb=required_mb - kv_savings,
            )

        # No reasonable suggestion
        return cls(
            message=(
                f"Model won't fit even with minimal settings. "
                f"Required: {required_mb} MB, Available: {available_mb} MB. "
                f"Consider a smaller quantization."
            ),
            suggested_layers=0,
        )


# ---------------------------------------------------------------------------
# VRAMCheck
# ---------------------------------------------------------------------------


class VRAMCheck:
    """Preflight check that validates VRAM requirements.

    Compares the service's configured vram_mb against available GPU
    memory and provides actionable suggestions when VRAM doesn't fit.

    Usage:
        tracker = VRAMTracker()
        check = VRAMCheck(vram_tracker=tracker)
        result = await check.check(service)
        if not result.passed:
            suggestions = check.get_suggestions()
    """

    def __init__(
        self,
        vram_tracker: VRAMTracker | None = None,
        safety_margin_mb: int = DEFAULT_SAFETY_MARGIN_MB,
    ) -> None:
        """Initialize VRAMCheck.

        Parameters:
            vram_tracker: VRAMTracker instance (creates one if not provided).
            safety_margin_mb: Extra VRAM buffer required (default 512 MB).
        """
        self._vram_tracker = vram_tracker
        self._safety_margin_mb = safety_margin_mb
        self._last_suggestions: list[VRAMSuggestion] | None = None

    @property
    def name(self) -> str:
        """Return check name."""
        return "vram"

    def _get_tracker(self) -> VRAMTracker:
        """Get or create VRAMTracker."""
        if self._vram_tracker is None:
            from gpumod.services.vram import VRAMTracker

            self._vram_tracker = VRAMTracker()
        return self._vram_tracker

    async def check(self, service: Service) -> CheckResult:
        """Check if service VRAM fits in available GPU memory.

        Parameters:
            service: Service to validate.

        Returns:
            CheckResult indicating pass/fail with suggestions.
        """
        self._last_suggestions = None

        # Skip for services with no VRAM requirement
        if service.vram_mb <= 0:
            return CheckResult(
                passed=True,
                severity="info",
                message="VRAM check skipped (no VRAM requirement configured)",
            )

        # Get current VRAM usage
        try:
            tracker = self._get_tracker()
            usage = await tracker.get_usage()
            free_mb = usage.free_mb
            total_mb = usage.total_mb
        except Exception as exc:
            logger.warning("VRAM check failed: %s", exc)
            return CheckResult(
                passed=False,
                severity="warning",
                message=f"Unable to check VRAM: {exc}",
                remediation="Ensure nvidia-smi is available and GPU is accessible.",
            )

        required_mb = service.vram_mb
        required_with_margin = required_mb + self._safety_margin_mb

        # Check if it fits
        if required_with_margin <= free_mb:
            return CheckResult(
                passed=True,
                severity="info",
                message=(
                    f"VRAM OK: {required_mb} MB required "
                    f"(+{self._safety_margin_mb} MB margin) fits in {free_mb} MB free"
                ),
            )

        # Doesn't fit - generate suggestions
        suggestions = self._generate_suggestions(service, required_mb, free_mb, total_mb)
        self._last_suggestions = suggestions

        remediation_lines = [
            f"Required: {required_mb} MB (+{self._safety_margin_mb} MB margin)",
            f"Available: {free_mb} MB (of {total_mb} MB total)",
            "",
            "Suggestions:",
        ]
        for i, suggestion in enumerate(suggestions, 1):
            remediation_lines.append(f"  {i}. {suggestion.message}")

        return CheckResult(
            passed=False,
            severity="error",
            message=(
                f"VRAM insufficient: {required_mb} MB required exceeds {free_mb} MB available"
            ),
            remediation="\n".join(remediation_lines),
        )

    def get_suggestions(self) -> list[VRAMSuggestion] | None:
        """Get suggestions from last failed check.

        Returns:
            List of VRAMSuggestion if last check failed, else None.
        """
        return self._last_suggestions

    def _generate_suggestions(
        self,
        service: Service,
        required_mb: int,
        available_mb: int,
        total_mb: int,
    ) -> list[VRAMSuggestion]:
        """Generate VRAM reduction suggestions.

        Parameters:
            service: Service that doesn't fit.
            required_mb: Required VRAM in MB.
            available_mb: Available VRAM in MB.
            total_mb: Total GPU VRAM in MB.

        Returns:
            List of suggestions.
        """
        suggestions: list[VRAMSuggestion] = []

        # Extract llama.cpp-specific config
        unit_vars = service.extra_config.get("unit_vars", {})
        if isinstance(unit_vars, dict):
            n_gpu_layers = unit_vars.get("n_gpu_layers")
            ctx_size = unit_vars.get("ctx_size")

            if n_gpu_layers is not None:
                # Assume total_layers from common models (80 for 70B+)
                total_layers = 80  # Default assumption
                suggestion = VRAMSuggestion.for_llamacpp(
                    required_mb=required_mb,
                    available_mb=available_mb,
                    current_layers=int(n_gpu_layers),
                    total_layers=total_layers,
                    ctx_size=int(ctx_size or 8192),
                )
                if suggestion:
                    suggestions.append(suggestion)

        # Generic suggestions
        suggestions.append(
            VRAMSuggestion(
                message="Use a smaller quantization (e.g., Q4_K_S instead of Q4_K_M)",
            )
        )

        suggestions.append(
            VRAMSuggestion(
                message=f"Reduce vram_mb in preset (current: {required_mb} MB)",
            )
        )

        return suggestions
