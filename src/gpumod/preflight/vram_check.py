"""VRAM preflight check for gpumod services (gpumod-89z, gpumod-ja0m).

Validates that a service's VRAM requirements fit within available GPU
memory BEFORE attempting to start the service.

Key Features:
- Compares configured vram_mb against free GPU memory
- Includes configurable safety margin (default 512 MB)
- Generates actionable suggestions when VRAM doesn't fit
- llama.cpp-specific: suggests reduced n_gpu_layers or ctx_size
- gpumod-ja0m: profile-aware KV cache savings via compound formula
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from gpumod.preflight.base import CheckResult

if TYPE_CHECKING:
    from gpumod.models import Service
    from gpumod.services.vram import VRAMTracker

logger = logging.getLogger(__name__)

# Default safety margin in MB
DEFAULT_SAFETY_MARGIN_MB = 512
# gpumod-lgt: MTP (Multi-Token Prediction) variants load ~1.5 GB more than
# the declared vram_mb because of the draft head + draft KV cache. The
# preset's vram_mb is usually estimated from the non-MTP twin's footprint;
# without this overhead, MTP services can OOM mid-load on tight 24 GB GPUs.
DEFAULT_MTP_OVERHEAD_MB = 1500
# Token in extra_args that marks a service as MTP-using.
MTP_FLAG_MARKER = "--spec-type draft-mtp"


# ---------------------------------------------------------------------------
# gpumod-ja0m: KV cache profile protocol and compound formula
# ---------------------------------------------------------------------------


class KVCacheProfileLike(Protocol):
    """Structural interface for KVCacheProfile (Dependency Inversion).

    Consumers depend on this protocol, not on the concrete Pydantic model
    in ``models.py``. Any object with matching attributes satisfies this.
    """

    num_sliding_layers: int
    num_shared_sliding_layers: int
    num_global_layers: int
    num_shared_global_layers: int
    sliding_window: int | None
    triattn_budget: int | None
    per_tok_bytes_local: int
    per_tok_bytes_global: int
    base_overhead_mb: float


def estimate_kv_mb(profile: KVCacheProfileLike, ctx: int) -> float:
    """Compute KV cache size in MB using the compound formula.

    The compound formula accounts for sliding-window capping, KV sharing,
    and TriAttention budgets — unlike the linear ``kv_cache_per_1k_tokens_mb``
    heuristic which assumes all layers scale identically with context.

    Parameters:
        profile: KV cache profile with per-layer-type parameters.
        ctx: Context size in tokens.

    Returns:
        Estimated KV cache size in MB.
    """
    unique_sliding = profile.num_sliding_layers - profile.num_shared_sliding_layers
    unique_global = profile.num_global_layers - profile.num_shared_global_layers

    sliding_tok = min(ctx, profile.sliding_window) if profile.sliding_window else ctx
    global_tok = min(ctx, profile.triattn_budget) if profile.triattn_budget else ctx

    bytes_total = (
        unique_sliding * sliding_tok * profile.per_tok_bytes_local
        + unique_global * global_tok * profile.per_tok_bytes_global
    )
    return bytes_total / (1024 * 1024) + profile.base_overhead_mb


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
        mtp_overhead_mb: int = DEFAULT_MTP_OVERHEAD_MB,
    ) -> None:
        """Initialize VRAMCheck.

        Parameters:
            vram_tracker: VRAMTracker instance (creates one if not provided).
            safety_margin_mb: Extra VRAM buffer required (default 512 MB).
            mtp_overhead_mb: Additional VRAM required for MTP draft head +
                draft KV cache when ``--spec-type draft-mtp`` is in the
                service's ``extra_args`` (default 1500 MB). Pass 0 to
                disable MTP-aware accounting.
        """
        self._vram_tracker = vram_tracker
        self._safety_margin_mb = safety_margin_mb
        self._mtp_overhead_mb = mtp_overhead_mb
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
        # gpumod-lgt: detect MTP and add its draft-context overhead. The
        # preset's vram_mb usually reflects the non-MTP twin's footprint;
        # MTP needs ~1.5 GB more for the draft head + draft KV cache.
        mtp_overhead = self._compute_mtp_overhead(service)
        required_with_margin = required_mb + mtp_overhead + self._safety_margin_mb

        # Check if it fits
        if required_with_margin <= free_mb:
            msg_parts = [f"VRAM OK: {required_mb} MB required"]
            if mtp_overhead > 0:
                msg_parts.append(f"+{mtp_overhead} MB MTP overhead")
            msg_parts.append(f"+{self._safety_margin_mb} MB margin fits in {free_mb} MB free")
            return CheckResult(
                passed=True,
                severity="info",
                message=" ".join(msg_parts),
            )

        # Doesn't fit - generate suggestions
        suggestions = self._generate_suggestions(service, required_mb, free_mb, total_mb)
        self._last_suggestions = suggestions

        remediation_lines = [
            f"Required: {required_with_margin} MB "
            f"({required_mb} declared"
            + (f" + {mtp_overhead} MTP overhead" if mtp_overhead > 0 else "")
            + f" + {self._safety_margin_mb} margin)",
            f"Available: {free_mb} MB (of {total_mb} MB total)",
            "",
            "Suggestions:",
        ]
        for i, suggestion in enumerate(suggestions, 1):
            remediation_lines.append(f"  {i}. {suggestion.message}")
        remediation_lines.append(
            "  See ~/k3s-setup/docs/benchmark-host/gpu-stability.md "
            "for the broader pinned-memory freeze class."
        )

        msg = f"VRAM insufficient: {required_with_margin} MB required ({required_mb} declared"
        if mtp_overhead > 0:
            msg += f" + {mtp_overhead} MB MTP overhead"
        msg += f") exceeds {free_mb} MB available"

        return CheckResult(
            passed=False,
            severity="error",
            message=msg,
            remediation="\n".join(remediation_lines),
        )

    def _compute_mtp_overhead(self, service: Service) -> int:
        """Return MTP overhead in MB if the service uses MTP, else 0.

        Detection looks for ``--spec-type draft-mtp`` in
        ``extra_config['unit_vars']['extra_args']``. Configurable via the
        ``mtp_overhead_mb`` constructor arg; pass 0 to disable entirely.
        """
        if self._mtp_overhead_mb <= 0:
            return 0
        try:
            unit_vars = service.extra_config.get("unit_vars", {})
        except AttributeError:
            return 0
        if not isinstance(unit_vars, dict):
            return 0
        extra_args = unit_vars.get("extra_args", "")
        if not isinstance(extra_args, str):
            return 0
        if MTP_FLAG_MARKER in extra_args:
            return self._mtp_overhead_mb
        return 0

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

        Branches on ``kv_cache_profile`` (gpumod-ja0m):
        - If profile is ``None``: existing scalar heuristic (``for_llamacpp``).
        - If profile present: compound formula for accurate KV savings.
        - When savings insufficient (bounded case): alternative remediation.

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
        kv_cache_profile: Any = getattr(service, "kv_cache_profile", None)

        if isinstance(unit_vars, dict):
            n_gpu_layers = unit_vars.get("n_gpu_layers")
            ctx_size = unit_vars.get("ctx_size")

            if kv_cache_profile is not None:
                # --- Profile-aware path (hybrid models, gpumod-ja0m) ---
                overage = required_mb - available_mb

                # Strategy 1: Layer reduction (same heuristic, always valid)
                if n_gpu_layers is not None and int(n_gpu_layers) > 0 and overage > 0:
                    total_layers = 80  # Default assumption
                    mb_per_layer = required_mb / total_layers
                    layers_to_remove = int(overage / mb_per_layer) + 1
                    suggested_layers = max(0, int(n_gpu_layers) - layers_to_remove)
                    if suggested_layers > 0:
                        estimated = required_mb - (layers_to_remove * mb_per_layer)
                        suggestions.append(
                            VRAMSuggestion(
                                message=(
                                    f"Reduce n_gpu_layers from {n_gpu_layers} "
                                    f"to {suggested_layers} to save "
                                    f"~{layers_to_remove * mb_per_layer:.0f} MB"
                                ),
                                suggested_layers=suggested_layers,
                                estimated_vram_mb=int(estimated),
                            )
                        )

                # Strategy 2: Context reduction via compound formula
                ctx = int(ctx_size or 8192)
                if ctx > 4096 and overage > 0:
                    self._add_profile_ctx_suggestion(
                        suggestions, kv_cache_profile, ctx, required_mb, overage
                    )

            elif n_gpu_layers is not None:
                # --- Existing scalar path (dense models) — UNCHANGED ---
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

    def _add_profile_ctx_suggestion(
        self,
        suggestions: list[VRAMSuggestion],
        profile: KVCacheProfileLike,
        ctx: int,
        required_mb: int,
        overage: int,
    ) -> None:
        """Add a profile-aware context-reduction suggestion (gpumod-ja0m).

        Uses the compound formula to compute actual KV cache savings from
        halving context. When savings are insufficient because sliding-window
        or TriAttention bounds cap the KV cache, surfaces an alternative
        remediation message instead.

        Parameters:
            suggestions: List to append suggestion to (mutated in place).
            profile: KV cache profile with compound formula fields.
            ctx: Current context size in tokens.
            required_mb: Required VRAM in MB.
            overage: How many MB over available VRAM.
        """
        current_kv = estimate_kv_mb(profile, ctx)
        halved_kv = estimate_kv_mb(profile, ctx // 2)
        savings_mb = current_kv - halved_kv

        if savings_mb >= overage:
            # Sufficient: suggest ctx reduction with accurate savings
            suggestions.append(
                VRAMSuggestion(
                    message=(
                        f"Reduce ctx_size from {ctx} to {ctx // 2} "
                        f"to save ~{savings_mb:.0f} MB of KV cache"
                    ),
                    suggested_ctx_size=ctx // 2,
                    estimated_vram_mb=int(required_mb - savings_mb),
                )
            )
        else:
            # Bounded: ctx reduction insufficient due to architecture limits
            sw = profile.sliding_window
            if sw is not None:
                suggestions.append(
                    VRAMSuggestion(
                        message=(
                            f"This model's KV cache is bounded by "
                            f"sliding_window={sw}; reducing ctx below "
                            f"{sw} saves nothing more — consider a "
                            f"smaller quantization instead."
                        ),
                    )
                )
            else:
                suggestions.append(
                    VRAMSuggestion(
                        message=(
                            "Context reduction insufficient — consider "
                            "a smaller quantization instead."
                        ),
                    )
                )
