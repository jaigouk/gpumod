"""Tokenizer validation check for vLLM services.

Validates that a model's tokenizer has required attributes before
starting a vLLM service. This prevents runtime failures from
tokenizer issues like the devstral-small-2505 bug.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpumod.models import DriverType
from gpumod.preflight.base import CheckResult

if TYPE_CHECKING:
    from gpumod.models import Service

# Required tokenizer attributes for vLLM compatibility
REQUIRED_ATTRS = ("all_special_ids", "eos_token_id")


class TokenizerCheck:
    """Check that a model's tokenizer has required attributes.

    This check validates that HuggingFace tokenizers have the
    attributes required by vLLM, particularly `all_special_ids`
    which is missing in some models (e.g., devstral-small-2505).
    """

    @property
    def name(self) -> str:
        """Return the name of this check."""
        return "tokenizer"

    async def check(self, service: Service) -> CheckResult:
        """Validate the tokenizer for a service.

        Parameters
        ----------
        service:
            The service to validate.

        Returns
        -------
        CheckResult:
            Pass if tokenizer has all required attributes,
            fail with details if any are missing.
        """
        # Skip for non-vLLM drivers (llama.cpp uses GGUF, not HF tokenizers)
        if service.driver != DriverType.VLLM:
            return CheckResult(
                passed=True,
                severity="info",
                message=f"Skipped: {service.driver.value} driver doesn't use HF tokenizer",
            )

        # Skip if no model_id configured
        if not service.model_id:
            return CheckResult(
                passed=True,
                severity="info",
                message="Skipped: no model_id configured",
            )

        # Attempt to load and validate tokenizer
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]

            tokenizer = AutoTokenizer.from_pretrained(service.model_id)
        except Exception as e:
            return CheckResult(
                passed=False,
                severity="error",
                message=f"Failed to load tokenizer: {e}",
                remediation="Check model_id is correct and model is accessible",
            )

        # Check for required attributes
        missing_attrs = [attr for attr in REQUIRED_ATTRS if not hasattr(tokenizer, attr)]

        if missing_attrs:
            return CheckResult(
                passed=False,
                severity="error",
                message=f"Tokenizer missing required attributes: {', '.join(missing_attrs)}",
                remediation="Use a different model or update transformers library",
            )

        return CheckResult(
            passed=True,
            severity="info",
            message="Tokenizer loaded successfully with all required attributes",
        )
