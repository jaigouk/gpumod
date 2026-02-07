"""Shared input validation for gpumod (SEC-V1).

Provides regex-based validators for service IDs, mode IDs, model IDs,
and context overrides. Also provides name sanitization extracted from
the visualization module.
"""

from __future__ import annotations

import re

SERVICE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$")
MODE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$")
MODEL_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-./]{0,127}$")
MAX_CONTEXT_TOKENS = 131072


def validate_service_id(value: str) -> str:
    """Validate a service ID against the allowed regex.

    Parameters
    ----------
    value:
        The service ID string to validate.

    Returns
    -------
    str
        The validated service ID.

    Raises
    ------
    ValueError
        If the service ID does not match the expected pattern.
    """
    if not SERVICE_ID_RE.match(value):
        msg = f"Invalid service_id: {value!r}"
        raise ValueError(msg)
    return value


def validate_mode_id(value: str) -> str:
    """Validate a mode ID against the allowed regex.

    Parameters
    ----------
    value:
        The mode ID string to validate.

    Returns
    -------
    str
        The validated mode ID.

    Raises
    ------
    ValueError
        If the mode ID does not match the expected pattern.
    """
    if not MODE_ID_RE.match(value):
        msg = f"Invalid mode_id: {value!r}"
        raise ValueError(msg)
    return value


def validate_model_id(value: str) -> str:
    """Validate a model ID against the allowed regex.

    Parameters
    ----------
    value:
        The model ID string to validate.

    Returns
    -------
    str
        The validated model ID.

    Raises
    ------
    ValueError
        If the model ID does not match the expected pattern.
    """
    if not MODEL_ID_RE.match(value):
        msg = f"Invalid model_id: {value!r}"
        raise ValueError(msg)
    return value


def validate_context_override(key: str, value: int) -> tuple[str, int]:
    """Validate a context override key-value pair.

    The key is validated as a service ID.  The value must be between
    1 and MAX_CONTEXT_TOKENS (inclusive).

    Parameters
    ----------
    key:
        The service ID this override applies to.
    value:
        The context window size in tokens.

    Returns
    -------
    tuple[str, int]
        The validated (key, value) pair.

    Raises
    ------
    ValueError
        If the key is not a valid service ID or the value is out of range.
    """
    validate_service_id(key)
    if value < 1 or value > MAX_CONTEXT_TOKENS:
        msg = f"Invalid context override for {key!r}: {value} (must be 1..{MAX_CONTEXT_TOKENS})"
        raise ValueError(msg)
    return key, value


# ---------------------------------------------------------------------------
# DB-level validation (SEC-D6, SEC-V5)
# ---------------------------------------------------------------------------

# Allowed keys for service extra_config (SEC-D6)
EXTRA_CONFIG_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "unit_vars",
        "context_size",
        "quantization",
        "tensor_parallel",
        "gpu_memory_utilization",
        "max_model_len",
        "dtype",
        "rope_scaling",
        "chat_template",
        "trust_remote_code",
    }
)

# Maximum VRAM in MB (1 TB = 1,048,576 MB) — no single GPU exceeds this
MAX_VRAM_MB = 1_048_576


def validate_extra_config(config: dict[str, object]) -> dict[str, object]:
    """Validate extra_config keys against the allowed set (SEC-D6).

    Parameters
    ----------
    config:
        The extra configuration dict to validate.

    Returns
    -------
    dict[str, object]
        The validated config (unchanged).

    Raises
    ------
    ValueError
        If any key is not in the allowed set.
    """
    unknown = set(config.keys()) - EXTRA_CONFIG_ALLOWED_KEYS
    if unknown:
        msg = f"Unknown extra_config keys: {sorted(unknown)}"
        raise ValueError(msg)
    return config


def validate_vram_mb(value: int, *, max_mb: int = MAX_VRAM_MB) -> int:
    """Validate VRAM value is within reasonable bounds (SEC-V5).

    Parameters
    ----------
    value:
        The VRAM value in MB.
    max_mb:
        Maximum allowed VRAM in MB. Default 1TB.

    Returns
    -------
    int
        The validated VRAM value.

    Raises
    ------
    ValueError
        If the value is negative, zero, or exceeds max_mb.
    """
    if value < 0:
        msg = f"VRAM must be non-negative, got {value}"
        raise ValueError(msg)
    if value > max_mb:
        msg = f"VRAM {value} MB exceeds maximum of {max_mb} MB"
        raise ValueError(msg)
    return value


def sanitize_name(name: str) -> str:
    """Sanitize a service name to prevent terminal escape injection.

    Strips Rich markup tags and ANSI control characters from the name.

    Parameters
    ----------
    name:
        The raw service name to sanitize.

    Returns
    -------
    str
        A cleaned service name safe for terminal display.
    """
    # Strip ANSI escape sequences
    cleaned = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", name)
    # Strip Rich markup tags like [bold red]...[/bold red]
    cleaned = re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", cleaned)
    # Remove any remaining control characters (except newline/tab)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    return cleaned
