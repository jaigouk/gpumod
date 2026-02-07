"""Centralized configuration for gpumod.

All settings are configurable via environment variables with the ``GPUMOD_``
prefix.  For example, ``GPUMOD_DB_PATH`` overrides the default database path.

Environment Variables
---------------------
GPUMOD_DB_PATH : str
    Path to the SQLite database file.
    Default: ``~/.config/gpumod/gpumod.db``
GPUMOD_PRESETS_DIR : str
    Path to the built-in presets directory.
    Default: auto-resolved via ``importlib.resources`` or relative path fallback.
GPUMOD_LOG_LEVEL : str
    Logging level. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    Default: ``INFO``
GPUMOD_LLM_BACKEND : str
    LLM provider backend name (e.g. openai, anthropic, ollama).
    Default: ``openai``
GPUMOD_LLM_API_KEY : str
    API key for the LLM backend. Stored as SecretStr — never appears in
    logs or repr output.
    Default: None
GPUMOD_LLM_MODEL : str
    LLM model identifier.
    Default: ``gpt-4o-mini``
GPUMOD_LLM_BASE_URL : str
    Custom base URL for the LLM API (e.g. for Ollama or proxy).
    Default: None
GPUMOD_MCP_RATE_LIMIT : int
    Maximum MCP requests per minute. Must be >= 1.
    Default: 10
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Presets directory resolution
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _resolve_default_presets_dir() -> Path:
    """Resolve the default presets directory.

    Attempts to locate the presets directory using ``importlib.resources``
    (works for installed packages), falling back to a relative path from
    this file (works in development src-layout).

    Returns
    -------
    Path
        The resolved presets directory path.
    """
    # Try importlib.resources first (works when installed as a package)
    try:
        ref = importlib.resources.files("gpumod") / ".." / ".." / "presets"
        resolved = Path(str(ref)).resolve()
        if resolved.is_dir():
            return resolved
    except (TypeError, ModuleNotFoundError):
        pass

    # Fallback: relative path from this file (src-layout development)
    fallback = Path(__file__).parent.parent.parent / "presets"
    return fallback.resolve()


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class GpumodSettings(BaseSettings):
    """Centralized settings for the gpumod application.

    All fields can be overridden via environment variables prefixed with
    ``GPUMOD_``.  See module docstring for the full list.
    """

    model_config = SettingsConfigDict(
        env_prefix="GPUMOD_",
    )

    # Database
    db_path: Path = Field(default_factory=lambda: Path.home() / ".config" / "gpumod" / "gpumod.db")

    # Presets
    presets_dir: Path = Field(default_factory=_resolve_default_presets_dir)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # LLM settings
    llm_backend: str = "openai"
    llm_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None

    # MCP settings
    mcp_rate_limit: int = Field(default=10, ge=1)

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        """Normalize log level to uppercase and validate."""
        return v.upper() if isinstance(v, str) else v

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def _validate_llm_base_url(cls, v: str | None) -> str | None:
        """Validate llm_base_url: http/https only, block metadata IPs (SEC-V3)."""
        if v is None:
            return v
        from urllib.parse import urlparse

        parsed = urlparse(v)
        # Only allow http and https schemes
        if parsed.scheme not in ("http", "https"):
            msg = f"llm_base_url must use http or https scheme, got {parsed.scheme!r}"
            raise ValueError(msg)
        # Block cloud metadata IP ranges (SSRF)
        hostname = parsed.hostname or ""
        _blocked_hosts = frozenset(
            {
                "169.254.169.254",  # AWS/GCP metadata
                "metadata.google.internal",  # GCP metadata
                "100.100.100.200",  # Alibaba metadata
            }
        )
        if hostname in _blocked_hosts:
            msg = "llm_base_url must not point to cloud metadata endpoints"
            raise ValueError(msg)
        # Block link-local range 169.254.0.0/16
        if hostname.startswith("169.254."):
            msg = "llm_base_url must not point to link-local addresses"
            raise ValueError(msg)
        return v

    @field_validator("db_path", mode="after")
    @classmethod
    def _validate_db_path(cls, v: Path) -> Path:
        """Validate db_path resolves under $HOME or /tmp (SEC-V4)."""
        resolved = v.resolve()
        home = Path.home().resolve()
        tmp = Path("/tmp").resolve()
        if not (
            str(resolved).startswith(str(home) + "/") or str(resolved).startswith(str(tmp) + "/")
        ):
            msg = f"db_path must resolve under $HOME or /tmp, got {resolved}"
            raise ValueError(msg)
        return v


# ---------------------------------------------------------------------------
# Singleton / cached accessor
# ---------------------------------------------------------------------------

_settings_instance: GpumodSettings | None = None


def get_settings() -> GpumodSettings:
    """Return the cached GpumodSettings singleton.

    Creates the instance on first call, then returns the same object
    on subsequent calls.  Use :func:`_clear_settings_cache` in tests
    to reset.

    Returns
    -------
    GpumodSettings
        The application settings instance.
    """
    global _settings_instance  # noqa: PLW0603
    if _settings_instance is None:
        _settings_instance = GpumodSettings()
    return _settings_instance


def _clear_settings_cache() -> None:
    """Clear the settings singleton cache.

    Intended for test teardown so each test can start with fresh settings.
    """
    global _settings_instance  # noqa: PLW0603
    _settings_instance = None
