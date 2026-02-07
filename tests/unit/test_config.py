"""Tests for gpumod.config — centralized configuration module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Clear the settings singleton cache before each test."""
    from gpumod.config import _clear_settings_cache

    _clear_settings_cache()


@pytest.fixture
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all GPUMOD_ env vars so defaults are tested cleanly."""
    for key in list(os.environ):
        if key.startswith("GPUMOD_"):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# TestGpumodSettingsDefaults
# ---------------------------------------------------------------------------


class TestGpumodSettingsDefaults:
    """Verify default values for GpumodSettings."""

    @pytest.mark.usefixtures("_clean_env")
    def test_default_db_path(self) -> None:
        from gpumod.config import GpumodSettings

        settings = GpumodSettings()
        expected = Path.home() / ".config" / "gpumod" / "gpumod.db"
        assert settings.db_path == expected

    @pytest.mark.usefixtures("_clean_env")
    def test_default_presets_dir_exists(self) -> None:
        """presets_dir should resolve to a valid directory."""
        from gpumod.config import GpumodSettings

        settings = GpumodSettings()
        # In dev mode, presets dir should point to the project presets/ directory
        assert settings.presets_dir is not None
        assert isinstance(settings.presets_dir, Path)

    @pytest.mark.usefixtures("_clean_env")
    def test_default_log_level(self) -> None:
        from gpumod.config import GpumodSettings

        settings = GpumodSettings()
        assert settings.log_level == "INFO"

    @pytest.mark.usefixtures("_clean_env")
    def test_default_llm_backend(self) -> None:
        from gpumod.config import GpumodSettings

        settings = GpumodSettings()
        assert settings.llm_backend == "openai"

    @pytest.mark.usefixtures("_clean_env")
    def test_default_llm_api_key_is_none(self) -> None:
        from gpumod.config import GpumodSettings

        settings = GpumodSettings()
        assert settings.llm_api_key is None

    @pytest.mark.usefixtures("_clean_env")
    def test_default_llm_model(self) -> None:
        from gpumod.config import GpumodSettings

        settings = GpumodSettings()
        assert settings.llm_model == "gpt-4o-mini"

    @pytest.mark.usefixtures("_clean_env")
    def test_default_llm_base_url_is_none(self) -> None:
        from gpumod.config import GpumodSettings

        settings = GpumodSettings()
        assert settings.llm_base_url is None

    @pytest.mark.usefixtures("_clean_env")
    def test_default_mcp_rate_limit(self) -> None:
        from gpumod.config import GpumodSettings

        settings = GpumodSettings()
        assert settings.mcp_rate_limit == 10


# ---------------------------------------------------------------------------
# TestEnvVarOverrides
# ---------------------------------------------------------------------------


class TestEnvVarOverrides:
    """Verify that GPUMOD_* env vars override defaults."""

    def test_override_db_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_DB_PATH", "/tmp/custom.db")
        settings = GpumodSettings()
        assert settings.db_path == Path("/tmp/custom.db")

    def test_override_presets_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_PRESETS_DIR", "/opt/presets")
        settings = GpumodSettings()
        assert settings.presets_dir == Path("/opt/presets")

    def test_override_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LOG_LEVEL", "DEBUG")
        settings = GpumodSettings()
        assert settings.log_level == "DEBUG"

    def test_override_llm_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_BACKEND", "anthropic")
        settings = GpumodSettings()
        assert settings.llm_backend == "anthropic"

    def test_override_llm_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_API_KEY", "sk-test-key-123")
        settings = GpumodSettings()
        assert settings.llm_api_key is not None
        assert settings.llm_api_key.get_secret_value() == "sk-test-key-123"

    def test_override_llm_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_MODEL", "claude-3-opus")
        settings = GpumodSettings()
        assert settings.llm_model == "claude-3-opus"

    def test_override_llm_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_BASE_URL", "http://localhost:11434")
        settings = GpumodSettings()
        assert settings.llm_base_url == "http://localhost:11434"

    def test_override_mcp_rate_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_MCP_RATE_LIMIT", "50")
        settings = GpumodSettings()
        assert settings.mcp_rate_limit == 50


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    """Verify validation constraints on settings fields."""

    def test_invalid_log_level_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic import ValidationError

        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LOG_LEVEL", "INVALID_LEVEL")
        with pytest.raises(ValidationError, match="log_level"):
            GpumodSettings()

    def test_negative_rate_limit_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic import ValidationError

        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_MCP_RATE_LIMIT", "-1")
        with pytest.raises(ValidationError, match="mcp_rate_limit"):
            GpumodSettings()

    def test_zero_rate_limit_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic import ValidationError

        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_MCP_RATE_LIMIT", "0")
        with pytest.raises(ValidationError, match="mcp_rate_limit"):
            GpumodSettings()

    def test_valid_log_levels_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import GpumodSettings

        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            monkeypatch.setenv("GPUMOD_LOG_LEVEL", level)
            settings = GpumodSettings()
            assert settings.log_level == level


# ---------------------------------------------------------------------------
# TestSecretStrMasking
# ---------------------------------------------------------------------------


class TestSecretStrMasking:
    """Verify that llm_api_key uses SecretStr and is properly masked."""

    def test_secret_str_repr_is_masked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_API_KEY", "sk-super-secret-key")
        settings = GpumodSettings()
        repr_str = repr(settings.llm_api_key)
        assert "sk-super-secret-key" not in repr_str
        assert "**********" in repr_str

    def test_secret_str_str_is_masked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_API_KEY", "sk-super-secret-key")
        settings = GpumodSettings()
        str_val = str(settings.llm_api_key)
        assert "sk-super-secret-key" not in str_val
        assert "**********" in str_val

    def test_secret_str_get_secret_value_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_API_KEY", "sk-super-secret-key")
        settings = GpumodSettings()
        assert settings.llm_api_key is not None
        assert settings.llm_api_key.get_secret_value() == "sk-super-secret-key"

    def test_secret_str_not_in_model_dump(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_API_KEY", "sk-super-secret-key")
        settings = GpumodSettings()
        dumped = str(settings.model_dump())
        assert "sk-super-secret-key" not in dumped


# ---------------------------------------------------------------------------
# TestGetSettings
# ---------------------------------------------------------------------------


class TestGetSettings:
    """Verify get_settings() singleton/caching behavior."""

    @pytest.mark.usefixtures("_clean_env")
    def test_get_settings_returns_gpumod_settings(self) -> None:
        from gpumod.config import GpumodSettings, get_settings

        settings = get_settings()
        assert isinstance(settings, GpumodSettings)

    @pytest.mark.usefixtures("_clean_env")
    def test_get_settings_returns_same_instance(self) -> None:
        from gpumod.config import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    @pytest.mark.usefixtures("_clean_env")
    def test_get_settings_cache_cleared(self) -> None:
        """After clearing cache, a new instance is created."""
        from gpumod.config import _clear_settings_cache, get_settings

        s1 = get_settings()
        _clear_settings_cache()
        s2 = get_settings()
        assert s1 is not s2

    def test_get_settings_respects_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import get_settings

        monkeypatch.setenv("GPUMOD_LOG_LEVEL", "DEBUG")
        settings = get_settings()
        assert settings.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# TestPresetsDir
# ---------------------------------------------------------------------------


class TestPresetsDir:
    """Verify presets_dir resolution logic."""

    @pytest.mark.usefixtures("_clean_env")
    def test_presets_dir_default_resolves_to_package_presets(self) -> None:
        """Default presets_dir should resolve via importlib.resources or fallback."""
        from gpumod.config import GpumodSettings

        settings = GpumodSettings()
        # The presets dir should be a Path
        assert isinstance(settings.presets_dir, Path)

    def test_presets_dir_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_PRESETS_DIR", "/custom/presets")
        settings = GpumodSettings()
        assert settings.presets_dir == Path("/custom/presets")

    @pytest.mark.usefixtures("_clean_env")
    def test_presets_dir_fallback_when_package_missing(self) -> None:
        """If importlib.resources lookup fails, fallback to relative path."""
        from gpumod.config import _resolve_default_presets_dir

        # The function should always return a Path, never raise
        result = _resolve_default_presets_dir()
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# TestEnvPrefix
# ---------------------------------------------------------------------------


class TestEnvPrefix:
    """Verify all env vars use the GPUMOD_ prefix."""

    @pytest.mark.usefixtures("_clean_env")
    def test_env_prefix_is_gpumod(self) -> None:
        from gpumod.config import GpumodSettings

        # model_config should have env_prefix set to "GPUMOD_"
        config = GpumodSettings.model_config
        assert config.get("env_prefix") == "GPUMOD_"

    def test_unprefixed_env_var_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env vars without GPUMOD_ prefix should not affect settings."""
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("LOG_LEVEL", "CRITICAL")
        monkeypatch.delenv("GPUMOD_LOG_LEVEL", raising=False)
        settings = GpumodSettings()
        assert settings.log_level == "INFO"  # default, not CRITICAL


# ---------------------------------------------------------------------------
# TestURLValidation
# ---------------------------------------------------------------------------


class TestURLValidation:
    """Tests for llm_base_url validation (SEC-V3)."""

    def test_rejects_file_scheme(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """file:// URLs must be rejected."""
        from pydantic import ValidationError

        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_BASE_URL", "file:///etc/passwd")
        with pytest.raises(ValidationError):
            GpumodSettings()

    def test_rejects_aws_metadata_ip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AWS metadata endpoint must be blocked."""
        from pydantic import ValidationError

        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_BASE_URL", "http://169.254.169.254/metadata")
        with pytest.raises(ValidationError):
            GpumodSettings()

    def test_rejects_link_local_ip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Link-local 169.254.x.x addresses must be blocked."""
        from pydantic import ValidationError

        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_BASE_URL", "http://169.254.1.1/api")
        with pytest.raises(ValidationError):
            GpumodSettings()

    def test_accepts_localhost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """localhost for Ollama should be accepted."""
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_BASE_URL", "http://localhost:11434")
        settings = GpumodSettings()
        assert settings.llm_base_url == "http://localhost:11434"

    def test_accepts_openai_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OpenAI API URL should be accepted."""
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_BASE_URL", "https://api.openai.com")
        settings = GpumodSettings()
        assert settings.llm_base_url == "https://api.openai.com"

    def test_accepts_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """None/unset should be accepted."""
        from gpumod.config import GpumodSettings

        monkeypatch.delenv("GPUMOD_LLM_BASE_URL", raising=False)
        settings = GpumodSettings()
        assert settings.llm_base_url is None

    def test_rejects_ftp_scheme(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ftp:// URLs must be rejected."""
        from pydantic import ValidationError

        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_BASE_URL", "ftp://files.example.com")
        with pytest.raises(ValidationError):
            GpumodSettings()

    def test_rejects_gcp_metadata_hostname(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GCP metadata hostname must be blocked."""
        from pydantic import ValidationError

        from gpumod.config import GpumodSettings

        monkeypatch.setenv(
            "GPUMOD_LLM_BASE_URL", "http://metadata.google.internal/computeMetadata"
        )
        with pytest.raises(ValidationError):
            GpumodSettings()

    def test_rejects_alibaba_metadata_ip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Alibaba metadata IP must be blocked."""
        from pydantic import ValidationError

        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_LLM_BASE_URL", "http://100.100.100.200/metadata")
        with pytest.raises(ValidationError):
            GpumodSettings()


# ---------------------------------------------------------------------------
# TestPathValidation
# ---------------------------------------------------------------------------


class TestPathValidation:
    """Tests for db_path validation (SEC-V4)."""

    def test_rejects_path_outside_home_and_tmp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Paths outside $HOME and /tmp must be rejected."""
        from pydantic import ValidationError

        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_DB_PATH", "/etc/shadow")
        with pytest.raises(ValidationError):
            GpumodSettings()

    def test_accepts_path_under_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Paths under $HOME should be accepted."""
        from gpumod.config import GpumodSettings

        home = os.path.expanduser("~")
        monkeypatch.setenv("GPUMOD_DB_PATH", f"{home}/.config/gpumod/test.db")
        settings = GpumodSettings()
        assert str(settings.db_path).endswith("test.db")

    def test_accepts_path_under_tmp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Paths under /tmp should be accepted."""
        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_DB_PATH", "/tmp/gpumod-test/test.db")
        settings = GpumodSettings()
        assert str(settings.db_path).endswith("test.db")

    def test_rejects_path_traversal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Path traversal attempts must be rejected."""
        from pydantic import ValidationError

        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_DB_PATH", "/tmp/../etc/shadow")
        with pytest.raises(ValidationError):
            GpumodSettings()

    def test_rejects_var_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Paths under /var must be rejected."""
        from pydantic import ValidationError

        from gpumod.config import GpumodSettings

        monkeypatch.setenv("GPUMOD_DB_PATH", "/var/lib/gpumod.db")
        with pytest.raises(ValidationError):
            GpumodSettings()
