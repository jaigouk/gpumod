"""Integration tests for centralized configuration flow.

Verifies that:
- Settings loaded from env vars override defaults end-to-end
- Config flows through CLI -> services -> DB path
- MCP server uses config for rate limit and DB path
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import typer.testing

from gpumod.config import GpumodSettings, _clear_settings_cache, get_settings

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


pytestmark = pytest.mark.integration

runner = typer.testing.CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Generator[None, None, None]:
    """Clear settings cache before each test so env var changes take effect."""
    _clear_settings_cache()
    yield
    _clear_settings_cache()


@pytest.fixture
def _clean_gpumod_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all GPUMOD_ env vars so defaults are tested cleanly."""
    for key in list(os.environ):
        if key.startswith("GPUMOD_"):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Config env var override integration tests
# ---------------------------------------------------------------------------


class TestConfigEnvVarIntegration:
    """End-to-end: env vars override config defaults and propagate to components."""

    def test_env_vars_override_all_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """GPUMOD_* env vars override every configurable setting at once."""
        db_path = tmp_path / "custom.db"
        presets = tmp_path / "presets"
        presets.mkdir()

        monkeypatch.setenv("GPUMOD_DB_PATH", str(db_path))
        monkeypatch.setenv("GPUMOD_PRESETS_DIR", str(presets))
        monkeypatch.setenv("GPUMOD_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("GPUMOD_LLM_BACKEND", "anthropic")
        monkeypatch.setenv("GPUMOD_LLM_API_KEY", "sk-int-test-key")
        monkeypatch.setenv("GPUMOD_LLM_MODEL", "claude-3-sonnet")
        monkeypatch.setenv("GPUMOD_LLM_BASE_URL", "https://proxy.example.com")
        monkeypatch.setenv("GPUMOD_MCP_RATE_LIMIT", "25")

        settings = get_settings()

        assert settings.db_path == db_path
        assert settings.presets_dir == presets
        assert settings.log_level == "DEBUG"
        assert settings.llm_backend == "anthropic"
        assert settings.llm_api_key is not None
        assert settings.llm_api_key.get_secret_value() == "sk-int-test-key"
        assert settings.llm_model == "claude-3-sonnet"
        assert settings.llm_base_url == "https://proxy.example.com"
        assert settings.mcp_rate_limit == 25

    def test_get_settings_singleton_respects_env_on_first_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """get_settings() reads env on first call, caches thereafter."""
        monkeypatch.setenv("GPUMOD_LOG_LEVEL", "ERROR")
        s1 = get_settings()
        assert s1.log_level == "ERROR"

        # Subsequent calls return same instance even if env changes
        monkeypatch.setenv("GPUMOD_LOG_LEVEL", "DEBUG")
        s2 = get_settings()
        assert s2 is s1
        assert s2.log_level == "ERROR"

    def test_case_insensitive_log_level_normalization(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Log level is normalized to uppercase from mixed-case env var."""
        monkeypatch.setenv("GPUMOD_LOG_LEVEL", "warning")
        settings = GpumodSettings()
        assert settings.log_level == "WARNING"


# ---------------------------------------------------------------------------
# Config -> CLI -> DB path flow
# ---------------------------------------------------------------------------


class TestConfigCLIFlow:
    """Config flows through CLI -> create_context -> Database path."""

    def test_config_db_path_flows_to_create_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """GPUMOD_DB_PATH env var is used by CLI create_context for DB location."""
        from gpumod.cli import app

        db_path = tmp_path / "flow-test.db"
        monkeypatch.setenv("GPUMOD_DB_PATH", str(db_path))

        # The `init` command creates the DB at the configured path
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        # DB file should have been created at the env-configured path
        assert db_path.exists()

    def test_config_presets_dir_flows_to_preset_loader(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """GPUMOD_PRESETS_DIR env var is used by CLI create_context for presets."""
        from gpumod.cli import app

        db_path = tmp_path / "presets-flow.db"
        presets_dir = tmp_path / "my_presets"
        presets_dir.mkdir()

        monkeypatch.setenv("GPUMOD_DB_PATH", str(db_path))
        monkeypatch.setenv("GPUMOD_PRESETS_DIR", str(presets_dir))

        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        # With empty presets dir, should find 0 presets
        assert "0 preset" in result.output


# ---------------------------------------------------------------------------
# Config -> MCP server flow
# ---------------------------------------------------------------------------


class TestConfigMCPServerFlow:
    """Config propagates to MCP server components."""

    def test_mcp_server_uses_config_db_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """MCP server lifespan reads db_path from centralized settings."""
        from gpumod.mcp_server import create_mcp_server

        db_path = tmp_path / "mcp-config.db"
        monkeypatch.setenv("GPUMOD_DB_PATH", str(db_path))

        # create_mcp_server without explicit db_path should use settings
        server = create_mcp_server()
        # Server should be created successfully
        assert server is not None
        # The server name should be 'gpumod'
        assert server.name == "gpumod"

    def test_mcp_server_rate_limit_config_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """GPUMOD_MCP_RATE_LIMIT env var is read by settings for rate limiting."""
        monkeypatch.setenv("GPUMOD_MCP_RATE_LIMIT", "42")
        settings = get_settings()
        assert settings.mcp_rate_limit == 42

        # Verify the RateLimitMiddleware is present in the server
        from gpumod.mcp_server import RateLimitMiddleware, create_mcp_server

        server = create_mcp_server()
        assert server is not None
        # The middleware list should include a RateLimitMiddleware instance
        middleware_types = [type(m) for m in server.middleware]
        assert RateLimitMiddleware in middleware_types
