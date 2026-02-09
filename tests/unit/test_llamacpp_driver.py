"""Unit tests for LlamaCppDriver."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from gpumod.models import DriverType, Service, ServiceState, SleepMode
from gpumod.services.drivers.llamacpp import LlamaCppDriver

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_llamacpp_service() -> Service:
    """Create a Service configured for llama.cpp testing (single-model mode)."""
    return Service(
        id="glm-code",
        name="GLM Code",
        driver=DriverType.LLAMACPP,
        vram_mb=19100,
        port=7070,
        unit_name="glm-code.service",
        extra_config={
            "model_name": "devstral",
        },
    )


def make_router_service() -> Service:
    """Create a Service configured for llama.cpp router mode (models_dir)."""
    return Service(
        id="nemotron-3-nano",
        name="Nemotron-3-Nano-30B-A3B (llama.cpp)",
        driver=DriverType.LLAMACPP,
        vram_mb=20000,
        port=7070,
        unit_name="nemotron-3-nano.service",
        sleep_mode=SleepMode.ROUTER,
        model_id="unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        extra_config={
            "unit_vars": {
                "models_dir": "/home/user/bin",
                "models_max": 1,
                "no_models_autoload": True,
                "jinja": True,
            }
        },
    )


def make_router_service_with_default_model() -> Service:
    """Create a router Service with explicit default_model in unit_vars."""
    return Service(
        id="nemotron-3-nano",
        name="Nemotron-3-Nano-30B-A3B (llama.cpp)",
        driver=DriverType.LLAMACPP,
        vram_mb=20000,
        port=7070,
        unit_name="nemotron-3-nano.service",
        sleep_mode=SleepMode.ROUTER,
        model_id="unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        extra_config={
            "unit_vars": {
                "models_dir": "/home/user/bin",
                "default_model": "Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL",
                "no_models_autoload": True,
                "jinja": True,
            }
        },
    )


# OAI-format model responses from llama.cpp router
OAI_MODELS_RESPONSE = {
    "data": [
        {
            "id": "Nemotron-3-Nano-30B-A3B-Q4_K_M",
            "object": "model",
            "owned_by": "llamacpp",
            "status": {"value": "unloaded"},
        },
        {
            "id": "Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL",
            "object": "model",
            "owned_by": "llamacpp",
            "status": {"value": "unloaded"},
        },
        {
            "id": "mistralai_Devstral-Small-2-24B-Instruct-2512-Q4_K_M",
            "object": "model",
            "owned_by": "llamacpp",
            "status": {"value": "loaded"},
        },
    ],
    "object": "list",
}


def _mock_http_client(
    get_response: object = None,
    post_response: object = None,
    get_side_effect: Exception | None = None,
) -> AsyncMock:
    """Build a mock httpx.AsyncClient context manager."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    if get_side_effect:
        mock_client.get = AsyncMock(side_effect=get_side_effect)
    elif get_response is not None:
        mock_resp = AsyncMock()
        mock_resp.json = lambda: get_response
        mock_resp.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_resp)
    if post_response is not None:
        mock_client.post = AsyncMock(return_value=post_response)
    else:
        mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
    return mock_client


# ---------------------------------------------------------------------------
# Tests: start / stop
# ---------------------------------------------------------------------------


class TestLlamaCppStart:
    """Tests for LlamaCppDriver.start()."""

    async def test_start_calls_systemd_start_with_unit_name(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        with patch("gpumod.services.drivers.llamacpp.systemd") as mock_systemd:
            mock_systemd.start = AsyncMock()
            await driver.start(svc)
            mock_systemd.start.assert_awaited_once_with("glm-code.service")

    async def test_start_raises_value_error_if_no_unit_name(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()
        svc = svc.model_copy(update={"unit_name": None})

        with pytest.raises(ValueError, match="unit_name"):
            await driver.start(svc)


class TestLlamaCppStop:
    """Tests for LlamaCppDriver.stop()."""

    async def test_stop_calls_systemd_stop_with_unit_name(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        with patch("gpumod.services.drivers.llamacpp.systemd") as mock_systemd:
            mock_systemd.stop = AsyncMock()
            await driver.stop(svc)
            mock_systemd.stop.assert_awaited_once_with("glm-code.service")


# ---------------------------------------------------------------------------
# Tests: health_check
# ---------------------------------------------------------------------------


class TestLlamaCppHealthCheck:
    """Tests for LlamaCppDriver.health_check()."""

    async def test_health_check_returns_true_on_200(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await driver.health_check(svc)
            assert result is True

    async def test_health_check_returns_false_on_connect_error(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client_cls.return_value = mock_client

            result = await driver.health_check(svc)
            assert result is False


# ---------------------------------------------------------------------------
# Tests: status
# ---------------------------------------------------------------------------


class TestLlamaCppStatus:
    """Tests for LlamaCppDriver.status()."""

    async def test_status_returns_stopped_when_inactive(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        with patch("gpumod.services.drivers.llamacpp.systemd") as mock_systemd:
            mock_systemd.is_active = AsyncMock(return_value=False)

            result = await driver.status(svc)
            assert result.state == ServiceState.STOPPED

    async def test_status_returns_sleeping_when_active_and_no_models_loaded(
        self,
    ) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        with (
            patch("gpumod.services.drivers.llamacpp.systemd") as mock_systemd,
            patch.object(driver, "_get_loaded_models", return_value=[]) as _mock_models,
        ):
            mock_systemd.is_active = AsyncMock(return_value=True)

            result = await driver.status(svc)
            assert result.state == ServiceState.SLEEPING

    async def test_status_returns_running_when_active_and_models_loaded(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        with (
            patch("gpumod.services.drivers.llamacpp.systemd") as mock_systemd,
            patch.object(
                driver,
                "_get_loaded_models",
                return_value=[{"id": "devstral", "status": {"value": "loaded"}}],
            ) as _mock_models,
        ):
            mock_systemd.is_active = AsyncMock(return_value=True)

            result = await driver.status(svc)
            assert result.state == ServiceState.RUNNING
            assert result.health_ok is True

    async def test_status_returns_unknown_on_error(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        with patch("gpumod.services.drivers.llamacpp.systemd") as mock_systemd:
            mock_systemd.is_active = AsyncMock(side_effect=RuntimeError("boom"))

            result = await driver.status(svc)
            assert result.state == ServiceState.UNKNOWN


# ---------------------------------------------------------------------------
# Tests: _get_all_models (OAI response parsing)
# ---------------------------------------------------------------------------


class TestGetAllModels:
    """Tests for LlamaCppDriver._get_all_models()."""

    async def test_parses_oai_format_response(self) -> None:
        """OAI response with data array should be parsed correctly."""
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()
        mock_client = _mock_http_client(get_response=OAI_MODELS_RESPONSE)

        with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await driver._get_all_models(svc)

        assert len(result) == 3
        assert result[0]["id"] == "Nemotron-3-Nano-30B-A3B-Q4_K_M"
        assert result[2]["id"] == "mistralai_Devstral-Small-2-24B-Instruct-2512-Q4_K_M"

    async def test_handles_flat_list_response(self) -> None:
        """Flat list response (older llama.cpp) should be returned as-is."""
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()
        flat_list = [{"id": "model-a"}, {"id": "model-b"}]
        mock_client = _mock_http_client(get_response=flat_list)

        with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await driver._get_all_models(svc)

        assert result == flat_list

    async def test_returns_empty_on_connect_error(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()
        mock_client = _mock_http_client(get_side_effect=httpx.ConnectError("refused"))

        with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await driver._get_all_models(svc)

        assert result == []

    async def test_returns_empty_on_empty_data(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()
        mock_client = _mock_http_client(get_response={"data": [], "object": "list"})

        with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await driver._get_all_models(svc)

        assert result == []


# ---------------------------------------------------------------------------
# Tests: _get_loaded_models (filters by loaded status)
# ---------------------------------------------------------------------------


class TestGetLoadedModels:
    """Tests for LlamaCppDriver._get_loaded_models()."""

    async def test_filters_to_loaded_models_only(self) -> None:
        """Only models with status.value == 'loaded' should be returned."""
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        with patch.object(driver, "_get_all_models") as mock_all:
            mock_all.return_value = OAI_MODELS_RESPONSE["data"]
            result = await driver._get_loaded_models(svc)

        assert len(result) == 1
        assert result[0]["id"] == "mistralai_Devstral-Small-2-24B-Instruct-2512-Q4_K_M"

    async def test_returns_empty_when_no_models_loaded(self) -> None:
        """All unloaded models should return empty list."""
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        all_unloaded = [
            {"id": "model-a", "status": {"value": "unloaded"}},
            {"id": "model-b", "status": {"value": "unloaded"}},
        ]

        with patch.object(driver, "_get_all_models") as mock_all:
            mock_all.return_value = all_unloaded
            result = await driver._get_loaded_models(svc)

        assert result == []

    async def test_returns_empty_on_connect_error(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        with patch.object(driver, "_get_all_models") as mock_all:
            mock_all.return_value = []
            result = await driver._get_loaded_models(svc)

        assert result == []


# ---------------------------------------------------------------------------
# Tests: sleep
# ---------------------------------------------------------------------------


class TestLlamaCppSleep:
    """Tests for LlamaCppDriver.sleep()."""

    async def test_sleep_discovers_loaded_model_and_unloads(self) -> None:
        """sleep() should query loaded models and unload the first one."""
        driver = LlamaCppDriver()
        svc = make_router_service()

        with (
            patch.object(
                driver,
                "_get_loaded_models",
                return_value=[
                    {"id": "Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL", "status": {"value": "loaded"}}
                ],
            ),
            patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as cls,
        ):
            mock_client = _mock_http_client()
            cls.return_value = mock_client
            await driver.sleep(svc)

            mock_client.post.assert_awaited_once_with(
                "http://localhost:7070/models/unload",
                json={"model": "Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL"},
            )

    async def test_sleep_falls_back_to_resolve_model_name(self) -> None:
        """When no loaded models found, fall back to _resolve_model_name."""
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()  # has model_name="devstral"

        with (
            patch.object(driver, "_get_loaded_models", return_value=[]),
            patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as cls,
        ):
            mock_client = _mock_http_client()
            cls.return_value = mock_client
            await driver.sleep(svc)

            mock_client.post.assert_awaited_once_with(
                "http://localhost:7070/models/unload",
                json={"model": "devstral"},
            )

    async def test_sleep_uses_model_id_as_final_fallback(self) -> None:
        """When no model_name and no loaded models, use model_id."""
        driver = LlamaCppDriver()
        svc = make_router_service()  # has model_id but no model_name

        with (
            patch.object(driver, "_get_loaded_models", return_value=[]),
            patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as cls,
        ):
            mock_client = _mock_http_client()
            cls.return_value = mock_client
            await driver.sleep(svc)

            mock_client.post.assert_awaited_once_with(
                "http://localhost:7070/models/unload",
                json={"model": "unsloth/Nemotron-3-Nano-30B-A3B-GGUF"},
            )


# ---------------------------------------------------------------------------
# Tests: wake
# ---------------------------------------------------------------------------


class TestLlamaCppWake:
    """Tests for LlamaCppDriver.wake()."""

    async def test_wake_uses_default_model_from_unit_vars(self) -> None:
        """wake() should prefer default_model from unit_vars."""
        driver = LlamaCppDriver()
        svc = make_router_service_with_default_model()

        with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as cls:
            mock_client = _mock_http_client()
            cls.return_value = mock_client
            await driver.wake(svc)

            mock_client.post.assert_awaited_once_with(
                "http://localhost:7070/models/load",
                json={"model": "Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL"},
            )

    async def test_wake_discovers_model_matching_model_id(self) -> None:
        """wake() should discover matching model from server when no default_model."""
        driver = LlamaCppDriver()
        svc = make_router_service()  # model_id="unsloth/Nemotron-3-Nano-30B-A3B-GGUF"

        with patch.object(driver, "_discover_model_to_load") as mock_discover:
            mock_discover.return_value = "Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL"

            with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as cls:
                mock_client = _mock_http_client()
                cls.return_value = mock_client
                await driver.wake(svc)

                mock_client.post.assert_awaited_once_with(
                    "http://localhost:7070/models/load",
                    json={"model": "Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL"},
                )

    async def test_wake_does_nothing_when_no_model_found(self) -> None:
        """wake() should skip POST when no model found to load."""
        driver = LlamaCppDriver()
        svc = make_router_service()

        with (
            patch.object(driver, "_discover_model_to_load", return_value=None),
            patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as cls,
        ):
            mock_client = _mock_http_client()
            cls.return_value = mock_client
            await driver.wake(svc)

            mock_client.post.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests: _discover_model_to_load
# ---------------------------------------------------------------------------


class TestDiscoverModelToLoad:
    """Tests for LlamaCppDriver._discover_model_to_load()."""

    async def test_matches_model_id_to_alias(self) -> None:
        """Should match 'Nemotron-3-Nano-30B-A3B' from model_id to router alias."""
        driver = LlamaCppDriver()
        svc = make_router_service()  # model_id="unsloth/Nemotron-3-Nano-30B-A3B-GGUF"

        with patch.object(driver, "_get_all_models") as mock_all:
            mock_all.return_value = OAI_MODELS_RESPONSE["data"]
            result = await driver._discover_model_to_load(svc)

        # Should match the first Nemotron variant (Q4_K_M)
        assert result == "Nemotron-3-Nano-30B-A3B-Q4_K_M"

    async def test_falls_back_to_first_unloaded(self) -> None:
        """When model_id doesn't match, load the first unloaded model."""
        driver = LlamaCppDriver()
        svc = make_router_service()
        svc = svc.model_copy(update={"model_id": "org/no-match"})

        all_models = [
            {"id": "first-model", "status": {"value": "unloaded"}},
            {"id": "second-model", "status": {"value": "unloaded"}},
        ]

        with patch.object(driver, "_get_all_models") as mock_all:
            mock_all.return_value = all_models
            result = await driver._discover_model_to_load(svc)

        assert result == "first-model"

    async def test_returns_none_when_all_loaded(self) -> None:
        """When all models are loaded, return None."""
        driver = LlamaCppDriver()
        svc = make_router_service()

        all_loaded = [
            {"id": "model-a", "status": {"value": "loaded"}},
        ]

        with patch.object(driver, "_get_all_models") as mock_all:
            mock_all.return_value = all_loaded
            result = await driver._discover_model_to_load(svc)

        assert result is None

    async def test_returns_none_when_no_models(self) -> None:
        """When server returns empty models list, return None."""
        driver = LlamaCppDriver()
        svc = make_router_service()

        with patch.object(driver, "_get_all_models") as mock_all:
            mock_all.return_value = []
            result = await driver._discover_model_to_load(svc)

        assert result is None

    async def test_skips_loaded_models_when_matching(self) -> None:
        """Model matching should only consider unloaded models."""
        driver = LlamaCppDriver()
        svc = make_router_service()

        # Devstral is loaded, Nemotron variants are unloaded
        with patch.object(driver, "_get_all_models") as mock_all:
            mock_all.return_value = OAI_MODELS_RESPONSE["data"]
            result = await driver._discover_model_to_load(svc)

        # Should NOT return the loaded devstral model
        assert "Devstral" not in (result or "")

    async def test_strips_gguf_suffix_from_model_id(self) -> None:
        """Should strip -GGUF from model_id before matching."""
        driver = LlamaCppDriver()
        svc = make_router_service()
        svc = svc.model_copy(update={"model_id": "org/MyModel-GGUF"})

        models = [
            {"id": "MyModel-Q4_K_M", "status": {"value": "unloaded"}},
        ]

        with patch.object(driver, "_get_all_models") as mock_all:
            mock_all.return_value = models
            result = await driver._discover_model_to_load(svc)

        assert result == "MyModel-Q4_K_M"


# ---------------------------------------------------------------------------
# Tests: _get_default_model
# ---------------------------------------------------------------------------


class TestGetDefaultModel:
    """Tests for LlamaCppDriver._get_default_model()."""

    def test_returns_default_model_from_unit_vars(self) -> None:
        svc = make_router_service_with_default_model()
        result = LlamaCppDriver._get_default_model(svc)
        assert result == "Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL"

    def test_returns_none_when_no_unit_vars(self) -> None:
        svc = make_llamacpp_service()
        result = LlamaCppDriver._get_default_model(svc)
        assert result is None

    def test_returns_none_when_no_default_model_in_unit_vars(self) -> None:
        svc = make_router_service()  # has unit_vars but no default_model
        result = LlamaCppDriver._get_default_model(svc)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: supports_sleep
# ---------------------------------------------------------------------------


class TestLlamaCppSupportsSleep:
    """Tests for LlamaCppDriver.supports_sleep property."""

    def test_supports_sleep_returns_true(self) -> None:
        driver = LlamaCppDriver()
        assert driver.supports_sleep is True
