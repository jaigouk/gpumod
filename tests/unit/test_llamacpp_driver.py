"""Unit tests for LlamaCppDriver."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from gpumod.models import DriverType, Service, ServiceState
from gpumod.services.drivers.llamacpp import LlamaCppDriver


def make_llamacpp_service() -> Service:
    """Create a Service configured for llama.cpp testing."""
    return Service(
        id="glm-code",
        name="GLM Code",
        driver=DriverType.LLAMACPP,
        vram_mb=19100,
        port=7070,
        unit_name="glm-code.service",
        extra_config={
            "model_name": "devstral",
            "model_path": "/models/devstral.gguf",
        },
    )


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
                return_value=[{"id": "devstral"}],
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


class TestLlamaCppSleep:
    """Tests for LlamaCppDriver.sleep()."""

    async def test_sleep_sends_post_unload_with_model_name(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
            mock_client_cls.return_value = mock_client

            await driver.sleep(svc)
            mock_client.post.assert_awaited_once_with(
                "http://localhost:7070/models/unload",
                json={"model": "devstral"},
            )

    async def test_sleep_uses_model_id_as_fallback(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()
        svc = svc.model_copy(
            update={
                "extra_config": {"model_path": "/models/devstral.gguf"},
                "model_id": "fallback-model",
            }
        )

        with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
            mock_client_cls.return_value = mock_client

            await driver.sleep(svc)
            mock_client.post.assert_awaited_once_with(
                "http://localhost:7070/models/unload",
                json={"model": "fallback-model"},
            )


class TestLlamaCppWake:
    """Tests for LlamaCppDriver.wake()."""

    async def test_wake_sends_post_load_with_model_path(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
            mock_client_cls.return_value = mock_client

            await driver.wake(svc)
            mock_client.post.assert_awaited_once_with(
                "http://localhost:7070/models/load",
                json={"model": "/models/devstral.gguf"},
            )


class TestGetLoadedModels:
    """Tests for LlamaCppDriver._get_loaded_models()."""

    async def test_get_loaded_models_parses_json_response(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        models_data = [{"id": "devstral"}, {"id": "other-model"}]
        mock_response = AsyncMock()
        mock_response.json = lambda: models_data
        mock_response.status_code = 200

        with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await driver._get_loaded_models(svc)
            assert result == [{"id": "devstral"}, {"id": "other-model"}]

    async def test_get_loaded_models_returns_empty_on_connect_error(self) -> None:
        driver = LlamaCppDriver()
        svc = make_llamacpp_service()

        with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client_cls.return_value = mock_client

            result = await driver._get_loaded_models(svc)
            assert result == []


class TestLlamaCppSupportsSleep:
    """Tests for LlamaCppDriver.supports_sleep property."""

    def test_supports_sleep_returns_true(self) -> None:
        driver = LlamaCppDriver()
        assert driver.supports_sleep is True
