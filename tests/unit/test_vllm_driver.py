"""Unit tests for VLLMDriver."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from gpumod.models import DriverType, Service, ServiceState
from gpumod.services.drivers.vllm import VLLMDriver


def make_vllm_service() -> Service:
    """Create a Service configured for vLLM testing."""
    return Service(
        id="test-vllm",
        name="Test vLLM",
        driver=DriverType.VLLM,
        vram_mb=5000,
        port=8200,
        unit_name="test-vllm.service",
    )


class TestVLLMStart:
    """Tests for VLLMDriver.start()."""

    async def test_start_calls_systemd_start_with_unit_name(self) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.systemd") as mock_systemd:
            mock_systemd.start = AsyncMock()
            await driver.start(svc)
            mock_systemd.start.assert_awaited_once_with("test-vllm.service")

    async def test_start_raises_value_error_if_no_unit_name(self) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()
        svc = svc.model_copy(update={"unit_name": None})

        with pytest.raises(ValueError, match="unit_name"):
            await driver.start(svc)


class TestVLLMStop:
    """Tests for VLLMDriver.stop()."""

    async def test_stop_calls_systemd_stop_with_unit_name(self) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.systemd") as mock_systemd:
            mock_systemd.stop = AsyncMock()
            await driver.stop(svc)
            mock_systemd.stop.assert_awaited_once_with("test-vllm.service")


class TestVLLMHealthCheck:
    """Tests for VLLMDriver.health_check()."""

    async def test_health_check_returns_true_on_200(self) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()

        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await driver.health_check(svc)
            assert result is True

    async def test_health_check_returns_false_on_connect_error(self) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client_cls.return_value = mock_client

            result = await driver.health_check(svc)
            assert result is False

    async def test_health_check_returns_false_on_non_200(self) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()

        mock_response = AsyncMock()
        mock_response.status_code = 500

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await driver.health_check(svc)
            assert result is False

    async def test_health_check_returns_false_when_port_is_none(self) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()
        svc = svc.model_copy(update={"port": None})

        result = await driver.health_check(svc)
        assert result is False


class TestVLLMSleep:
    """Tests for VLLMDriver.sleep()."""

    async def test_sleep_sends_post_with_l1_level(self) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
            mock_client_cls.return_value = mock_client

            await driver.sleep(svc, level="l1")
            mock_client.post.assert_awaited_once_with(
                "http://localhost:8200/sleep", json={"level": "l1"}
            )

    async def test_sleep_sends_post_with_l2_level(self) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
            mock_client_cls.return_value = mock_client

            await driver.sleep(svc, level="l2")
            mock_client.post.assert_awaited_once_with(
                "http://localhost:8200/sleep", json={"level": "l2"}
            )


class TestVLLMWake:
    """Tests for VLLMDriver.wake()."""

    async def test_wake_sends_post(self) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
            mock_client_cls.return_value = mock_client

            await driver.wake(svc)
            mock_client.post.assert_awaited_once_with("http://localhost:8200/wake")


class TestVLLMStatus:
    """Tests for VLLMDriver.status()."""

    async def test_status_returns_stopped_when_inactive(self) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.systemd") as mock_systemd:
            mock_systemd.is_active = AsyncMock(return_value=False)

            result = await driver.status(svc)
            assert result.state == ServiceState.STOPPED

    async def test_status_returns_running_when_active_and_healthy_not_sleeping(
        self,
    ) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()

        with (
            patch("gpumod.services.drivers.vllm.systemd") as mock_systemd,
            patch.object(driver, "health_check", return_value=True) as _mock_health,
            patch.object(driver, "_is_sleeping", return_value=False) as _mock_sleep,
        ):
            mock_systemd.is_active = AsyncMock(return_value=True)

            result = await driver.status(svc)
            assert result.state == ServiceState.RUNNING
            assert result.health_ok is True

    async def test_status_returns_sleeping_when_active_and_sleeping(self) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()

        with (
            patch("gpumod.services.drivers.vllm.systemd") as mock_systemd,
            patch.object(driver, "health_check", return_value=True) as _mock_health,
            patch.object(driver, "_is_sleeping", return_value=True) as _mock_sleep,
        ):
            mock_systemd.is_active = AsyncMock(return_value=True)

            result = await driver.status(svc)
            assert result.state == ServiceState.SLEEPING

    async def test_status_returns_unhealthy_when_active_but_health_fails(self) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()

        with (
            patch("gpumod.services.drivers.vllm.systemd") as mock_systemd,
            patch.object(driver, "health_check", return_value=False) as _mock_health,
        ):
            mock_systemd.is_active = AsyncMock(return_value=True)

            result = await driver.status(svc)
            assert result.state == ServiceState.UNHEALTHY
            assert result.health_ok is False

    async def test_status_returns_unknown_on_unexpected_error(self) -> None:
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.systemd") as mock_systemd:
            mock_systemd.is_active = AsyncMock(side_effect=RuntimeError("boom"))

            result = await driver.status(svc)
            assert result.state == ServiceState.UNKNOWN


class TestVLLMSupportsSleep:
    """Tests for VLLMDriver.supports_sleep property."""

    def test_supports_sleep_returns_true(self) -> None:
        driver = VLLMDriver()
        assert driver.supports_sleep is True
