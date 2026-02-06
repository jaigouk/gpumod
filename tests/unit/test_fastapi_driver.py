"""Tests for the FastAPI service driver."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from gpumod.models import DriverType, Service, ServiceState, SleepMode
from gpumod.services.drivers.fastapi import FastAPIDriver


def make_fastapi_service(
    health_endpoint: str = "/health",
    unit_name: str | None = "qwen3-asr.service",
) -> Service:
    """Create a Service configured for the FastAPI driver."""
    return Service(
        id="qwen3-asr",
        name="Qwen3 ASR",
        driver=DriverType.FASTAPI,
        port=8300,
        vram_mb=3000,
        sleep_mode=SleepMode.NONE,
        health_endpoint=health_endpoint,
        unit_name=unit_name,
    )


class TestFastAPIDriverStart:
    """start() delegates to systemd."""

    async def test_start_calls_systemd_start(self) -> None:
        driver = FastAPIDriver()
        svc = make_fastapi_service()
        with patch("gpumod.services.drivers.fastapi.systemd") as mock_systemd:
            mock_systemd.start = AsyncMock()
            await driver.start(svc)
            mock_systemd.start.assert_awaited_once_with("qwen3-asr.service")

    async def test_start_raises_if_no_unit_name(self) -> None:
        driver = FastAPIDriver()
        svc = make_fastapi_service(unit_name=None)
        with pytest.raises(ValueError, match="unit_name"):
            await driver.start(svc)


class TestFastAPIDriverStop:
    """stop() delegates to systemd."""

    async def test_stop_calls_systemd_stop(self) -> None:
        driver = FastAPIDriver()
        svc = make_fastapi_service()
        with patch("gpumod.services.drivers.fastapi.systemd") as mock_systemd:
            mock_systemd.stop = AsyncMock()
            await driver.stop(svc)
            mock_systemd.stop.assert_awaited_once_with("qwen3-asr.service")


class TestFastAPIDriverHealthCheck:
    """health_check() hits the configured HTTP health endpoint."""

    async def test_health_check_uses_custom_endpoint(self) -> None:
        driver = FastAPIDriver()
        svc = make_fastapi_service(health_endpoint="/api/health")
        with patch("gpumod.services.drivers.fastapi.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_response)
            result = await driver.health_check(svc)
            mock_client.get.assert_awaited_once_with(
                "http://localhost:8300/api/health", timeout=5.0
            )
            assert result is True

    async def test_health_check_uses_default_endpoint(self) -> None:
        driver = FastAPIDriver()
        svc = make_fastapi_service()  # default /health
        with patch("gpumod.services.drivers.fastapi.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_response)
            result = await driver.health_check(svc)
            mock_client.get.assert_awaited_once_with("http://localhost:8300/health", timeout=5.0)
            assert result is True

    async def test_health_check_returns_true_on_200(self) -> None:
        driver = FastAPIDriver()
        svc = make_fastapi_service()
        with patch("gpumod.services.drivers.fastapi.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_response)
            result = await driver.health_check(svc)
            assert result is True

    async def test_health_check_returns_false_on_connection_error(self) -> None:
        driver = FastAPIDriver()
        svc = make_fastapi_service()
        with patch("gpumod.services.drivers.fastapi.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            result = await driver.health_check(svc)
            assert result is False

    async def test_health_check_returns_false_on_non_200(self) -> None:
        driver = FastAPIDriver()
        svc = make_fastapi_service()
        with patch("gpumod.services.drivers.fastapi.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_response = AsyncMock()
            mock_response.status_code = 503
            mock_client.get = AsyncMock(return_value=mock_response)
            result = await driver.health_check(svc)
            assert result is False


class TestFastAPIDriverStatus:
    """status() combines systemd state + health check."""

    async def test_status_returns_stopped_when_unit_inactive(self) -> None:
        driver = FastAPIDriver()
        svc = make_fastapi_service()
        with patch("gpumod.services.drivers.fastapi.systemd") as mock_systemd:
            mock_systemd.is_active = AsyncMock(return_value=False)
            status = await driver.status(svc)
            assert status.state == ServiceState.STOPPED

    async def test_status_returns_running_when_active_and_healthy(self) -> None:
        driver = FastAPIDriver()
        svc = make_fastapi_service()
        with (
            patch("gpumod.services.drivers.fastapi.systemd") as mock_systemd,
            patch.object(driver, "health_check", new_callable=AsyncMock) as mock_health,
        ):
            mock_systemd.is_active = AsyncMock(return_value=True)
            mock_health.return_value = True
            status = await driver.status(svc)
            assert status.state == ServiceState.RUNNING
            assert status.health_ok is True

    async def test_status_returns_unhealthy_when_active_but_health_fails(self) -> None:
        driver = FastAPIDriver()
        svc = make_fastapi_service()
        with (
            patch("gpumod.services.drivers.fastapi.systemd") as mock_systemd,
            patch.object(driver, "health_check", new_callable=AsyncMock) as mock_health,
        ):
            mock_systemd.is_active = AsyncMock(return_value=True)
            mock_health.return_value = False
            status = await driver.status(svc)
            assert status.state == ServiceState.UNHEALTHY
            assert status.health_ok is False

    async def test_status_returns_unknown_on_unexpected_error(self) -> None:
        driver = FastAPIDriver()
        svc = make_fastapi_service()
        with patch("gpumod.services.drivers.fastapi.systemd") as mock_systemd:
            mock_systemd.is_active = AsyncMock(side_effect=RuntimeError("unexpected"))
            status = await driver.status(svc)
            assert status.state == ServiceState.UNKNOWN
            assert status.last_error is not None


class TestFastAPIDriverSupportsSleep:
    """FastAPIDriver does not support sleep."""

    def test_supports_sleep_is_false(self) -> None:
        driver = FastAPIDriver()
        assert driver.supports_sleep is False
