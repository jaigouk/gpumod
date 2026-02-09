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
    """Tests for VLLMDriver.sleep().

    Per vLLM docs, sleep endpoint uses query params, not JSON body:
    POST /sleep?level=1 (L1: offload to CPU)
    POST /sleep?level=2 (L2: discard weights)
    """

    async def test_sleep_uses_query_param_not_json_body(self) -> None:
        """sleep() should use ?level=N query param, not JSON body."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
            mock_client_cls.return_value = mock_client

            await driver.sleep(svc, level=1)
            mock_client.post.assert_awaited_once()
            call_args = mock_client.post.call_args
            # Should use params={"level": 1}, not json={"level": ...}
            assert call_args.kwargs.get("params") == {"level": 1}
            assert "json" not in call_args.kwargs

    async def test_sleep_l1_sends_level_1_query_param(self) -> None:
        """sleep(level=1) should send POST /sleep?level=1."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
            mock_client_cls.return_value = mock_client

            await driver.sleep(svc, level=1)
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://localhost:8200/sleep"
            assert call_args.kwargs["params"]["level"] == 1

    async def test_sleep_l2_sends_level_2_query_param(self) -> None:
        """sleep(level=2) should send POST /sleep?level=2."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
            mock_client_cls.return_value = mock_client

            await driver.sleep(svc, level=2)
            call_args = mock_client.post.call_args
            assert call_args.kwargs["params"]["level"] == 2

    async def test_sleep_rejects_invalid_level(self) -> None:
        """sleep() should reject levels other than 1 or 2."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        with pytest.raises(ValueError, match="level"):
            await driver.sleep(svc, level=3)

        with pytest.raises(ValueError, match="level"):
            await driver.sleep(svc, level=0)

    async def test_sleep_handles_connection_error(self) -> None:
        """sleep() should raise on connection failure."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.ConnectError):
                await driver.sleep(svc, level=1)

    async def test_sleep_default_level_is_1(self) -> None:
        """sleep() without level arg should default to L1."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
            mock_client_cls.return_value = mock_client

            await driver.sleep(svc)  # No level specified
            call_args = mock_client.post.call_args
            assert call_args.kwargs["params"]["level"] == 1


class TestVLLMWake:
    """Tests for VLLMDriver.wake().

    Per vLLM docs, the wake endpoint is /wake_up, not /wake.
    Supports optional tags query param for selective wake.
    """

    async def test_wake_uses_wake_up_endpoint(self) -> None:
        """wake() should call /wake_up, not /wake."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
            mock_client_cls.return_value = mock_client

            await driver.wake(svc)
            call_args = mock_client.post.call_args
            url = call_args[0][0]
            assert "/wake_up" in url
            # Ensure it's not just /wake
            assert url.endswith("/wake_up")

    async def test_wake_without_tags_sends_no_params(self) -> None:
        """wake() without tags should not include params."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
            mock_client_cls.return_value = mock_client

            await driver.wake(svc)
            call_args = mock_client.post.call_args
            # No params or params is None/empty
            params = call_args.kwargs.get("params")
            assert params is None or params == {}

    async def test_wake_with_tags_sends_tags_param(self) -> None:
        """wake(tags='weights') should send POST /wake_up?tags=weights."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
            mock_client_cls.return_value = mock_client

            await driver.wake(svc, tags="weights")
            call_args = mock_client.post.call_args
            assert call_args.kwargs["params"]["tags"] == "weights"

    async def test_wake_handles_connection_error(self) -> None:
        """wake() should raise on connection failure."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.ConnectError):
                await driver.wake(svc)


class TestVLLMIsSleeping:
    """Tests for VLLMDriver.is_sleeping().

    Per vLLM docs: GET /is_sleeping returns {"is_sleeping": bool}
    """

    async def test_is_sleeping_queries_api_endpoint(self) -> None:
        """is_sleeping() should query GET /is_sleeping."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {"is_sleeping": True}

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await driver.is_sleeping(svc)
            mock_client.get.assert_awaited_once()
            call_url = mock_client.get.call_args[0][0]
            assert "/is_sleeping" in call_url
            assert result is True

    async def test_is_sleeping_returns_false_when_not_sleeping(self) -> None:
        """is_sleeping() should return False from API response."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {"is_sleeping": False}

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await driver.is_sleeping(svc)
            assert result is False

    async def test_is_sleeping_returns_false_on_connection_error(self) -> None:
        """is_sleeping() should return False if service not reachable."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_client_cls.return_value = mock_client

            # Should not raise, just return False
            result = await driver.is_sleeping(svc)
            assert result is False

    async def test_is_sleeping_returns_false_on_404(self) -> None:
        """is_sleeping() should return False if endpoint doesn't exist (dev mode off)."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        mock_response = AsyncMock()
        mock_response.status_code = 404

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await driver.is_sleeping(svc)
            assert result is False

    async def test_is_sleeping_returns_false_on_missing_key(self) -> None:
        """is_sleeping() should return False if response lacks 'is_sleeping' key."""
        driver = VLLMDriver()
        svc = make_vllm_service()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = dict  # Missing key - returns empty dict

        with patch("gpumod.services.drivers.vllm.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await driver.is_sleeping(svc)
            assert result is False


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
            patch.object(driver, "is_sleeping", return_value=False) as _mock_sleep,
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
            patch.object(driver, "is_sleeping", return_value=True) as _mock_sleep,
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
