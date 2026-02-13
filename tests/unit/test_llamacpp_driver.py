"""Unit tests for LlamaCppDriver."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from gpumod.models import DriverType, Service, ServiceState, SleepMode, VRAMUsage
from gpumod.services.drivers.llamacpp import LlamaCppDriver
from gpumod.services.vram import InsufficientVRAMError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_llamacpp_service() -> Service:
    """Create a Service configured for llama.cpp testing (single-model mode)."""
    return Service(
        id="qwen3-coder",
        name="Qwen3 Coder",
        driver=DriverType.LLAMACPP,
        vram_mb=19100,
        port=7070,
        unit_name="qwen3-coder.service",
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
            mock_systemd.start.assert_awaited_once_with("qwen3-coder.service")

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
            mock_systemd.stop.assert_awaited_once_with("qwen3-coder.service")


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


# ---------------------------------------------------------------------------
# Tests: VRAM preflight in wake() (gpumod-277)
# ---------------------------------------------------------------------------


class TestWakeVramPreflight:
    """Tests for VRAM preflight check in wake() (gpumod-277).

    When a VRAMTracker is provided to LlamaCppDriver, wake() should check
    available VRAM before attempting to load a model. This prevents system
    crashes from extreme VRAM deficits (e.g., 30GB model vs 1GB free).
    """

    async def test_wake_raises_insufficient_vram_error_when_not_enough_vram(self) -> None:
        """wake() should raise InsufficientVRAMError when VRAM is insufficient.

        This tests the fix for gpumod-277: manual curl to /models/load with
        insufficient VRAM caused system crashes. Now wake() checks VRAM first.
        """
        # Mock VRAMTracker with very little free VRAM
        vram_tracker = AsyncMock()
        vram_tracker.get_usage = AsyncMock(
            return_value=VRAMUsage(total_mb=24000, used_mb=23000, free_mb=1000)
        )

        # Create driver with VRAMTracker injection
        driver = LlamaCppDriver(vram_tracker=vram_tracker)

        # Service requires 20GB VRAM
        svc = make_router_service_with_default_model()  # vram_mb=20000

        # wake() should raise InsufficientVRAMError
        with pytest.raises(InsufficientVRAMError) as exc_info:
            await driver.wake(svc)

        assert exc_info.value.required_mb == 20000
        assert exc_info.value.available_mb == 1000

    async def test_wake_succeeds_when_enough_vram(self) -> None:
        """wake() should proceed normally when sufficient VRAM is available."""
        # Mock VRAMTracker with plenty of free VRAM
        vram_tracker = AsyncMock()
        vram_tracker.get_usage = AsyncMock(
            return_value=VRAMUsage(total_mb=24000, used_mb=1000, free_mb=23000)
        )

        driver = LlamaCppDriver(vram_tracker=vram_tracker)
        svc = make_router_service_with_default_model()  # vram_mb=20000

        with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as cls:
            mock_client = _mock_http_client()
            cls.return_value = mock_client
            await driver.wake(svc)

            # Should proceed to load the model
            mock_client.post.assert_awaited_once()

    async def test_wake_without_vram_tracker_does_not_check(self) -> None:
        """wake() without VRAMTracker should not perform VRAM check (backward compatible)."""
        driver = LlamaCppDriver()  # No VRAMTracker
        svc = make_router_service_with_default_model()

        with patch("gpumod.services.drivers.llamacpp.httpx.AsyncClient") as cls:
            mock_client = _mock_http_client()
            cls.return_value = mock_client
            # Should NOT raise, even though we can't verify VRAM
            await driver.wake(svc)

            mock_client.post.assert_awaited_once()

    async def test_wake_includes_safety_margin_in_vram_check(self) -> None:
        """VRAM check should include a safety margin (512MB default).

        Service needs 20GB, but check should require 20GB + 512MB margin.
        """
        vram_tracker = AsyncMock()
        # 20400MB free: more than 20000MB required but less than 20000+512=20512MB
        vram_tracker.get_usage = AsyncMock(
            return_value=VRAMUsage(total_mb=24000, used_mb=3600, free_mb=20400)
        )

        driver = LlamaCppDriver(vram_tracker=vram_tracker)
        svc = make_router_service_with_default_model()  # vram_mb=20000

        # Should raise because 20400 < 20000 + 512
        with pytest.raises(InsufficientVRAMError):
            await driver.wake(svc)
