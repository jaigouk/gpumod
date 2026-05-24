"""Tests for the quiesce gate (gpumod-jj0): heavy-service detection and cooldown."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock

import pytest

from gpumod.models import DriverType, Service, SleepMode
from gpumod.services.heavy import (
    QUIESCE_LAST_HEAVY_STOP_KEY,
    QuiesceVerdict,
    check_quiesce,
    is_heavy,
    record_heavy_stop,
)

# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_service(vram_mb: int = 0) -> Service:
    return Service(
        id="test-svc",
        name="Test Service",
        driver=DriverType.VLLM,
        port=8000,
        vram_mb=vram_mb,
        sleep_mode=SleepMode.NONE,
        health_endpoint="/health",
        unit_name="test-svc.service",
        depends_on=[],
        startup_timeout=60,
        extra_config={},
    )


def _make_db(setting_value: str | None = None) -> AsyncMock:
    """Build a mock Database with get_setting/set_setting."""
    db = AsyncMock()
    db.get_setting = AsyncMock(return_value=setting_value)
    db.set_setting = AsyncMock()
    return db


# ── is_heavy ─────────────────────────────────────────────────────────────


class TestIsHeavy:
    """is_heavy() returns True only when vram_mb > 0."""

    def test_light_service_vram_zero(self) -> None:
        svc = _make_service(vram_mb=0)
        assert is_heavy(svc) is False

    def test_heavy_service_vram_positive(self) -> None:
        svc = _make_service(vram_mb=8000)
        assert is_heavy(svc) is True

    def test_heavy_service_vram_one(self) -> None:
        svc = _make_service(vram_mb=1)
        assert is_heavy(svc) is True


# ── record_heavy_stop ────────────────────────────────────────────────────


class TestRecordHeavyStop:
    """record_heavy_stop() writes epoch timestamp to the settings table."""

    @pytest.mark.asyncio
    async def test_writes_to_db(self) -> None:
        db = _make_db()
        await record_heavy_stop(db)

        db.set_setting.assert_called_once()
        call_args = db.set_setting.call_args
        assert call_args[0][0] == QUIESCE_LAST_HEAVY_STOP_KEY
        # Value should be a parseable float (epoch)
        value = call_args[0][1]
        epoch = float(value)
        assert epoch > 0

    @pytest.mark.asyncio
    async def test_description_is_set(self) -> None:
        db = _make_db()
        await record_heavy_stop(db)

        call_kwargs = db.set_setting.call_args
        # description is passed as keyword or positional
        # set_setting(key, value, description=...)
        assert "description" in call_kwargs.kwargs or len(call_kwargs.args) >= 3


# ── check_quiesce ────────────────────────────────────────────────────────


class TestCheckQuiesce:
    """check_quiesce() enforces the quiesce cooldown window."""

    @pytest.mark.asyncio
    async def test_no_prior_stop_returns_allowed(self) -> None:
        """When no prior stop is recorded, start is allowed."""
        db = _make_db(setting_value=None)
        verdict = await check_quiesce(db, quiesce_seconds=10.0)

        assert verdict.allowed is True
        assert verdict.remaining_seconds == 0
        assert verdict.remediation == ""

    @pytest.mark.asyncio
    async def test_within_window_returns_refused(self) -> None:
        """When a heavy stop occurred recently (within window), refuse."""
        import time

        # Stopped 3 seconds ago, window is 10s
        recent_stop = str(time.time() - 3)
        db = _make_db(setting_value=recent_stop)
        verdict = await check_quiesce(db, quiesce_seconds=10.0)

        assert verdict.allowed is False
        assert verdict.remaining_seconds > 0
        assert verdict.remaining_seconds <= 7.5  # approx
        assert "Quiesce period active" in verdict.remediation
        assert "--no-quiesce" in verdict.remediation

    @pytest.mark.asyncio
    async def test_outside_window_returns_allowed(self) -> None:
        """When a heavy stop occurred long ago (past window), allow."""
        import time

        old_stop = str(time.time() - 20)
        db = _make_db(setting_value=old_stop)
        verdict = await check_quiesce(db, quiesce_seconds=10.0)

        assert verdict.allowed is True
        assert verdict.remaining_seconds == 0
        assert verdict.remediation == ""

    @pytest.mark.asyncio
    async def test_clock_skew_returns_allowed(self) -> None:
        """When clock skew makes elapsed negative, treat as allowed."""
        import time

        future_stop = str(time.time() + 9999)
        db = _make_db(setting_value=future_stop)
        verdict = await check_quiesce(db, quiesce_seconds=10.0)

        assert verdict.allowed is True
        assert verdict.remaining_seconds == 0
        assert verdict.remediation == ""

    @pytest.mark.asyncio
    async def test_corrupted_value_returns_allowed(self) -> None:
        """When the stored value is garbage, treat as no prior stop."""
        db = _make_db(setting_value="not-a-number")
        verdict = await check_quiesce(db, quiesce_seconds=10.0)

        assert verdict.allowed is True
        assert verdict.remaining_seconds == 0
        assert verdict.remediation == ""

    @pytest.mark.asyncio
    async def test_empty_string_value_returns_allowed(self) -> None:
        """Empty string is treated as corrupted → allowed."""
        db = _make_db(setting_value="")
        verdict = await check_quiesce(db, quiesce_seconds=10.0)

        assert verdict.allowed is True
        assert verdict.remaining_seconds == 0
        assert verdict.remediation == ""

    @pytest.mark.asyncio
    async def test_zero_quiesce_seconds_always_allowed(self) -> None:
        """With quiesce_seconds=0, the gate never blocks."""
        import time

        recent_stop = str(time.time() - 0.1)
        db = _make_db(setting_value=recent_stop)
        verdict = await check_quiesce(db, quiesce_seconds=0.0)

        assert verdict.allowed is True

    @pytest.mark.asyncio
    async def test_verdict_is_frozen_dataclass(self) -> None:
        """QuiesceVerdict is immutable."""
        verdict = QuiesceVerdict(allowed=True, remaining_seconds=0, remediation="")
        with pytest.raises(FrozenInstanceError):
            verdict.allowed = False  # type: ignore[misc]
