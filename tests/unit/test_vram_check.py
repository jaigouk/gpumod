"""Tests for VRAM preflight check (gpumod-89z).

Tests cover:
- VRAMCheck preflight validation
- Passing when VRAM fits
- Failing with suggestions when VRAM exceeds capacity
- Handling services without VRAM requirements
- Suggestion generation (reduced n_gpu_layers, smaller quants)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gpumod.preflight.vram_check import VRAMCheck, VRAMSuggestion

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service() -> MagicMock:
    """Create a mock llama.cpp service with VRAM config."""
    service = MagicMock()
    service.id = "qwen3-coder"
    service.driver = "llamacpp"
    service.vram_mb = 22000  # Configured VRAM requirement
    service.model_id = "unsloth/Qwen3-Coder-Next-GGUF"
    service.extra_config = {
        "unit_vars": {
            "n_gpu_layers": 45,
            "ctx_size": 8192,
            "models_dir": "/home/user/models",
            "model_file": "Qwen3-Coder-Next-UD-Q2_K_XL.gguf",
        }
    }
    return service


@pytest.fixture
def mock_service_small() -> MagicMock:
    """Create a mock service with small VRAM requirement."""
    service = MagicMock()
    service.id = "embedding"
    service.driver = "fastapi"
    service.vram_mb = 1024  # 1 GB
    service.model_id = None
    service.extra_config = {}
    return service


@pytest.fixture
def mock_vram_tracker() -> MagicMock:
    """Create a mock VRAMTracker with 24GB GPU."""
    tracker = MagicMock()
    tracker.get_usage = AsyncMock(
        return_value=MagicMock(
            total_mb=24000,
            used_mb=500,  # System overhead
            free_mb=23500,
        )
    )
    tracker.get_gpu_info = AsyncMock(
        return_value=MagicMock(
            name="NVIDIA GeForce RTX 4090",
            vram_total_mb=24000,
            driver="550.67",
        )
    )
    return tracker


# ---------------------------------------------------------------------------
# VRAMCheck Tests
# ---------------------------------------------------------------------------


class TestVRAMCheck:
    """Tests for VRAMCheck preflight validation."""

    async def test_check_passes_when_vram_fits(
        self, mock_service: MagicMock, mock_vram_tracker: MagicMock
    ) -> None:
        """Check passes when service VRAM fits in available GPU memory."""
        check = VRAMCheck(vram_tracker=mock_vram_tracker)
        result = await check.check(mock_service)

        assert result.passed is True
        assert result.severity == "info"
        assert "22000" in result.message or "22 GB" in result.message.lower()

    async def test_check_fails_when_vram_exceeds(self, mock_vram_tracker: MagicMock) -> None:
        """Check fails when service VRAM exceeds available GPU memory."""
        # Service needs 26GB but GPU only has 24GB
        service = MagicMock()
        service.id = "huge-model"
        service.driver = "llamacpp"
        service.vram_mb = 26000  # Exceeds 24GB GPU
        service.model_id = "test/huge-model"
        service.extra_config = {
            "unit_vars": {
                "n_gpu_layers": 80,
                "ctx_size": 32768,
            }
        }

        check = VRAMCheck(vram_tracker=mock_vram_tracker)
        result = await check.check(service)

        assert result.passed is False
        assert result.severity == "error"
        assert "exceeds" in result.message.lower() or "insufficient" in result.message.lower()
        assert result.remediation is not None
        # Should suggest reducing n_gpu_layers
        assert "n_gpu_layers" in result.remediation.lower()

    async def test_check_passes_for_small_service(
        self, mock_service_small: MagicMock, mock_vram_tracker: MagicMock
    ) -> None:
        """Check passes for services with small VRAM requirements."""
        check = VRAMCheck(vram_tracker=mock_vram_tracker)
        result = await check.check(mock_service_small)

        assert result.passed is True
        assert result.severity == "info"

    async def test_check_skipped_for_zero_vram(self, mock_vram_tracker: MagicMock) -> None:
        """Check is skipped for services with 0 VRAM."""
        service = MagicMock()
        service.id = "no-gpu"
        service.driver = "fastapi"
        service.vram_mb = 0
        service.extra_config = {}

        check = VRAMCheck(vram_tracker=mock_vram_tracker)
        result = await check.check(service)

        assert result.passed is True
        assert result.severity == "info"
        assert "skipped" in result.message.lower()

    async def test_name_property(self, mock_vram_tracker: MagicMock) -> None:
        """Check has correct name."""
        check = VRAMCheck(vram_tracker=mock_vram_tracker)
        assert check.name == "vram"

    async def test_includes_safety_margin(self, mock_vram_tracker: MagicMock) -> None:
        """VRAM check includes safety margin (default 512 MB)."""
        # Service needs 23500 MB, GPU has 23500 MB free
        # With 512 MB safety margin, this should fail
        service = MagicMock()
        service.id = "tight-fit"
        service.driver = "llamacpp"
        service.vram_mb = 23500  # Exactly free VRAM
        service.model_id = "test/tight"
        service.extra_config = {}

        check = VRAMCheck(vram_tracker=mock_vram_tracker, safety_margin_mb=512)
        result = await check.check(service)

        # Should fail because 23500 + 512 > 23500 free
        assert result.passed is False
        assert result.severity == "error"

    async def test_custom_safety_margin(self, mock_vram_tracker: MagicMock) -> None:
        """Safety margin can be customized."""
        service = MagicMock()
        service.id = "tight-fit"
        service.driver = "llamacpp"
        service.vram_mb = 23400  # Fits with 100 MB margin
        service.model_id = "test/tight"
        service.extra_config = {}

        check = VRAMCheck(vram_tracker=mock_vram_tracker, safety_margin_mb=100)
        result = await check.check(service)

        assert result.passed is True


# ---------------------------------------------------------------------------
# VRAMSuggestion Tests
# ---------------------------------------------------------------------------


class TestVRAMSuggestion:
    """Tests for VRAM suggestion generation."""

    def test_suggest_reduced_layers(self) -> None:
        """Suggests reducing n_gpu_layers when VRAM is tight."""
        suggestion = VRAMSuggestion.for_llamacpp(
            required_mb=26000,
            available_mb=24000,
            current_layers=80,
            total_layers=80,
            ctx_size=8192,
        )

        assert suggestion is not None
        assert suggestion.suggested_layers < 80
        assert "n_gpu_layers" in suggestion.message.lower()

    def test_suggest_reduced_context(self) -> None:
        """Suggests reducing context size when layers cannot be reduced."""
        suggestion = VRAMSuggestion.for_llamacpp(
            required_mb=26000,
            available_mb=24000,
            current_layers=0,  # Already at minimum - forces context reduction
            total_layers=80,
            ctx_size=32768,  # High context
        )

        assert suggestion is not None
        # With 0 layers, should suggest reducing context
        assert "ctx_size" in suggestion.message.lower() or (
            suggestion.suggested_ctx_size is not None and suggestion.suggested_ctx_size < 32768
        )

    def test_no_suggestion_when_impossible(self) -> None:
        """Returns None when no reasonable suggestion is possible."""
        suggestion = VRAMSuggestion.for_llamacpp(
            required_mb=50000,  # Way too much
            available_mb=24000,
            current_layers=10,  # Already minimal
            total_layers=80,
            ctx_size=2048,  # Already minimal
        )

        # May return None or a message saying it won't fit
        if suggestion is not None:
            assert "won't fit" in suggestion.message.lower() or suggestion.suggested_layers == 0


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for VRAM check."""

    async def test_handles_vram_tracker_error(self) -> None:
        """Gracefully handles VRAMTracker errors."""
        tracker = MagicMock()
        tracker.get_usage = AsyncMock(side_effect=Exception("nvidia-smi failed"))

        service = MagicMock()
        service.id = "test"
        service.driver = "llamacpp"
        service.vram_mb = 8000
        service.extra_config = {}

        check = VRAMCheck(vram_tracker=tracker)
        result = await check.check(service)

        # Should return warning, not crash
        assert result.passed is False
        assert result.severity == "warning"
        assert "unable to check" in result.message.lower() or "error" in result.message.lower()

    async def test_handles_missing_unit_vars(self, mock_vram_tracker: MagicMock) -> None:
        """Handles services without unit_vars in extra_config."""
        service = MagicMock()
        service.id = "no-vars"
        service.driver = "llamacpp"
        service.vram_mb = 8000
        service.model_id = "test/model"
        service.extra_config = {}  # No unit_vars

        check = VRAMCheck(vram_tracker=mock_vram_tracker)
        result = await check.check(service)

        # Should still work, just without specific suggestions
        assert result.passed is True  # 8000 < 23500

    async def test_handles_non_llamacpp_driver(self, mock_vram_tracker: MagicMock) -> None:
        """Works correctly for non-llama.cpp drivers."""
        service = MagicMock()
        service.id = "vllm-chat"
        service.driver = "vllm"
        service.vram_mb = 15000
        service.model_id = "meta-llama/Llama-3.1-8B"
        service.extra_config = {}

        check = VRAMCheck(vram_tracker=mock_vram_tracker)
        result = await check.check(service)

        assert result.passed is True
        assert result.severity == "info"

    async def test_get_suggestions_returns_list(self, mock_vram_tracker: MagicMock) -> None:
        """get_suggestions() returns list of suggestions on failure."""
        service = MagicMock()
        service.id = "huge-model"
        service.driver = "llamacpp"
        service.vram_mb = 30000  # Way over
        service.model_id = "test/huge"
        service.extra_config = {
            "unit_vars": {
                "n_gpu_layers": 80,
                "ctx_size": 32768,
            }
        }

        check = VRAMCheck(vram_tracker=mock_vram_tracker)
        result = await check.check(service)

        assert result.passed is False
        # Should have stored suggestions
        suggestions = check.get_suggestions()
        assert suggestions is not None


# ---------------------------------------------------------------------------
# gpumod-lgt: MTP-aware VRAM checking
# ---------------------------------------------------------------------------


def _make_mtp_service(vram_mb: int = 22000) -> MagicMock:
    """Service with MTP flags in extra_args (triggers MTP overhead path)."""
    service = MagicMock()
    service.id = "qwen36-35b-a3b-mtp-iq4xs"
    service.driver = "llamacpp"
    service.vram_mb = vram_mb
    service.model_id = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
    service.extra_config = {
        "unit_vars": {
            "n_gpu_layers": -1,
            "ctx_size": 32768,
            "extra_args": (
                "--parallel 1 --threads 16 "
                "--spec-type draft-mtp --spec-draft-n-max 2 "
                "--chat-template-kwargs '{\"enable_thinking\":true}'"
            ),
        }
    }
    return service


class TestVRAMCheckMTPOverhead:
    """gpumod-lgt: MTP variants load ~1.5 GB more than declared vram_mb due
    to the draft head + draft KV cache. The preflight check must account
    for it so the service doesn't OOM mid-load on a 24 GB GPU.
    """

    async def test_non_mtp_service_unaffected(
        self, mock_service: MagicMock, mock_vram_tracker: MagicMock
    ) -> None:
        # mock_service has no --spec-type in extra_args -> no MTP overhead
        check = VRAMCheck(vram_tracker=mock_vram_tracker, mtp_overhead_mb=1500)
        result = await check.check(mock_service)

        assert result.passed is True
        # Message should NOT mention MTP overhead for non-MTP services
        assert "mtp" not in result.message.lower()

    async def test_mtp_overhead_applied(self, mock_vram_tracker: MagicMock) -> None:
        # Service declares 22000 MB; with default 1500 MB MTP overhead,
        # required = 22000 + 1500 + 512 (safety) = 24012 MB.
        # GPU has 23500 free -> should FAIL with MTP overhead applied.
        service = _make_mtp_service(vram_mb=22000)
        check = VRAMCheck(vram_tracker=mock_vram_tracker, mtp_overhead_mb=1500)
        result = await check.check(service)

        assert result.passed is False
        # Failure message MUST reference MTP overhead explicitly
        assert "mtp" in result.message.lower() or "mtp" in (result.remediation or "").lower()

    async def test_mtp_overhead_disabled_with_zero(self, mock_vram_tracker: MagicMock) -> None:
        # Override overhead to 0 -> MTP detection has no behavior effect
        service = _make_mtp_service(vram_mb=22000)
        check = VRAMCheck(vram_tracker=mock_vram_tracker, mtp_overhead_mb=0)
        result = await check.check(service)

        # 22000 + 0 + 512 = 22512 < 23500 free -> passes
        assert result.passed is True

    async def test_mtp_overhead_configurable_higher_value(
        self, mock_vram_tracker: MagicMock
    ) -> None:
        # Aggressive 3000 MB overhead refuses tighter scenarios that 1500
        # would allow.
        # vram_mb=20000 + 1500 overhead + 512 = 22012 -> passes (< 23500)
        # vram_mb=20000 + 3000 overhead + 512 = 23512 -> fails (> 23500)
        service = _make_mtp_service(vram_mb=20000)

        default = VRAMCheck(vram_tracker=mock_vram_tracker, mtp_overhead_mb=1500)
        result_default = await default.check(service)
        assert result_default.passed is True

        aggressive = VRAMCheck(vram_tracker=mock_vram_tracker, mtp_overhead_mb=3000)
        result_aggressive = await aggressive.check(service)
        assert result_aggressive.passed is False

    async def test_mtp_detection_via_extra_args_string(self, mock_vram_tracker: MagicMock) -> None:
        # MTP detection should match the exact `--spec-type draft-mtp` token.
        # A random use of "spec" should NOT trigger overhead.
        service = MagicMock()
        service.id = "false-positive-test"
        service.driver = "llamacpp"
        service.vram_mb = 22000
        service.model_id = "test/model"
        service.extra_config = {
            "unit_vars": {
                "extra_args": "--threads 16 --no-prefetch --batch-size 512",
            }
        }
        check = VRAMCheck(vram_tracker=mock_vram_tracker, mtp_overhead_mb=1500)
        result = await check.check(service)

        # No MTP -> 22000 + 512 = 22512 < 23500 -> passes
        assert result.passed is True
        assert "mtp" not in result.message.lower()
