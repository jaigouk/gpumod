"""Tests for preflight validation system.

Follows TDD: tests written first (Red), then implementation (Green).

Phase 1: Core Infrastructure + TokenizerCheck
- PreflightCheck protocol
- CheckResult dataclass
- TokenizerCheck implementation
- PreflightRunner integration
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpumod.models import DriverType, Service, SleepMode


def _make_service(
    id: str = "test-svc",
    model_id: str = "org/model",
    driver: DriverType = DriverType.VLLM,
) -> Service:
    """Create a test service."""
    return Service(
        id=id,
        name="Test Service",
        driver=driver,
        port=8000,
        vram_mb=5000,
        sleep_mode=SleepMode.NONE,
        health_endpoint="/health",
        model_id=model_id,
        unit_name=f"{id}.service",
        depends_on=[],
        startup_timeout=60,
        extra_config={},
    )


# ---------------------------------------------------------------------------
# CheckResult dataclass tests
# ---------------------------------------------------------------------------


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_creates_passing_result(self) -> None:
        """CheckResult captures passed=True with severity."""
        from gpumod.preflight import CheckResult

        result = CheckResult(
            passed=True,
            severity="info",
            message="Tokenizer loaded successfully",
        )

        assert result.passed is True
        assert result.severity == "info"
        assert result.message == "Tokenizer loaded successfully"
        assert result.remediation is None

    def test_creates_failing_result_with_remediation(self) -> None:
        """CheckResult can include remediation steps for failures."""
        from gpumod.preflight import CheckResult

        result = CheckResult(
            passed=False,
            severity="error",
            message="Tokenizer missing all_special_ids attribute",
            remediation="Use a different model or update transformers library",
        )

        assert result.passed is False
        assert result.severity == "error"
        assert "all_special_ids" in result.message
        assert result.remediation is not None

    def test_creates_warning_result(self) -> None:
        """CheckResult supports warning severity."""
        from gpumod.preflight import CheckResult

        result = CheckResult(
            passed=True,
            severity="warning",
            message="VRAM usage will be at 95%",
        )

        assert result.passed is True
        assert result.severity == "warning"


# ---------------------------------------------------------------------------
# TokenizerCheck tests
# ---------------------------------------------------------------------------


class TestTokenizerCheck:
    """Tests for TokenizerCheck implementation."""

    def test_has_name_attribute(self) -> None:
        """TokenizerCheck should have a name attribute."""
        from gpumod.preflight import TokenizerCheck

        check = TokenizerCheck()
        assert check.name == "tokenizer"

    @pytest.mark.asyncio
    async def test_passes_for_valid_tokenizer(self) -> None:
        """TokenizerCheck passes when tokenizer has required attributes."""
        from gpumod.preflight import TokenizerCheck

        service = _make_service(model_id="mistralai/Mistral-7B-v0.1")

        # Mock a valid tokenizer with all required attributes
        mock_tokenizer = MagicMock()
        mock_tokenizer.all_special_ids = [0, 1, 2]
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.pad_token_id = 0

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            check = TokenizerCheck()
            result = await check.check(service)

        assert result.passed is True
        assert result.severity == "info"

    @pytest.mark.asyncio
    async def test_fails_for_missing_all_special_ids(self) -> None:
        """TokenizerCheck fails when tokenizer lacks all_special_ids."""
        from gpumod.preflight import TokenizerCheck

        service = _make_service(model_id="mistralai/devstral-small-2")

        # Mock a broken tokenizer missing all_special_ids
        mock_tokenizer = MagicMock(spec=[])  # No attributes
        del mock_tokenizer.all_special_ids  # Ensure AttributeError

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            check = TokenizerCheck()
            result = await check.check(service)

        assert result.passed is False
        assert result.severity == "error"
        assert "all_special_ids" in result.message

    @pytest.mark.asyncio
    async def test_fails_for_missing_eos_token_id(self) -> None:
        """TokenizerCheck fails when tokenizer lacks eos_token_id."""
        from gpumod.preflight import TokenizerCheck

        service = _make_service(model_id="broken/model")

        mock_tokenizer = MagicMock()
        mock_tokenizer.all_special_ids = [0, 1]
        del mock_tokenizer.eos_token_id  # Missing

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            check = TokenizerCheck()
            result = await check.check(service)

        assert result.passed is False
        assert "eos_token_id" in result.message

    @pytest.mark.asyncio
    async def test_handles_tokenizer_load_failure(self) -> None:
        """TokenizerCheck handles exceptions during tokenizer load."""
        from gpumod.preflight import TokenizerCheck

        service = _make_service(model_id="nonexistent/model")

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = OSError("Model not found")

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            check = TokenizerCheck()
            result = await check.check(service)

        assert result.passed is False
        assert result.severity == "error"
        assert "Model not found" in result.message or "load" in result.message.lower()

    @pytest.mark.asyncio
    async def test_skips_for_non_vllm_driver(self) -> None:
        """TokenizerCheck skips for llama.cpp (uses GGUF, not HF tokenizer)."""
        from gpumod.preflight import TokenizerCheck

        service = _make_service(
            model_id="local/model.gguf",
            driver=DriverType.LLAMACPP,
        )

        check = TokenizerCheck()
        result = await check.check(service)

        # Should pass/skip for non-vLLM drivers
        assert result.passed is True
        assert "skipped" in result.message.lower() or result.severity == "info"

    @pytest.mark.asyncio
    async def test_skips_when_no_model_id(self) -> None:
        """TokenizerCheck skips when service has no model_id."""
        from gpumod.preflight import TokenizerCheck

        service = _make_service(model_id=None)

        check = TokenizerCheck()
        result = await check.check(service)

        assert result.passed is True
        assert "no model" in result.message.lower() or "skipped" in result.message.lower()


# ---------------------------------------------------------------------------
# PreflightRunner tests
# ---------------------------------------------------------------------------


class TestPreflightRunner:
    """Tests for PreflightRunner that orchestrates all checks."""

    @pytest.mark.asyncio
    async def test_runs_all_registered_checks(self) -> None:
        """Runner executes all registered preflight checks."""
        from gpumod.preflight import CheckResult, PreflightRunner

        service = _make_service()

        # Create mock checks
        check1 = AsyncMock()
        check1.name = "check1"
        check1.check.return_value = CheckResult(passed=True, severity="info", message="OK")

        check2 = AsyncMock()
        check2.name = "check2"
        check2.check.return_value = CheckResult(passed=True, severity="info", message="OK")

        runner = PreflightRunner(checks=[check1, check2])
        results = await runner.run_all(service)

        assert len(results) == 2
        check1.check.assert_called_once_with(service)
        check2.check.assert_called_once_with(service)

    @pytest.mark.asyncio
    async def test_collects_all_results(self) -> None:
        """Runner returns results from all checks."""
        from gpumod.preflight import CheckResult, PreflightRunner

        service = _make_service()

        check1 = AsyncMock()
        check1.name = "tokenizer"
        check1.check.return_value = CheckResult(
            passed=True, severity="info", message="Tokenizer OK"
        )

        check2 = AsyncMock()
        check2.name = "vram"
        check2.check.return_value = CheckResult(
            passed=False, severity="error", message="VRAM exceeded"
        )

        runner = PreflightRunner(checks=[check1, check2])
        results = await runner.run_all(service)

        assert results["tokenizer"].passed is True
        assert results["vram"].passed is False

    @pytest.mark.asyncio
    async def test_has_errors_returns_true_when_error_severity(self) -> None:
        """has_errors returns True when any check has error severity."""
        from gpumod.preflight import CheckResult, PreflightRunner

        service = _make_service()

        check1 = AsyncMock()
        check1.name = "check1"
        check1.check.return_value = CheckResult(passed=True, severity="info", message="OK")

        check2 = AsyncMock()
        check2.name = "check2"
        check2.check.return_value = CheckResult(passed=False, severity="error", message="Failed")

        runner = PreflightRunner(checks=[check1, check2])
        results = await runner.run_all(service)

        assert runner.has_errors(results) is True

    @pytest.mark.asyncio
    async def test_has_errors_returns_false_for_warnings_only(self) -> None:
        """has_errors returns False when only warnings (no errors)."""
        from gpumod.preflight import CheckResult, PreflightRunner

        service = _make_service()

        check = AsyncMock()
        check.name = "vram"
        check.check.return_value = CheckResult(
            passed=True, severity="warning", message="VRAM at 90%"
        )

        runner = PreflightRunner(checks=[check])
        results = await runner.run_all(service)

        assert runner.has_errors(results) is False

    @pytest.mark.asyncio
    async def test_format_errors_returns_readable_message(self) -> None:
        """format_errors produces human-readable error summary."""
        from gpumod.preflight import CheckResult, PreflightRunner

        service = _make_service()

        check = AsyncMock()
        check.name = "tokenizer"
        check.check.return_value = CheckResult(
            passed=False,
            severity="error",
            message="Missing all_special_ids",
            remediation="Update transformers or use different model",
        )

        runner = PreflightRunner(checks=[check])
        results = await runner.run_all(service)

        error_msg = runner.format_errors(results)

        assert "tokenizer" in error_msg.lower()
        assert "all_special_ids" in error_msg
        assert "Update transformers" in error_msg or "remediation" in error_msg.lower()


# ---------------------------------------------------------------------------
# Integration with LifecycleManager (conceptual test)
# ---------------------------------------------------------------------------


class TestPreflightIntegration:
    """Tests for preflight integration into service lifecycle."""

    @pytest.mark.asyncio
    async def test_default_runner_includes_tokenizer_check(self) -> None:
        """Default PreflightRunner includes TokenizerCheck."""
        from gpumod.preflight import PreflightRunner

        runner = PreflightRunner.default()

        check_names = [c.name for c in runner.checks]
        assert "tokenizer" in check_names

    @pytest.mark.asyncio
    async def test_run_preflight_helper_function(self) -> None:
        """run_preflight convenience function works correctly."""
        from gpumod.preflight import run_preflight

        service = _make_service(model_id="test/model")

        # Mock the tokenizer to pass
        mock_tokenizer = MagicMock()
        mock_tokenizer.all_special_ids = [0, 1, 2]
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.pad_token_id = 0

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            results, has_errors = await run_preflight(service)

        assert isinstance(results, dict)
        assert isinstance(has_errors, bool)
