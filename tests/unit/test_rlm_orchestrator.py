"""Tests for RLMOrchestrator — manages RLM lifecycle for consult queries.

TDD: RED phase - these tests define the RLMOrchestrator interface.
Tests cover consultation flow, max turns, validation, and error handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from gpumod.rlm.orchestrator import ConsultResult, RLMOrchestrator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tool_wrappers():
    """Mock tool wrappers for RLM environment."""
    return {
        "gpu_status": AsyncMock(return_value={"mode": "dev", "vram_free_mb": 20000}),
        "list_gguf_files": AsyncMock(return_value={"files": [], "count": 0}),
        "fetch_model_config": AsyncMock(return_value={"architectures": ["LlamaForCausalLM"]}),
        "fetch_driver_docs": AsyncMock(return_value={"content": "# docs"}),
        "search_hf_models": AsyncMock(return_value={"models": [], "count": 0}),
        "simulate_mode": AsyncMock(return_value={"fits": True}),
        "generate_preset": AsyncMock(return_value={"preset": "yaml..."}),
    }


# ---------------------------------------------------------------------------
# TestConsultResult - Pydantic model
# ---------------------------------------------------------------------------


class TestConsultResult:
    """Tests for ConsultResult Pydantic model."""

    def test_consult_result_fields(self):
        """ConsultResult has all expected fields."""
        result = ConsultResult(
            recommendation="Use IQ2_XXS with --n-cpu-moe 24",
            reasoning_steps=["Step 1", "Step 2"],
            suggested_commands=["generate_preset(...)"],
            sources=["huggingface.co/model/config.json"],
            turns_used=3,
        )

        assert result.recommendation == "Use IQ2_XXS with --n-cpu-moe 24"
        assert len(result.reasoning_steps) == 2
        assert len(result.suggested_commands) == 1
        assert len(result.sources) == 1
        assert result.turns_used == 3

    def test_consult_result_defaults(self):
        """ConsultResult has sensible defaults."""
        result = ConsultResult()

        assert result.recommendation == ""
        assert result.reasoning_steps == []
        assert result.suggested_commands == []
        assert result.sources == []
        assert result.turns_used == 0
        assert result.can_run is None
        assert result.incomplete is False

    def test_consult_result_serializable(self):
        """ConsultResult can be serialized to dict for MCP response."""
        result = ConsultResult(
            recommendation="Use Q4_K_M",
            reasoning_steps=["Step 1"],
            suggested_commands=["cmd1"],
            sources=["source1"],
            turns_used=1,
        )

        result_dict = result.model_dump()
        assert "recommendation" in result_dict
        assert "turns_used" in result_dict
        assert "can_run" in result_dict
        assert "incomplete" in result_dict

    def test_consult_result_can_run_field(self):
        """can_run field is Optional[bool]."""
        result_yes = ConsultResult(can_run=True, recommendation="Yes")
        result_no = ConsultResult(can_run=False, recommendation="No")
        result_unknown = ConsultResult(recommendation="Unknown")

        assert result_yes.can_run is True
        assert result_no.can_run is False
        assert result_unknown.can_run is None


# ---------------------------------------------------------------------------
# TestRLMOrchestrator - consult lifecycle
# ---------------------------------------------------------------------------


class TestRLMOrchestrator:
    """Tests for RLMOrchestrator class."""

    def test_consult_returns_result(self, mock_tool_wrappers):
        """consult() returns a ConsultResult structure."""
        orchestrator = RLMOrchestrator(tool_wrappers=mock_tool_wrappers)

        # Mock _run_loop to avoid needing real LLM client
        with patch.object(orchestrator, "_run_loop") as mock_loop:
            mock_loop.return_value = ConsultResult(
                recommendation="Use Q4_K_M quantization",
                reasoning_steps=["Checked model config", "Checked VRAM"],
                suggested_commands=[],
                sources=[],
                turns_used=2,
            )

            # Also mock get_client to avoid LLM init
            with (
                patch("gpumod.rlm.orchestrator.get_client"),
                patch("gpumod.rlm.orchestrator.LMHandler"),
            ):
                result = orchestrator.consult("Can I run Llama 3.1 70B on 24GB?")

            assert isinstance(result, ConsultResult)
            assert "Q4_K_M" in result.recommendation
            assert result.turns_used <= 5

    def test_max_turns_capped_at_hard_limit(self, mock_tool_wrappers):
        """max_turns is capped at MAX_ITERATIONS_HARD_LIMIT (10)."""
        from gpumod.rlm.orchestrator import MAX_ITERATIONS_HARD_LIMIT

        orchestrator = RLMOrchestrator(tool_wrappers=mock_tool_wrappers)

        with patch.object(orchestrator, "_run_loop") as mock_loop:
            mock_loop.return_value = ConsultResult(turns_used=10)

            with (
                patch("gpumod.rlm.orchestrator.get_client"),
                patch("gpumod.rlm.orchestrator.LMHandler"),
            ):
                orchestrator.consult("test", max_turns=50)

            # _run_loop should be called with capped max_turns
            call_kwargs = mock_loop.call_args
            actual_max = call_kwargs[0][3]  # 4th positional arg
            assert actual_max == MAX_ITERATIONS_HARD_LIMIT

    def test_default_max_turns_is_five(self):
        """Default max turns constant is 5."""
        from gpumod.rlm.orchestrator import DEFAULT_MAX_ITERATIONS

        assert DEFAULT_MAX_ITERATIONS == 5

    def test_orchestrator_creates_environment(self, mock_tool_wrappers):
        """consult() creates GpumodConsultEnv with tool wrappers."""
        orchestrator = RLMOrchestrator(tool_wrappers=mock_tool_wrappers)

        with (
            patch("gpumod.rlm.orchestrator.GpumodConsultEnv") as env_cls,
            patch("gpumod.rlm.orchestrator.get_client"),
            patch("gpumod.rlm.orchestrator.LMHandler"),
            patch.object(orchestrator, "_run_loop", return_value=ConsultResult()),
        ):
            mock_env = env_cls.return_value
            mock_env.cleanup = lambda: None

            orchestrator.consult("test query")

            env_cls.assert_called_once()
            call_kwargs = env_cls.call_args.kwargs
            assert "tool_wrappers" in call_kwargs
            assert "context_payload" in call_kwargs

    def test_orchestrator_cleans_up_env(self, mock_tool_wrappers):
        """consult() calls env.cleanup() even on error."""
        orchestrator = RLMOrchestrator(tool_wrappers=mock_tool_wrappers)

        with (
            patch("gpumod.rlm.orchestrator.GpumodConsultEnv") as env_cls,
            patch("gpumod.rlm.orchestrator.get_client"),
            patch("gpumod.rlm.orchestrator.LMHandler") as handler_cls,
        ):
            mock_env = env_cls.return_value
            mock_handler = handler_cls.return_value

            with (
                patch.object(orchestrator, "_run_loop", side_effect=RuntimeError("boom")),
                pytest.raises(RuntimeError, match="boom"),
            ):
                orchestrator.consult("test query")

            mock_env.cleanup.assert_called_once()
            mock_handler.stop.assert_called_once()

    def test_resolve_model_default(self, mock_tool_wrappers):
        """Default model is claude-sonnet for anthropic backend."""
        orchestrator = RLMOrchestrator(
            tool_wrappers=mock_tool_wrappers,
            backend="anthropic",
        )
        assert "claude" in orchestrator._model or "sonnet" in orchestrator._model

    def test_resolve_model_override(self, mock_tool_wrappers):
        """Explicit model param overrides default."""
        orchestrator = RLMOrchestrator(
            tool_wrappers=mock_tool_wrappers,
            model="custom-model-v1",
        )
        assert orchestrator._model == "custom-model-v1"

    def test_context_passed_to_environment(self, mock_tool_wrappers):
        """Context dict is forwarded to GpumodConsultEnv payload."""
        orchestrator = RLMOrchestrator(tool_wrappers=mock_tool_wrappers)
        user_context = {"vram_mb": 24000, "current_mode": "dev"}

        with (
            patch("gpumod.rlm.orchestrator.GpumodConsultEnv") as env_cls,
            patch("gpumod.rlm.orchestrator.get_client"),
            patch("gpumod.rlm.orchestrator.LMHandler"),
            patch.object(orchestrator, "_run_loop", return_value=ConsultResult()),
        ):
            mock_env = env_cls.return_value
            mock_env.cleanup = lambda: None

            orchestrator.consult("test query", context=user_context)

            call_kwargs = env_cls.call_args.kwargs
            payload = call_kwargs["context_payload"]
            assert payload["query"] == "test query"
            assert payload["data"] == user_context


# ---------------------------------------------------------------------------
# TestParseFinalAnswer - ConsultResult parsing from FINAL(...)
# ---------------------------------------------------------------------------


class TestParseFinalAnswer:
    """Tests for _parse_final_answer which converts FINAL(...) payloads."""

    def test_parse_valid_json(self):
        """Well-formed JSON maps to ConsultResult fields."""
        from gpumod.rlm.orchestrator import _parse_final_answer

        raw = (
            '{"can_run": true, "recommendation": "Use Q4_K_M",'
            ' "reasoning_steps": ["step1"], "sources": ["hf.co"]}'
        )
        result = _parse_final_answer(raw)

        assert result.can_run is True
        assert result.recommendation == "Use Q4_K_M"
        assert result.reasoning_steps == ["step1"]
        assert result.sources == ["hf.co"]

    def test_parse_json_with_false_can_run(self):
        """can_run: false is preserved, not treated as None."""
        from gpumod.rlm.orchestrator import _parse_final_answer

        raw = '{"can_run": false, "recommendation": "Too large"}'
        result = _parse_final_answer(raw)

        assert result.can_run is False
        assert result.recommendation == "Too large"

    def test_parse_json_with_null_can_run(self):
        """can_run: null maps to None."""
        from gpumod.rlm.orchestrator import _parse_final_answer

        raw = '{"can_run": null, "recommendation": "Uncertain"}'
        result = _parse_final_answer(raw)

        assert result.can_run is None

    def test_parse_embedded_json_in_prose(self):
        """JSON embedded in prose text is extracted."""
        from gpumod.rlm.orchestrator import _parse_final_answer

        raw = 'Based on analysis: {"can_run": true, "recommendation": "Yes, use Q4_K_M"}'
        result = _parse_final_answer(raw)

        assert result.can_run is True
        assert "Q4_K_M" in result.recommendation

    def test_parse_plain_text_fallback(self):
        """Plain text (no JSON) becomes recommendation."""
        from gpumod.rlm.orchestrator import _parse_final_answer

        raw = "You should use Q4_K_M quantization for this model."
        result = _parse_final_answer(raw)

        assert result.recommendation == raw
        assert result.can_run is None

    def test_parse_empty_string(self):
        """Empty string produces empty recommendation."""
        from gpumod.rlm.orchestrator import _parse_final_answer

        result = _parse_final_answer("")
        assert result.recommendation == ""

    def test_parse_json_missing_fields(self):
        """JSON with only some fields still works."""
        from gpumod.rlm.orchestrator import _parse_final_answer

        raw = '{"recommendation": "Try smaller quant"}'
        result = _parse_final_answer(raw)

        assert result.recommendation == "Try smaller quant"
        assert result.can_run is None
        assert result.reasoning_steps == []

    def test_parse_suggested_commands(self):
        """suggested_commands field is preserved."""
        from gpumod.rlm.orchestrator import _parse_final_answer

        raw = (
            '{"recommendation": "Use preset",'
            ' "suggested_commands": ["generate_preset(repo_id=\\"test\\")"]}'
        )
        result = _parse_final_answer(raw)

        assert len(result.suggested_commands) == 1
        assert "generate_preset" in result.suggested_commands[0]
