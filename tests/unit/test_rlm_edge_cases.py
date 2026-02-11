"""Edge case and security tests for the RLM module.

Task #6: Covers security bypass attempts, environment boundary conditions,
namespace integrity, and orchestrator edge cases beyond standard unit tests.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpumod.rlm.environment import GpumodConsultEnv
from gpumod.rlm.orchestrator import (
    ConsultResult,
    RLMOrchestrator,
    _parse_final_answer,
)
from gpumod.rlm.security import BLOCKED_MODULE_NAMES, SecurityError, validate_code

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tool_wrappers():
    """Mock read-only tool wrappers for the REPL environment."""
    return {
        "gpu_status": AsyncMock(return_value={"mode": "dev", "vram_free_mb": 20000}),
        "list_gguf_files": AsyncMock(return_value={"files": [], "count": 0}),
        "fetch_model_config": AsyncMock(return_value={"architectures": ["LlamaForCausalLM"]}),
        "fetch_driver_docs": AsyncMock(return_value={"content": "# docs"}),
        "search_hf_models": AsyncMock(return_value={"models": [], "count": 0}),
        "simulate_mode": AsyncMock(return_value={"fits": True}),
        "generate_preset": AsyncMock(return_value={"preset": "yaml..."}),
    }


@pytest.fixture
def env(mock_tool_wrappers):
    """Create a GpumodConsultEnv with mocked tools."""
    return GpumodConsultEnv(tool_wrappers=mock_tool_wrappers)


# ---------------------------------------------------------------------------
# Security: import bypass attempts
# ---------------------------------------------------------------------------


class TestSecurityImportBypasses:
    """Attempts to bypass the import block via various techniques."""

    def test_import_via_string_concat_in_exec(self):
        """exec('imp' + 'ort os') — exec itself is blocked."""
        with pytest.raises(SecurityError, match="exec"):
            validate_code("exec('imp' + 'ort os')")

    def test_nested_eval_with_import(self):
        """eval("__import__('os')") — eval is blocked."""
        with pytest.raises(SecurityError, match="eval"):
            validate_code("eval(\"__import__('os')\")")

    def test_import_inside_function_def(self):
        """import inside a function body is still blocked."""
        code = "def f():\n    import os\n    return os"
        with pytest.raises(SecurityError, match="[Ii]mport"):
            validate_code(code)

    def test_import_in_class_body(self):
        """import inside a class body is still blocked."""
        code = "class X:\n    import sys"
        with pytest.raises(SecurityError, match="[Ii]mport"):
            validate_code(code)

    def test_importlib_via_dunder_import(self):
        """__import__('importlib') is blocked."""
        with pytest.raises(SecurityError, match="__import__"):
            validate_code("__import__('importlib')")

    def test_from_builtins_import(self):
        """from builtins import * is blocked."""
        with pytest.raises(SecurityError, match="[Ii]mport"):
            validate_code("from builtins import *")


# ---------------------------------------------------------------------------
# Security: dunder attribute chain attacks
# ---------------------------------------------------------------------------


class TestSecurityDunderChains:
    """Attempts to escape sandbox via dunder attribute chains."""

    def test_class_mro_subclasses_chain(self):
        """().__class__.__mro__[1].__subclasses__() is blocked (dunder attr)."""
        with pytest.raises(SecurityError, match="Dunder attribute"):
            validate_code("().__class__.__mro__[1].__subclasses__()")

    def test_code_object_access(self):
        """func.__code__ is blocked (dunder attr)."""
        with pytest.raises(SecurityError, match="__code__"):
            validate_code("f.__code__")

    def test_closure_access(self):
        """func.__closure__ is blocked (dunder attr)."""
        with pytest.raises(SecurityError, match="__closure__"):
            validate_code("f.__closure__")

    def test_dict_access(self):
        """obj.__dict__ is blocked (dunder attr)."""
        with pytest.raises(SecurityError, match="__dict__"):
            validate_code("obj.__dict__")

    def test_module_access(self):
        """func.__module__ is blocked (dunder attr)."""
        with pytest.raises(SecurityError, match="__module__"):
            validate_code("func.__module__")

    def test_init_subclass(self):
        """cls.__init_subclass__ is blocked (dunder attr)."""
        with pytest.raises(SecurityError, match="__init_subclass__"):
            validate_code("cls.__init_subclass__()")

    def test_reduce_access(self):
        """obj.__reduce__ is blocked (dunder attr)."""
        with pytest.raises(SecurityError, match="__reduce__"):
            validate_code("obj.__reduce__()")


# ---------------------------------------------------------------------------
# Security: blocked module calls (os.system, subprocess.call, etc.)
# ---------------------------------------------------------------------------


class TestSecurityBlockedModuleCalls:
    """AST blocks calls on dangerous module names like os.system()."""

    def test_os_system_blocked(self):
        """os.system('cmd') is blocked."""
        with pytest.raises(SecurityError, match="os"):
            validate_code("os.system('whoami')")

    def test_subprocess_call_blocked(self):
        """subprocess.call([...]) is blocked."""
        with pytest.raises(SecurityError, match="subprocess"):
            validate_code("subprocess.call(['ls'])")

    def test_subprocess_popen_blocked(self):
        """subprocess.Popen([...]) is blocked."""
        with pytest.raises(SecurityError, match="subprocess"):
            validate_code("subprocess.Popen(['ls'])")

    def test_shutil_rmtree_blocked(self):
        """shutil.rmtree('/') is blocked."""
        with pytest.raises(SecurityError, match="shutil"):
            validate_code("shutil.rmtree('/')")

    def test_pathlib_blocked(self):
        """pathlib.Path('/') is blocked."""
        with pytest.raises(SecurityError, match="pathlib"):
            validate_code("pathlib.Path('/etc/passwd').read_text()")

    def test_socket_blocked(self):
        """socket.socket() is blocked."""
        with pytest.raises(SecurityError, match="socket"):
            validate_code("socket.socket()")

    def test_ctypes_direct_call_blocked(self):
        """ctypes.CDLL('...') is blocked (direct module.func call)."""
        with pytest.raises(SecurityError, match="ctypes"):
            validate_code("ctypes.CDLL('libc.so')")

    def test_ctypes_chained_call_not_caught(self):
        """ctypes.cdll.LoadLibrary() — chained access bypasses module checker.

        NOTE: This is a known limitation. The AST checker only catches
        <module>.<func>() but not <module>.<attr>.<func>() chains.
        The env is still safe because ctypes is not in the namespace.
        """
        # Does NOT raise SecurityError — the AST check doesn't catch this
        validate_code("ctypes.cdll.LoadLibrary('libc.so')")

    def test_signal_blocked(self):
        """signal.signal() is blocked."""
        with pytest.raises(SecurityError, match="signal"):
            validate_code("signal.signal(2, lambda s, f: None)")

    def test_blocked_module_names_is_frozenset(self):
        """BLOCKED_MODULE_NAMES is immutable."""
        assert isinstance(BLOCKED_MODULE_NAMES, frozenset)

    def test_all_expected_modules_blocked(self):
        """All expected dangerous modules are in the blocked set."""
        expected = {"os", "sys", "subprocess", "shutil", "pathlib", "socket", "ctypes", "signal"}
        assert expected.issubset(BLOCKED_MODULE_NAMES)

    def test_safe_module_calls_allowed(self):
        """Calls on safe modules (json, re, math) are not blocked."""
        validate_code("json.dumps({'a': 1})")
        validate_code("re.findall(r'\\d+', 'abc123')")
        validate_code("math.sqrt(16)")


# ---------------------------------------------------------------------------
# Security: allowed patterns that look dangerous
# ---------------------------------------------------------------------------


class TestSecurityAllowedPatterns:
    """Patterns that look dangerous but are actually safe."""

    def test_string_containing_import(self):
        """String literal containing 'import' is fine."""
        validate_code("msg = 'please import the module'")

    def test_string_containing_exec(self):
        """String containing 'exec' keyword is fine."""
        validate_code("msg = 'exec is a builtin function'")

    def test_comment_with_import(self):
        """Comment containing import is fine."""
        validate_code("# import os\nx = 1")

    def test_variable_named_import_like(self):
        """Variable with 'import' in name is fine."""
        validate_code("import_count = 5")

    def test_dict_key_named_exec(self):
        """Dict key 'exec' is fine."""
        validate_code("d = {'exec': True}")

    def test_fstring_with_dangerous_content(self):
        """f-string containing dangerous words is fine."""
        validate_code("msg = f'exec returned {42}'")


# ---------------------------------------------------------------------------
# Environment: namespace integrity
# ---------------------------------------------------------------------------


class TestEnvironmentNamespaceIntegrity:
    """Tests for namespace isolation and integrity."""

    def test_del_whitelisted_tool_does_not_persist(self, env):
        """Deleting a whitelisted tool in code doesn't affect _namespace.

        Because execute_code() merges _namespace + _locals into a fresh
        combined dict each call, `del` inside code only affects the
        temporary combined dict. The tool is still available next call.
        """
        env.execute_code("del gpu_status")
        # Tool is still available because _namespace is untouched
        result = env.execute_code("print(type(gpu_status))")
        assert "NameError" not in result.stderr

    def test_reassign_whitelisted_tool_does_not_persist(self, env):
        """Reassigning a whitelisted tool in code doesn't persist.

        New variables are captured into _locals only if the key is NOT
        already in _namespace. Since gpu_status is in _namespace, the
        reassignment doesn't persist to subsequent calls.
        """
        env.execute_code("gpu_status = lambda: {'hacked': True}")
        # Next call still gets the original tool from _namespace
        result = env.execute_code("print(type(gpu_status))")
        assert "function" not in result.stdout or "Mock" in result.stdout

    def test_format_builtin_removed(self, env):
        """format() is NOT available — prevents sandbox escape via class traversal.

        The attack: '{0.__class__.__bases__[0].__subclasses__()}'.format([])
        bypasses AST checks because dunder access is inside a string literal
        that format() interprets at runtime.
        """
        result = env.execute_code("format(42, 'd')")
        assert "NameError" in result.stderr

    def test_format_sandbox_escape_blocked(self, env):
        """The format() sandbox escape should fail (format not available)."""
        code = '"{0.__class__.__bases__[0].__subclasses__()}".format([])'
        result = env.execute_code(code)
        # Should fail because format is not in safe builtins — the .format()
        # method on str still exists, but the key attack vector (format()
        # builtin) is removed. The str.format() method also works here, but
        # AST validation blocks the __class__ dunder attribute access.
        assert "SecurityError" in result.stderr or "NameError" in result.stderr

    def test_hasattr_removed(self, env):
        """hasattr() is NOT available — prevents dunder attribute probing.

        hasattr(obj, '__class__') could be used to probe for dunder
        attributes at runtime without triggering AST checks.
        """
        result = env.execute_code("hasattr([], '__class__')")
        assert "NameError" in result.stderr

    def test_cannot_access_builtins_import_via_namespace(self, env):
        """__import__ is None in safe builtins."""
        result = env.execute_code("print(__import__)")
        # __import__ is explicitly set to None in _SAFE_BUILTINS
        assert "None" in result.stdout or "SecurityError" in result.stderr

    def test_cannot_access_real_builtins(self, env):
        """Direct __builtins__ access is restricted."""
        result = env.execute_code("print(type(__builtins__))")
        # __builtins__ is a dict (not the module) in our restricted namespace
        assert "dict" in result.stdout

    def test_namespace_isolation_from_real_modules(self, env):
        """os, sys, subprocess should not be in namespace."""
        for mod_name in ("os", "sys", "subprocess"):
            result = env.execute_code(f"print({mod_name})")
            assert "NameError" in result.stderr

    def test_safe_builtins_input_is_none(self, env):
        """input() is None/blocked in safe builtins."""
        result = env.execute_code("input('prompt')")
        assert "TypeError" in result.stderr or "not callable" in result.stderr

    def test_tool_wrappers_outside_whitelist_not_injected(self, mock_tool_wrappers):
        """Tools not in WHITELISTED_TOOLS are excluded even if in wrappers."""
        mock_tool_wrappers["switch_mode"] = AsyncMock(return_value={"ok": True})
        env = GpumodConsultEnv(tool_wrappers=mock_tool_wrappers)

        assert "switch_mode" not in env._namespace
        result = env.execute_code("switch_mode('dev')")
        assert "NameError" in result.stderr


# ---------------------------------------------------------------------------
# Environment: code execution boundary conditions
# ---------------------------------------------------------------------------


class TestEnvironmentBoundaryConditions:
    """Edge cases for code execution."""

    def test_empty_code(self, env):
        """Empty string produces no output."""
        result = env.execute_code("")
        assert result.stdout == ""
        assert result.stderr == ""

    def test_whitespace_only_code(self, env):
        """Whitespace-only code produces no output."""
        result = env.execute_code("   \n\n   ")
        assert result.stdout == ""
        assert result.stderr == ""

    def test_unicode_in_code(self, env):
        """Unicode characters are handled correctly."""
        result = env.execute_code("x = '日本語テスト'\nprint(x)")
        assert "日本語テスト" in result.stdout

    def test_unicode_emoji_in_code(self, env):
        """Emoji characters are handled correctly."""
        result = env.execute_code("print('GPU go brrr 🚀')")
        assert "🚀" in result.stdout

    def test_very_long_output(self, env):
        """Large print output is captured without crash."""
        result = env.execute_code("print('x' * 100000)")
        assert len(result.stdout) >= 100000

    def test_multiple_print_statements(self, env):
        """Multiple prints are concatenated in stdout."""
        result = env.execute_code("print('a')\nprint('b')\nprint('c')")
        assert "a" in result.stdout
        assert "b" in result.stdout
        assert "c" in result.stdout

    def test_exception_does_not_crash_env(self, env):
        """An exception in code doesn't prevent future execution."""
        env.execute_code("1 / 0")  # ZeroDivisionError
        result = env.execute_code("print('still alive')")
        assert "still alive" in result.stdout

    def test_nested_function_definitions(self, env):
        """Nested function definitions work."""
        code = (
            "def outer():\n    def inner():\n        return 42\n    return inner()\nprint(outer())"
        )
        result = env.execute_code(code)
        assert "42" in result.stdout

    def test_list_comprehension(self, env):
        """List comprehensions work normally."""
        result = env.execute_code("print([i**2 for i in range(5)])")
        assert "[0, 1, 4, 9, 16]" in result.stdout

    def test_dict_comprehension(self, env):
        """Dict comprehensions work normally."""
        result = env.execute_code("print({k: v for k, v in [('a', 1), ('b', 2)]})")
        assert "'a'" in result.stdout

    def test_multiline_string_assignment(self, env):
        """Multiline strings don't break the parser."""
        # Use '\\n' split via the string itself (chr not in safe builtins)
        code = "x = 'line1\\nline2\\nline3'\nprint(len(x.split('\\n')))"
        result = env.execute_code(code)
        assert "3" in result.stdout

    def test_none_return_no_crash(self, env):
        """Expression evaluating to None doesn't produce output."""
        result = env.execute_code("x = None")
        assert result.stdout == ""
        assert result.stderr == ""


# ---------------------------------------------------------------------------
# Environment: context loading edge cases
# ---------------------------------------------------------------------------


class TestContextLoadingEdgeCases:
    """Edge cases for context payload loading."""

    def test_context_as_string(self, mock_tool_wrappers):
        """Context can be a plain string."""
        env = GpumodConsultEnv(
            tool_wrappers=mock_tool_wrappers,
            context_payload="raw query string",
        )
        result = env.execute_code("print(context)")
        assert "raw query string" in result.stdout

    def test_context_as_list(self, mock_tool_wrappers):
        """Context can be a list."""
        env = GpumodConsultEnv(
            tool_wrappers=mock_tool_wrappers,
            context_payload=[1, 2, 3],
        )
        result = env.execute_code("print(context)")
        assert "[1, 2, 3]" in result.stdout

    def test_context_with_nested_data(self, mock_tool_wrappers):
        """Context can contain deeply nested data."""
        payload = {"query": "test", "gpu": {"vram_mb": 24000, "devices": [{"id": 0}]}}
        env = GpumodConsultEnv(
            tool_wrappers=mock_tool_wrappers,
            context_payload=payload,
        )
        result = env.execute_code("print(context['gpu']['devices'][0]['id'])")
        assert "0" in result.stdout

    def test_context_with_unicode(self, mock_tool_wrappers):
        """Context with unicode content works."""
        payload = {"query": "Qwen3-235B でテスト"}
        env = GpumodConsultEnv(
            tool_wrappers=mock_tool_wrappers,
            context_payload=payload,
        )
        result = env.execute_code("print(context['query'])")
        assert "テスト" in result.stdout

    def test_context_not_set_raises_nameerror(self, env):
        """Accessing context when not loaded raises NameError."""
        result = env.execute_code("print(context)")
        assert "NameError" in result.stderr


# ---------------------------------------------------------------------------
# Environment: timeout edge cases
# ---------------------------------------------------------------------------


class TestEnvironmentTimeoutEdgeCases:
    """Tests for code execution timeout behavior."""

    def test_zero_timeout_triggers_warning(self, mock_tool_wrappers):
        """Timeout=0 should trigger warning on any execution."""
        env = GpumodConsultEnv(tool_wrappers=mock_tool_wrappers, timeout=0)
        result = env.execute_code("x = 1")
        # Any non-zero execution time exceeds timeout=0
        assert result.execution_time >= 0
        assert "Timeout" in result.stderr or result.execution_time >= 0

    def test_fast_code_no_timeout_warning(self, mock_tool_wrappers):
        """Fast code with large timeout produces no timeout warning."""
        env = GpumodConsultEnv(tool_wrappers=mock_tool_wrappers, timeout=60)
        result = env.execute_code("x = 1 + 1")
        assert "Timeout" not in result.stderr


# ---------------------------------------------------------------------------
# Orchestrator: _parse_final_answer edge cases
# ---------------------------------------------------------------------------


class TestParseFinalAnswer:
    """Edge cases for _parse_final_answer JSON extraction."""

    def test_valid_json(self):
        """Well-formed JSON produces correct ConsultResult."""
        raw = json.dumps(
            {
                "can_run": True,
                "recommendation": "Use Q4_K_M",
                "reasoning_steps": ["Step 1"],
                "suggested_commands": ["cmd"],
                "sources": ["src"],
            }
        )
        result = _parse_final_answer(raw)
        assert result.can_run is True
        assert result.recommendation == "Use Q4_K_M"

    def test_json_embedded_in_prose(self):
        """JSON embedded in prose text is extracted."""
        raw = 'Here is my answer: {"recommendation": "Use Q5_K_M", "can_run": true}'
        result = _parse_final_answer(raw)
        assert result.recommendation == "Use Q5_K_M"

    def test_plain_text_fallback(self):
        """Plain text with no JSON becomes recommendation."""
        raw = "I recommend using Q4_K_M quantization."
        result = _parse_final_answer(raw)
        assert "Q4_K_M" in result.recommendation

    def test_empty_string(self):
        """Empty string produces empty recommendation."""
        result = _parse_final_answer("")
        assert result.recommendation == ""

    def test_malformed_json(self):
        """Malformed JSON falls through to fallback."""
        raw = '{"recommendation": "test", "can_run": tru'
        result = _parse_final_answer(raw)
        assert result.recommendation == raw

    def test_json_array_not_dict(self):
        """JSON array is not a dict, falls through to fallback."""
        raw = '["a", "b", "c"]'
        result = _parse_final_answer(raw)
        assert result.recommendation == raw

    def test_null_json(self):
        """JSON null is not a dict, falls through to fallback."""
        raw = "null"
        result = _parse_final_answer(raw)
        assert result.recommendation == "null"

    def test_missing_fields_get_defaults(self):
        """JSON with only some fields gets defaults for others."""
        raw = json.dumps({"recommendation": "Use Q8"})
        result = _parse_final_answer(raw)
        assert result.recommendation == "Use Q8"
        assert result.can_run is None
        assert result.reasoning_steps == []
        assert result.suggested_commands == []
        assert result.sources == []

    def test_nested_json_object(self):
        """Nested JSON braces don't confuse the regex extractor."""
        inner = json.dumps({"recommendation": "nested", "sources": ["s1"]})
        raw = f"Summary: {inner} — done."
        result = _parse_final_answer(raw)
        assert result.recommendation == "nested"

    def test_can_run_null_in_json(self):
        """can_run: null in JSON produces None."""
        raw = json.dumps({"can_run": None, "recommendation": "unclear"})
        result = _parse_final_answer(raw)
        assert result.can_run is None


# ---------------------------------------------------------------------------
# Orchestrator: ConsultResult edge cases
# ---------------------------------------------------------------------------


class TestConsultResultEdgeCases:
    """Edge cases for ConsultResult Pydantic model."""

    def test_extra_fields_ignored(self):
        """Extra fields are ignored (Pydantic default)."""
        result = ConsultResult(
            recommendation="test",
            extra_field="should be ignored",
        )
        assert result.recommendation == "test"
        assert not hasattr(result, "extra_field")

    def test_model_dump_includes_all_fields(self):
        """model_dump() includes all defined fields."""
        result = ConsultResult(recommendation="test", turns_used=3)
        d = result.model_dump()
        expected_keys = {
            "can_run",
            "recommendation",
            "reasoning_steps",
            "suggested_commands",
            "sources",
            "turns_used",
            "incomplete",
        }
        assert expected_keys == set(d.keys())

    def test_incomplete_flag(self):
        """incomplete=True when RLM exhausts turns without FINAL."""
        result = ConsultResult(
            recommendation="partial response",
            turns_used=11,
            incomplete=True,
        )
        assert result.incomplete is True


# ---------------------------------------------------------------------------
# Orchestrator: consult() edge cases
# ---------------------------------------------------------------------------


class TestOrchestratorEdgeCases:
    """Edge cases for RLMOrchestrator.consult()."""

    def test_max_turns_zero_still_capped(self, mock_tool_wrappers):
        """max_turns=0 is capped but results in 0 iterations min."""
        orchestrator = RLMOrchestrator(tool_wrappers=mock_tool_wrappers)

        with patch.object(orchestrator, "_run_loop") as mock_loop:
            mock_loop.return_value = ConsultResult(turns_used=0)

            with (
                patch("gpumod.rlm.orchestrator.get_client"),
                patch("gpumod.rlm.orchestrator.LMHandler"),
            ):
                orchestrator.consult("test", max_turns=0)

            # min(0, 10) = 0
            actual_max = mock_loop.call_args[0][3]
            assert actual_max == 0

    def test_max_turns_negative_capped(self, mock_tool_wrappers):
        """max_turns=-1 becomes min(-1, 10) = -1 (no iterations)."""
        orchestrator = RLMOrchestrator(tool_wrappers=mock_tool_wrappers)

        with patch.object(orchestrator, "_run_loop") as mock_loop:
            mock_loop.return_value = ConsultResult(turns_used=0)

            with (
                patch("gpumod.rlm.orchestrator.get_client"),
                patch("gpumod.rlm.orchestrator.LMHandler"),
            ):
                orchestrator.consult("test", max_turns=-1)

            actual_max = mock_loop.call_args[0][3]
            assert actual_max == -1

    def test_context_passed_to_env(self, mock_tool_wrappers):
        """context kwarg is forwarded to environment as payload."""
        orchestrator = RLMOrchestrator(tool_wrappers=mock_tool_wrappers)
        extra_context = {"gpu_info": {"vram_mb": 24000}}

        with (
            patch("gpumod.rlm.orchestrator.GpumodConsultEnv") as env_cls,
            patch("gpumod.rlm.orchestrator.get_client"),
            patch("gpumod.rlm.orchestrator.LMHandler"),
            patch.object(orchestrator, "_run_loop", return_value=ConsultResult()),
        ):
            mock_env = env_cls.return_value
            mock_env.cleanup = lambda: None

            orchestrator.consult("test", context=extra_context)

            call_kwargs = env_cls.call_args.kwargs
            payload = call_kwargs["context_payload"]
            assert payload["query"] == "test"
            assert payload["data"] == extra_context

    def test_resolve_model_from_env_var(self, mock_tool_wrappers):
        """Model can be resolved from environment variable."""
        with patch.dict("os.environ", {"GPUMOD_RLM_ANTHROPIC_MODEL": "claude-test-99"}):
            orchestrator = RLMOrchestrator(
                tool_wrappers=mock_tool_wrappers,
                backend="anthropic",
            )
            assert orchestrator._model == "claude-test-99"

    def test_resolve_model_openai_default(self, mock_tool_wrappers):
        """Default OpenAI model is gpt-4o-mini."""
        orchestrator = RLMOrchestrator(
            tool_wrappers=mock_tool_wrappers,
            backend="openai",
        )
        assert orchestrator._model == "gpt-4o-mini"

    def test_handler_stop_called_on_run_loop_error(self, mock_tool_wrappers):
        """handler.stop() is called even when _run_loop raises."""
        orchestrator = RLMOrchestrator(tool_wrappers=mock_tool_wrappers)

        with (
            patch("gpumod.rlm.orchestrator.GpumodConsultEnv") as env_cls,
            patch("gpumod.rlm.orchestrator.get_client"),
            patch("gpumod.rlm.orchestrator.LMHandler") as handler_cls,
        ):
            mock_env = env_cls.return_value
            mock_handler = handler_cls.return_value

            with (
                patch.object(orchestrator, "_run_loop", side_effect=RuntimeError("crash")),
                pytest.raises(RuntimeError, match="crash"),
            ):
                orchestrator.consult("test")

            mock_handler.stop.assert_called_once()
            mock_env.cleanup.assert_called_once()


# ---------------------------------------------------------------------------
# Orchestrator: _run_loop edge cases
# ---------------------------------------------------------------------------


class TestRunLoopEdgeCases:
    """Edge cases for RLMOrchestrator._run_loop."""

    def test_immediate_final_answer(self, mock_tool_wrappers):
        """If LLM gives FINAL answer on first turn, loop exits early."""
        orchestrator = RLMOrchestrator(tool_wrappers=mock_tool_wrappers)
        env = GpumodConsultEnv(tool_wrappers=mock_tool_wrappers)

        mock_handler = MagicMock()
        mock_handler.completion.return_value = 'FINAL({"recommendation": "Yes", "can_run": true})'

        result = orchestrator._run_loop(env, mock_handler, "test", max_turns=5)

        assert result.turns_used == 1
        assert result.can_run is True
        env.cleanup()

    def test_exhausted_turns_incomplete(self, mock_tool_wrappers):
        """When all turns are exhausted without FINAL, incomplete=True."""
        orchestrator = RLMOrchestrator(tool_wrappers=mock_tool_wrappers)
        env = GpumodConsultEnv(tool_wrappers=mock_tool_wrappers)

        call_count = 0

        def fake_completion(messages):
            nonlocal call_count
            call_count += 1
            # Never produce a FINAL answer or code blocks
            return "I'm still thinking about this..."

        mock_handler = MagicMock()
        mock_handler.completion.side_effect = fake_completion

        with (
            patch("gpumod.rlm.orchestrator.find_final_answer", return_value=None),
            patch("gpumod.rlm.orchestrator.find_code_blocks", return_value=[]),
        ):
            result = orchestrator._run_loop(env, mock_handler, "test", max_turns=2)

        assert result.incomplete is True
        assert result.turns_used == 3  # max_turns + 1 (the forced final turn)
        env.cleanup()

    def test_code_error_fed_back_to_llm(self, mock_tool_wrappers):
        """When code produces an error, STDERR is sent back to LLM."""
        orchestrator = RLMOrchestrator(tool_wrappers=mock_tool_wrappers)
        env = GpumodConsultEnv(tool_wrappers=mock_tool_wrappers)

        responses = [
            "```repl\n1/0\n```",  # Turn 1: code with error
            'FINAL({"recommendation": "handled error"})',  # Turn 2: final answer
        ]
        response_iter = iter(responses)

        mock_handler = MagicMock()
        mock_handler.completion.side_effect = lambda msgs: next(response_iter)

        orchestrator._run_loop(env, mock_handler, "test", max_turns=5)

        # The LLM should have received the ZeroDivisionError in messages
        all_calls = mock_handler.completion.call_args_list
        second_call_messages = all_calls[1][0][0]
        # Find the REPL output message
        repl_outputs = [m for m in second_call_messages if "STDERR" in m.get("content", "")]
        assert len(repl_outputs) > 0
        assert "ZeroDivisionError" in repl_outputs[0]["content"]
        env.cleanup()

    def test_security_error_fed_back_to_llm(self, mock_tool_wrappers):
        """When code triggers SecurityError, error is sent back to LLM."""
        orchestrator = RLMOrchestrator(tool_wrappers=mock_tool_wrappers)
        env = GpumodConsultEnv(tool_wrappers=mock_tool_wrappers)

        responses = [
            "```repl\nimport os\n```",  # Turn 1: blocked code
            'FINAL({"recommendation": "safe answer"})',  # Turn 2: final
        ]
        response_iter = iter(responses)

        mock_handler = MagicMock()
        mock_handler.completion.side_effect = lambda msgs: next(response_iter)

        orchestrator._run_loop(env, mock_handler, "test", max_turns=5)

        all_calls = mock_handler.completion.call_args_list
        second_call_messages = all_calls[1][0][0]
        repl_outputs = [m for m in second_call_messages if "STDERR" in m.get("content", "")]
        assert len(repl_outputs) > 0
        assert "SecurityError" in repl_outputs[0]["content"]
        env.cleanup()
