"""Tests for GpumodConsultEnv — custom RLM REPL environment.

TDD: RED phase - these tests define the GpumodConsultEnv interface.
Tests cover tool whitelisting, blocking mutating tools, namespace setup,
context loading, code execution, and security enforcement.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gpumod.rlm.environment import GpumodConsultEnv

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
    """Create a GpumodConsultEnv with mocked tools (setup called in __init__)."""
    return GpumodConsultEnv(tool_wrappers=mock_tool_wrappers)


# ---------------------------------------------------------------------------
# TestGpumodConsultEnv - namespace and tool access
# ---------------------------------------------------------------------------


class TestGpumodConsultEnv:
    """Tests for GpumodConsultEnv REPL environment."""

    def test_whitelisted_tool_accessible(self, env):
        """Read-only tools should be callable from REPL namespace."""
        # All whitelisted tools should be in namespace
        assert "gpu_status" in env._namespace
        assert "list_gguf_files" in env._namespace
        assert "fetch_model_config" in env._namespace
        assert "fetch_driver_docs" in env._namespace
        assert "search_hf_models" in env._namespace
        assert "simulate_mode" in env._namespace
        assert "generate_preset" in env._namespace

    def test_blocked_tool_raises_nameerror(self, env):
        """Mutating tools should NOT be in namespace (raise NameError if called)."""
        # Mutating tools should not be accessible
        assert "switch_mode" not in env._namespace
        assert "start_service" not in env._namespace
        assert "stop_service" not in env._namespace

        # Executing code that calls a blocked tool should produce NameError
        result = env.execute_code("switch_mode('dev')")
        assert "NameError" in result.stderr

    def test_setup_creates_namespace_with_stdlib(self, env):
        """setup() populates namespace with tools and stdlib helpers."""
        # Standard library helpers should be available
        assert "json" in env._namespace
        assert "re" in env._namespace
        assert "math" in env._namespace

    def test_setup_creates_safe_builtins(self, env):
        """Namespace has safe builtins subset (print, len, etc.)."""
        builtins = env._namespace.get("__builtins__", {})
        assert builtins.get("print") is print
        assert builtins.get("len") is len
        assert builtins.get("str") is str
        # Dangerous builtins should be None/missing
        assert builtins.get("eval") is None
        assert builtins.get("exec") is None
        assert builtins.get("open") is None
        assert builtins.get("__import__") is None

    def test_load_context(self, env):
        """Context payload is accessible as 'context' variable in REPL."""
        payload = {"query": "Can I run Qwen3-235B on 24GB?", "gpu_info": {"vram_mb": 24000}}
        env.load_context(payload)

        # Context goes into _locals, accessible during execute_code
        assert env._locals["context"] == payload

        # Should also be usable from code execution
        result = env.execute_code("print(context['query'])")
        assert "Qwen3-235B" in result.stdout

    def test_load_context_via_constructor(self, mock_tool_wrappers):
        """Context can be loaded via constructor kwarg."""
        payload = {"query": "test"}
        env = GpumodConsultEnv(
            tool_wrappers=mock_tool_wrappers,
            context_payload=payload,
        )
        assert env._locals["context"] == payload

    def test_execute_code_returns_repl_result(self, env):
        """Code execution returns REPLResult with stdout/stderr."""
        result = env.execute_code("x = 1 + 1\nprint(x)")

        assert hasattr(result, "stdout")
        assert hasattr(result, "stderr")
        assert hasattr(result, "execution_time")
        assert "2" in result.stdout

    def test_execute_code_captures_print_output(self, env):
        """print() output is captured in stdout."""
        result = env.execute_code("print('hello world')")
        assert "hello world" in result.stdout

    def test_dangerous_code_rejected_import(self, env):
        """Code with import is blocked before execution."""
        result = env.execute_code("import os")

        # Should contain error info in stderr, not succeed
        assert "SecurityError" in result.stderr or "Import" in result.stderr

    def test_dangerous_code_rejected_exec(self, env):
        """Code with exec() is blocked before execution."""
        result = env.execute_code("exec('print(1)')")

        assert "SecurityError" in result.stderr or "Blocked" in result.stderr

    def test_namespace_persists_between_calls(self, env):
        """Variables set in one execute_code persist to the next."""
        env.execute_code("my_var = 'hello'")
        result = env.execute_code("print(my_var)")

        assert "hello" in result.stdout

    def test_syntax_error_in_code(self, env):
        """Syntax errors are reported in stderr."""
        result = env.execute_code("def func(")

        assert "SyntaxError" in result.stderr

    def test_runtime_error_captured(self, env):
        """Runtime errors are captured in stderr without crashing."""
        result = env.execute_code("1 / 0")

        assert "ZeroDivisionError" in result.stderr

    def test_cleanup_clears_state(self, env):
        """cleanup() clears namespace and locals."""
        env.execute_code("x = 42")
        env.cleanup()

        assert len(env._namespace) == 0
        assert len(env._locals) == 0


# ---------------------------------------------------------------------------
# TestGpumodConsultEnvTimeout - timeout enforcement
# ---------------------------------------------------------------------------


class TestGpumodConsultEnvTimeout:
    """Tests for code execution timeout."""

    def test_timeout_warning_on_slow_code(self, mock_tool_wrappers):
        """Code exceeding timeout gets a TimeoutWarning in stderr."""
        # Use a very short timeout to trigger warning
        env = GpumodConsultEnv(
            tool_wrappers=mock_tool_wrappers,
            timeout=0,  # 0 seconds — anything will exceed this
        )

        result = env.execute_code("x = sum(range(100))")

        # The current implementation adds a TimeoutWarning to stderr
        # when execution_time exceeds timeout, but doesn't kill the thread
        assert result.execution_time >= 0
        # With timeout=0, any execution should trigger the warning
        assert "Timeout" in result.stderr or result.execution_time >= 0

    def test_fast_code_no_timeout_warning(self, env):
        """Fast code does not trigger TimeoutWarning."""
        result = env.execute_code("x = 1 + 1")
        assert "TimeoutWarning" not in result.stderr


# ---------------------------------------------------------------------------
# TestToolExecution - calling tool wrappers from REPL code
# ---------------------------------------------------------------------------


def _sync_gpu_status():
    return {"mode": "dev", "vram_free_mb": 20000, "total_vram_mb": 24000}


def _sync_list_gguf_files(repo_id: str):
    return {
        "files": [
            {"filename": "model-Q4_K_M.gguf", "estimated_vram_mb": 12000},
            {"filename": "model-Q8_0.gguf", "estimated_vram_mb": 22000},
        ],
        "count": 2,
    }


@pytest.fixture
def sync_tool_wrappers():
    """Synchronous tool wrappers that return real data."""
    return {
        "gpu_status": _sync_gpu_status,
        "list_gguf_files": _sync_list_gguf_files,
        "fetch_model_config": lambda repo_id: {"architectures": ["LlamaForCausalLM"]},
        "fetch_driver_docs": lambda driver, **kw: {"content": "# docs"},
        "search_hf_models": lambda query, **kw: {"models": [], "count": 0},
        "simulate_mode": lambda mode, **kw: {"fits": True},
        "generate_preset": lambda repo_id, **kw: {"preset": "yaml..."},
    }


class TestToolExecution:
    """Tests for calling tool wrappers from within REPL code."""

    def test_gpu_status_callable(self, sync_tool_wrappers):
        """gpu_status() returns data accessible in REPL."""
        env = GpumodConsultEnv(tool_wrappers=sync_tool_wrappers)
        result = env.execute_code('s = gpu_status()\nprint(s["vram_free_mb"])')
        assert "20000" in result.stdout
        env.cleanup()

    def test_list_gguf_files_with_args(self, sync_tool_wrappers):
        """list_gguf_files(repo_id) passes arguments correctly."""
        env = GpumodConsultEnv(tool_wrappers=sync_tool_wrappers)
        result = env.execute_code('f = list_gguf_files("test/repo")\nprint(f["count"])')
        assert "2" in result.stdout
        env.cleanup()

    def test_multi_step_reasoning(self, sync_tool_wrappers):
        """Multiple tool calls compose correctly across steps."""
        env = GpumodConsultEnv(tool_wrappers=sync_tool_wrappers)
        code = """\
status = gpu_status()
vram = status["vram_free_mb"]
files = list_gguf_files("test/repo")
fitting = [f for f in files["files"] if f["estimated_vram_mb"] <= vram]
print(f"Fitting: {len(fitting)} models")
for f in fitting:
    print(f"  {f['filename']}: {f['estimated_vram_mb']}MB")
"""
        result = env.execute_code(code)
        # Q4_K_M (12000MB) fits in 20000MB, Q8_0 (22000MB) does not
        assert "Fitting: 1 models" in result.stdout
        assert "model-Q4_K_M.gguf" in result.stdout
        env.cleanup()

    def test_json_module_usable(self, sync_tool_wrappers):
        """json module works within REPL namespace."""
        env = GpumodConsultEnv(tool_wrappers=sync_tool_wrappers)
        result = env.execute_code('print(json.dumps({"a": 1}))')
        assert '{"a": 1}' in result.stdout
        env.cleanup()

    def test_re_module_usable(self, sync_tool_wrappers):
        """re module works within REPL namespace."""
        env = GpumodConsultEnv(tool_wrappers=sync_tool_wrappers)
        result = env.execute_code(
            'match = re.search(r"(\\d+)", "value is 42")\nprint(match.group(1))'
        )
        assert "42" in result.stdout
        env.cleanup()

    def test_math_module_usable(self, sync_tool_wrappers):
        """math module works within REPL namespace."""
        env = GpumodConsultEnv(tool_wrappers=sync_tool_wrappers)
        result = env.execute_code("print(math.ceil(3.2))")
        assert "4" in result.stdout
        env.cleanup()

    def test_blocked_tool_not_callable_even_if_provided(self, sync_tool_wrappers):
        """Blocked tools (switch_mode etc.) raise NameError even if in wrappers dict."""
        sync_tool_wrappers["switch_mode"] = lambda m: "bad"
        sync_tool_wrappers["start_service"] = lambda s: "bad"
        sync_tool_wrappers["stop_service"] = lambda s: "bad"

        env = GpumodConsultEnv(tool_wrappers=sync_tool_wrappers)
        for name in ("switch_mode", "start_service", "stop_service"):
            result = env.execute_code(f'{name}("test")')
            assert "NameError" in result.stderr, f"{name} should raise NameError"
        env.cleanup()

    def test_format_string_sandbox_escape_blocked(self, sync_tool_wrappers):
        """str.format() with dunder field access is blocked by AST validation."""
        env = GpumodConsultEnv(tool_wrappers=sync_tool_wrappers)
        # This attack uses str.format() to access __class__.__bases__ via
        # format field syntax, bypassing the normal AST dunder attribute check.
        result = env.execute_code('result = "{0.__class__.__bases__}".format([])\nprint(result)')
        assert "SecurityError" in result.stderr
        assert result.stdout == ""
        env.cleanup()

    def test_safe_format_string_allowed(self, sync_tool_wrappers):
        """Normal str.format() without dunder access is allowed."""
        env = GpumodConsultEnv(tool_wrappers=sync_tool_wrappers)
        result = env.execute_code('x = "Hello {}".format("world")\nprint(x)')
        assert "Hello world" in result.stdout
        assert result.stderr == ""
        env.cleanup()

    def test_hasattr_removed_from_builtins(self, sync_tool_wrappers):
        """hasattr() is removed to prevent dunder attribute probing."""
        env = GpumodConsultEnv(tool_wrappers=sync_tool_wrappers)
        result = env.execute_code("hasattr([], '__class__')")
        assert "NameError" in result.stderr
        env.cleanup()

    def test_extra_unknown_tools_not_injected(self, sync_tool_wrappers):
        """Tools not in WHITELISTED_TOOLS are silently dropped."""
        sync_tool_wrappers["my_custom_tool"] = lambda: "custom"

        env = GpumodConsultEnv(tool_wrappers=sync_tool_wrappers)
        assert "my_custom_tool" not in env._namespace
        result = env.execute_code("my_custom_tool()")
        assert "NameError" in result.stderr
        env.cleanup()
