"""Custom RLM environment with whitelisted gpumod MCP tools.

Only read-only tools are exposed. Mutating operations
(switch_mode, start_service, stop_service) raise NameError.
"""

from __future__ import annotations

import io
import json
import math
import re
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from rlm.core.types import REPLResult
from rlm.environments.base_env import NonIsolatedEnv

from gpumod.rlm.security import SecurityError, validate_code

# Timeout for individual code block execution (seconds).
CODE_TIMEOUT_SECONDS = 5

# Safe builtins subset — mirrors rlm's LocalREPL but tighter.
_SAFE_BUILTINS: dict[str, Any] = {
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "repr": repr,
    # "format" removed: enables sandbox escape via class traversal
    # e.g. "{0.__class__.__bases__[0].__subclasses__()}".format([])
    "iter": iter,
    "next": next,
    "callable": callable,
    # "hasattr" removed: enables dunder attribute probing via string args
    # Exceptions
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "StopIteration": StopIteration,
    # Explicitly blocked
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
    "__import__": None,
    "open": None,
}


class GpumodConsultEnv(NonIsolatedEnv):
    """Custom RLM environment with whitelisted gpumod MCP tools.

    Extends NonIsolatedEnv so it plugs into the RLM iteration loop.
    Only read-only tool wrappers are injected; mutating operations
    are blocked by omission (they simply don't exist in the namespace,
    producing a NameError if called).
    """

    WHITELISTED_TOOLS: frozenset[str] = frozenset(
        {
            "gpu_status",
            "list_gguf_files",
            "fetch_model_config",
            "fetch_driver_docs",
            "search_hf_models",
            "simulate_mode",
            "generate_preset",
        }
    )

    BLOCKED_TOOLS: frozenset[str] = frozenset(
        {
            "switch_mode",
            "start_service",
            "stop_service",
        }
    )

    def __init__(
        self,
        tool_wrappers: dict[str, Callable[..., Any]],
        *,
        context_payload: dict[str, Any] | list[Any] | str | None = None,
        timeout: int = CODE_TIMEOUT_SECONDS,
        **kwargs: Any,
    ) -> None:
        self._tool_wrappers = tool_wrappers
        self._timeout = timeout
        self._namespace: dict[str, Any] = {}
        self._locals: dict[str, Any] = {}
        self._lock = threading.Lock()
        super().__init__(**kwargs)

        # setup() is NOT called by BaseEnv.__init__, call it explicitly.
        self.setup()

        if context_payload is not None:
            self.load_context(context_payload)

    # ------------------------------------------------------------------
    # BaseEnv abstract interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize namespace with whitelisted tools and safe stdlib."""
        self._namespace = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__gpumod_consult__",
        }
        self._locals = {}

        # Inject only whitelisted tool wrappers.
        for name, func in self._tool_wrappers.items():
            if name in self.WHITELISTED_TOOLS:
                self._namespace[name] = func

        # Safe stdlib modules.
        self._namespace["json"] = json
        self._namespace["re"] = re
        self._namespace["math"] = math

    def load_context(self, context_payload: dict[str, Any] | list[Any] | str) -> None:
        """Make *context_payload* accessible as ``context`` in the REPL."""
        self._locals["context"] = context_payload

    def execute_code(self, code: str) -> REPLResult:
        """Execute *code* in the restricted namespace.

        The code is first validated via AST analysis. Execution is wrapped
        with a timeout; if the code exceeds ``self._timeout`` seconds the
        thread is *not* killed (Python limitation) but the result is
        returned with an error message.
        """
        start = time.perf_counter()

        # AST validation
        try:
            validate_code(code)
        except (SecurityError, SyntaxError) as exc:
            return REPLResult(
                stdout="",
                stderr=f"{type(exc).__name__}: {exc}",
                locals=self._locals.copy(),
                execution_time=time.perf_counter() - start,
            )

        # Execute with captured output.
        stdout_text = ""
        stderr_text = ""

        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            try:
                sys.stdout = stdout_buf
                sys.stderr = stderr_buf

                combined = {**self._namespace, **self._locals}
                exec(code, combined, combined)  # noqa: S102 — intentional sandboxed exec

                # Capture new variables back into _locals.
                for key, value in combined.items():
                    if key not in self._namespace and not key.startswith("_"):
                        self._locals[key] = value

                stdout_text = stdout_buf.getvalue()
                stderr_text = stderr_buf.getvalue()
            except Exception as exc:
                stdout_text = stdout_buf.getvalue()
                stderr_text = stderr_buf.getvalue() + f"\n{type(exc).__name__}: {exc}"
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        elapsed = time.perf_counter() - start
        if elapsed > self._timeout:
            stderr_text += (
                f"\nTimeoutWarning: execution took {elapsed:.1f}s (limit {self._timeout}s)"
            )

        return REPLResult(
            stdout=stdout_text,
            stderr=stderr_text,
            locals=self._locals.copy(),
            execution_time=elapsed,
        )

    def cleanup(self) -> None:
        """Release references."""
        self._namespace.clear()
        self._locals.clear()
