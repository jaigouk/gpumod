"""AST-based code validation for RLM REPL execution.

Blocks dangerous constructs (imports, exec, eval, etc.) before code
is executed in the GpumodConsultEnv sandbox.
"""

from __future__ import annotations

import ast
import re

# Matches Python format-spec dunder access like {0.__class__} or {x.__bases__}.
_FORMAT_DUNDER_RE = re.compile(r"\{[^}]*\.__\w+")


class SecurityError(Exception):
    """Raised when code contains dangerous constructs."""


BLOCKED_CALL_NAMES: frozenset[str] = frozenset(
    {
        "exec",
        "eval",
        "open",
        "__import__",
        "compile",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "exit",
        "quit",
    }
)

# Module names that should never appear as call targets (e.g. os.system).
BLOCKED_MODULE_NAMES: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
        "socket",
        "ctypes",
        "signal",
    }
)


def validate_code(code: str) -> None:
    """Validate code via AST analysis before execution.

    Raises:
        SecurityError: If the code contains blocked constructs.
        SyntaxError: If the code cannot be parsed.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            raise SecurityError(f"Import statements not allowed: {ast.dump(node)}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in BLOCKED_CALL_NAMES
        ):
            raise SecurityError(f"Blocked function call: {node.func.id}")
        # Block attribute access to __builtins__, __class__, __subclasses__ etc.
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SecurityError(f"Dunder attribute access not allowed: {node.attr}")
        # Block calls on dangerous module names (e.g. os.system, subprocess.call).
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in BLOCKED_MODULE_NAMES
        ):
            raise SecurityError(f"Blocked module call: {node.func.value.id}.{node.func.attr}")
        # Block str.format() on strings containing dunder field access.
        # e.g. "{0.__class__.__bases__}".format([]) bypasses AST dunder checks
        # because the attribute access is inside a string constant.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str)
            and _FORMAT_DUNDER_RE.search(node.func.value.value)
        ):
            raise SecurityError(
                "Format string contains dunder attribute access — potential sandbox escape"
            )
