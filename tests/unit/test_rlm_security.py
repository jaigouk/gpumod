"""Tests for RLM security — AST-based code validation.

Tests cover blocked constructs (imports, exec, eval, open, etc.),
allowed patterns (valid code, strings containing keywords),
and dunder attribute access blocking.
"""

from __future__ import annotations

import pytest

from gpumod.rlm.security import BLOCKED_CALL_NAMES, SecurityError, validate_code

# ---------------------------------------------------------------------------
# TestCodeValidation - AST-based code validation
# ---------------------------------------------------------------------------


class TestCodeValidation:
    """Tests for validate_code AST security checker."""

    def test_valid_code_passes(self):
        """Simple valid code should pass validation."""
        validate_code("x = gpu_status()\nprint(x)")

    def test_valid_multiline_code_passes(self):
        """Multiline code with function calls should pass."""
        code = """\
result = gpu_status()
files = list_gguf_files("unsloth/model-GGUF")
for f in files:
    print(f)
"""
        validate_code(code)

    def test_empty_code_passes(self):
        """Empty code string should pass validation."""
        validate_code("")

    def test_whitespace_only_passes(self):
        """Whitespace-only code should pass."""
        validate_code("   \n\n   ")

    def test_string_containing_import_passes(self):
        """String literal containing 'import' should not be blocked."""
        validate_code('msg = "you need to import the module"')

    def test_comment_containing_import_passes(self):
        """Comment containing 'import' keyword should not be blocked."""
        validate_code("# import os  -- this is just a comment\nx = 1")

    # -----------------------------------------------------------------------
    # Blocked: import statements
    # -----------------------------------------------------------------------

    def test_import_blocked(self):
        """import statement should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("import os")

    def test_from_import_blocked(self):
        """from ... import statement should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("from os import system")

    def test_import_star_blocked(self):
        """from ... import * should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("from os import *")

    # -----------------------------------------------------------------------
    # Blocked: dangerous builtins
    # -----------------------------------------------------------------------

    def test_exec_blocked(self):
        """exec() call should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("exec('print(1)')")

    def test_eval_blocked(self):
        """eval() call should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("eval('1+1')")

    def test_open_blocked(self):
        """open() call should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("f = open('/etc/passwd')")

    def test_dunder_import_blocked(self):
        """__import__() call should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("__import__('os')")

    def test_compile_blocked(self):
        """compile() call should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("compile('import os', '<string>', 'exec')")

    def test_globals_blocked(self):
        """globals() call should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("g = globals()")

    def test_locals_blocked(self):
        """locals() call should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("l = locals()")

    def test_getattr_blocked(self):
        """getattr() call should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("getattr(obj, 'system')")

    def test_setattr_blocked(self):
        """setattr() call should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("setattr(obj, 'attr', val)")

    def test_delattr_blocked(self):
        """delattr() call should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("delattr(obj, 'attr')")

    def test_breakpoint_blocked(self):
        """breakpoint() call should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("breakpoint()")

    def test_exit_blocked(self):
        """exit() call should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("exit()")

    def test_quit_blocked(self):
        """quit() call should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("quit()")

    # -----------------------------------------------------------------------
    # Blocked: dunder attribute access
    # -----------------------------------------------------------------------

    def test_dunder_class_blocked(self):
        """__class__ attribute access should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("x.__class__")

    def test_dunder_subclasses_blocked(self):
        """__subclasses__ attribute access should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("x.__subclasses__()")

    def test_dunder_builtins_blocked(self):
        """__builtins__ attribute access should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("x.__builtins__")

    def test_dunder_globals_blocked(self):
        """__globals__ attribute access should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("func.__globals__")

    def test_dunder_import_attr_blocked(self):
        """__import__ attribute access should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("x.__import__('os')")

    # -----------------------------------------------------------------------
    # Blocked: nested/obfuscated attempts
    # -----------------------------------------------------------------------

    def test_nested_eval_import(self):
        """Nested eval with __import__ should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("eval(\"__import__('os')\")")

    def test_exec_with_string_concat(self):
        """exec() with string arguments should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("exec('imp' + 'ort os')")

    # -----------------------------------------------------------------------
    # Blocked: format string sandbox escape
    # -----------------------------------------------------------------------

    def test_format_dunder_class_blocked(self):
        """str.format() with __class__ in format spec is blocked."""
        with pytest.raises(SecurityError):
            validate_code('"{0.__class__}".format([])')

    def test_format_dunder_bases_chain_blocked(self):
        """str.format() with chained dunder access is blocked."""
        with pytest.raises(SecurityError):
            validate_code('"{0.__class__.__bases__[0].__subclasses__}".format([])')

    def test_format_safe_string_passes(self):
        """str.format() without dunder access should pass."""
        validate_code('"Hello {}".format("world")')

    def test_format_named_field_safe_passes(self):
        """str.format() with named fields (no dunders) should pass."""
        validate_code('"{name} is {age}".format(name="Alice", age=30)')

    # -----------------------------------------------------------------------
    # Blocked: module attribute calls (os.system, etc.)
    # -----------------------------------------------------------------------

    def test_os_system_via_attribute(self):
        """os.system() should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("os.system('ls')")

    def test_subprocess_call_blocked(self):
        """subprocess.call() should be blocked."""
        with pytest.raises(SecurityError):
            validate_code("subprocess.call(['ls'])")

    # -----------------------------------------------------------------------
    # Syntax errors
    # -----------------------------------------------------------------------

    def test_syntax_error_raises_syntax_error(self):
        """Invalid Python syntax raises SyntaxError, not SecurityError."""
        with pytest.raises(SyntaxError):
            validate_code("def func(")

    # -----------------------------------------------------------------------
    # Blocked names set completeness
    # -----------------------------------------------------------------------

    def test_blocked_call_names_is_frozenset(self):
        """BLOCKED_CALL_NAMES is an immutable frozenset."""
        assert isinstance(BLOCKED_CALL_NAMES, frozenset)

    def test_all_dangerous_builtins_blocked(self):
        """All expected dangerous builtins are in the blocked set."""
        expected = {
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
        assert expected.issubset(BLOCKED_CALL_NAMES)


# ---------------------------------------------------------------------------
# TestSecurityError - exception class
# ---------------------------------------------------------------------------


class TestSecurityError:
    """Tests for SecurityError exception."""

    def test_security_error_is_exception(self):
        """SecurityError inherits from Exception."""
        assert issubclass(SecurityError, Exception)

    def test_security_error_has_message(self):
        """SecurityError carries a descriptive message."""
        err = SecurityError("import statement not allowed")
        assert "import" in str(err)
