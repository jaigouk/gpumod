"""Tests for level definitions and pytest-based validators.

TDD Phase: RED - Write failing tests first.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# LevelDefinition tests
# ---------------------------------------------------------------------------


class TestLevelDefinition:
    """Tests for LevelDefinition dataclass."""

    def test_level_definition_exists(self) -> None:
        """LevelDefinition dataclass exists in levels module."""
        from gpumod.benchmarks.qwen35.levels import LevelDefinition

        assert LevelDefinition is not None

    def test_level_has_detailed_prompt(self) -> None:
        """Level definition has detailed prompt (500+ chars)."""
        from gpumod.benchmarks.qwen35.levels import LevelDefinition

        level = LevelDefinition(
            level=1,
            name="Test Level",
            points=25,
            prompt="x" * 500,
            test_code="def test_example(): pass",
        )

        assert len(level.prompt) >= 500

    def test_level_has_test_code(self) -> None:
        """Level definition includes pytest test code."""
        from gpumod.benchmarks.qwen35.levels import LevelDefinition

        level = LevelDefinition(
            level=1,
            name="Test Level",
            points=25,
            prompt="x" * 500,
            test_code="def test_basic(): assert True",
        )

        assert "def test_" in level.test_code


# ---------------------------------------------------------------------------
# PytestValidator tests
# ---------------------------------------------------------------------------


class TestPytestValidator:
    """Tests for PytestValidator that runs generated code through pytest."""

    def test_validator_exists(self) -> None:
        """PytestValidator class exists."""
        from gpumod.benchmarks.qwen35.levels import PytestValidator

        assert PytestValidator is not None

    def test_validate_returns_result_with_passed(self) -> None:
        """Validator returns result with passed flag."""
        from gpumod.benchmarks.qwen35.levels import PytestValidator

        validator = PytestValidator()

        # Valid code that passes tests
        code = """
def add(a, b):
    return a + b
"""
        test_code = """
def test_add():
    from solution import add
    assert add(1, 2) == 3
"""
        result = validator.validate(code, test_code)

        assert hasattr(result, "passed")
        assert isinstance(result.passed, bool)

    def test_validate_returns_partial_score(self) -> None:
        """Validator returns partial score based on test pass rate."""
        from gpumod.benchmarks.qwen35.levels import PytestValidator

        validator = PytestValidator()

        # Code that passes some tests
        code = """
def add(a, b):
    return a + b

def subtract(a, b):
    return a  # Bug: should be a - b
"""
        test_code = """
def test_add():
    from solution import add
    assert add(1, 2) == 3

def test_subtract():
    from solution import subtract
    assert subtract(5, 3) == 2
"""
        result = validator.validate(code, test_code)

        # Should have partial score (1/2 tests pass = 0.5)
        assert hasattr(result, "pass_rate")
        assert 0 <= result.pass_rate <= 1

    def test_validate_handles_syntax_errors(self) -> None:
        """Validator handles code with syntax errors gracefully."""
        from gpumod.benchmarks.qwen35.levels import PytestValidator

        validator = PytestValidator()

        # Invalid Python syntax
        code = "def broken(:"
        test_code = """
def test_broken():
    from solution import broken
    broken()
"""

        result = validator.validate(code, test_code)

        assert result.passed is False
        assert result.pass_rate == 0.0
        assert result.error is not None

    def test_validate_has_timeout(self) -> None:
        """Validator times out on infinite loops."""
        from gpumod.benchmarks.qwen35.levels import PytestValidator

        validator = PytestValidator(timeout_seconds=1)

        # Code with infinite loop
        code = """
def infinite():
    while True:
        pass
"""
        test_code = """
def test_infinite():
    from solution import infinite
    infinite()
"""
        result = validator.validate(code, test_code)

        # Should timeout and fail
        assert result.passed is False
        assert "timeout" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# ValidationResult tests
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_result_has_required_fields(self) -> None:
        """ValidationResult has all required fields."""
        from gpumod.benchmarks.qwen35.levels import ValidationResult

        result = ValidationResult(
            passed=True,
            pass_rate=1.0,
            tests_passed=5,
            tests_total=5,
            error=None,
        )

        assert result.passed is True
        assert result.pass_rate == 1.0
        assert result.tests_passed == 5
        assert result.tests_total == 5
        assert result.error is None


# ---------------------------------------------------------------------------
# Level Registry tests
# ---------------------------------------------------------------------------


class TestLevelRegistry:
    """Tests for LEVEL_REGISTRY that allows adding levels without modifying runner."""

    def test_registry_exists(self) -> None:
        """LEVEL_REGISTRY exists in levels module."""
        from gpumod.benchmarks.qwen35.levels import LEVEL_REGISTRY

        assert LEVEL_REGISTRY is not None
        assert isinstance(LEVEL_REGISTRY, dict)

    def test_registry_has_default_levels(self) -> None:
        """Registry has default levels (1-5)."""
        from gpumod.benchmarks.qwen35.levels import LEVEL_REGISTRY

        assert 1 in LEVEL_REGISTRY
        assert 5 in LEVEL_REGISTRY

    def test_can_register_new_level(self) -> None:
        """Can register a new level without modifying runner."""
        from gpumod.benchmarks.qwen35.levels import (
            LevelDefinition,
            register_level,
        )

        # Register a custom level
        custom_level = LevelDefinition(
            level=99,
            name="Custom Level",
            points=50,
            prompt="x" * 500,
            test_code="def test_custom(): pass",
        )

        register_level(custom_level)

        from gpumod.benchmarks.qwen35.levels import LEVEL_REGISTRY

        assert 99 in LEVEL_REGISTRY
        assert LEVEL_REGISTRY[99].name == "Custom Level"

        # Clean up
        del LEVEL_REGISTRY[99]

    def test_get_level_returns_definition(self) -> None:
        """get_level() returns LevelDefinition by level number."""
        from gpumod.benchmarks.qwen35.levels import get_level

        level = get_level(1)

        assert level is not None
        assert level.level == 1


# ---------------------------------------------------------------------------
# Default level prompts tests
# ---------------------------------------------------------------------------


class TestDefaultLevelPrompts:
    """Tests for default level prompts (L1-L5)."""

    def test_level_1_has_detailed_prompt(self) -> None:
        """Level 1 (Basic Queue) has detailed prompt."""
        from gpumod.benchmarks.qwen35.levels import get_level

        level = get_level(1)

        assert len(level.prompt) >= 500
        assert "job" in level.prompt.lower() or "queue" in level.prompt.lower()

    def test_level_1_has_test_code(self) -> None:
        """Level 1 has pytest test code."""
        from gpumod.benchmarks.qwen35.levels import get_level

        level = get_level(1)

        assert "def test_" in level.test_code
        assert "add_job" in level.test_code or "get_result" in level.test_code

    def test_all_levels_have_valid_prompts(self) -> None:
        """All default levels (1-5) have valid prompts."""
        from gpumod.benchmarks.qwen35.levels import LEVEL_REGISTRY

        for level_num in [1, 2, 3, 4, 5]:
            level = LEVEL_REGISTRY[level_num]
            assert len(level.prompt) >= 500, f"Level {level_num} prompt too short"
            assert "def test_" in level.test_code, f"Level {level_num} missing tests"
