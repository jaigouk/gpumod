"""Tests for code extraction from model responses (gpumod-9ial).

Two pure-correctness fixes:
  Bug 1 — extractor must unwrap the legacy artifact wrapper
          (<reasoning_content>...</reasoning_content>\\n\\n<content>...</content>)
          so re-extraction for artifact debugging matches what was validated.
  Bug 2 — extractor must dedent code blocks wrapped inside indented contexts
          (Gemma 4 emits ```python inside numbered list items at column 4+).
"""

from __future__ import annotations


class TestExtractCodePlainFences:
    """Existing-behavior regression — must keep passing after the fixes."""

    def test_extracts_python_fenced_block(self) -> None:
        from gpumod.benchmarks.coding.code_extraction import extract_code

        response = "Here is the code:\n```python\nclass Foo:\n    pass\n```\nDone."
        assert extract_code(response) == "class Foo:\n    pass"

    def test_extracts_bare_fence_block_with_python_first_line(self) -> None:
        from gpumod.benchmarks.coding.code_extraction import extract_code

        response = "```\nclass Foo:\n    pass\n```"
        assert extract_code(response) == "class Foo:\n    pass"

    def test_returns_stripped_response_when_no_fence(self) -> None:
        from gpumod.benchmarks.coding.code_extraction import extract_code

        response = "   class Foo:\n    pass   "
        assert extract_code(response) == "class Foo:\n    pass"

    def test_returns_empty_string_for_empty_response(self) -> None:
        from gpumod.benchmarks.coding.code_extraction import extract_code

        assert extract_code("") == ""


class TestExtractCodeDedent:
    """Bug 2: extractor must dedent code wrapped in indented contexts."""

    def test_dedents_python_fence_at_column_4(self) -> None:
        """Gemma 4 emits ```python inside numbered list items at column 4."""
        from gpumod.benchmarks.coding.code_extraction import extract_code

        response = (
            "Here is the plan:\n"
            "1. Write the class:\n"
            "    ```python\n"
            "    class JobQueue:\n"
            "        def __init__(self):\n"
            "            self.jobs = []\n"
            "    ```\n"
        )
        # Without dedent: 'class JobQueue:' at column 4 → IndentationError at top level
        # With dedent: 'class JobQueue:' at column 0 → valid Python
        result = extract_code(response)
        assert result == "class JobQueue:\n    def __init__(self):\n        self.jobs = []"

    def test_dedents_bare_fence_at_column_4(self) -> None:
        from gpumod.benchmarks.coding.code_extraction import extract_code

        response = (
            "  - step 1:\n"
            "    ```\n"
            "    from collections import deque\n"
            "    class X:\n"
            "        pass\n"
            "    ```\n"
        )
        result = extract_code(response)
        assert result == "from collections import deque\nclass X:\n    pass"

    def test_dedent_is_idempotent_on_correctly_formatted_code(self) -> None:
        """Dedent on already-flush code must not change anything."""
        from gpumod.benchmarks.coding.code_extraction import extract_code

        response = "```python\nclass X:\n    def m(self):\n        return 1\n```"
        result = extract_code(response)
        assert result == "class X:\n    def m(self):\n        return 1"


class TestExtractCodeUnwrapsArtifactResponse:
    """Bug 1: extractor must unwrap <reasoning_content>...</content> wrapper.

    The benchmark runner stores level artifacts as
        <reasoning_content>{thinking-trace}</reasoning_content>\\n\\n<content>{final-answer}</content>
    so post-hoc debugging can see both fields. But re-extracting code from
    this wrapper picks up the FIRST fence — which is in reasoning (a draft)
    instead of in content (the polished final code that was actually
    validated). The extractor must unwrap and use the <content> portion.
    """

    def test_unwraps_and_prefers_content_block(self) -> None:
        from gpumod.benchmarks.coding.code_extraction import extract_code

        wrapped = (
            "<reasoning_content>\n"
            "Let me draft this:\n"
            "```python\n"
            "class DraftJobQueue:\n"
            "    pass\n"
            "```\n"
            "Actually, let me reconsider the name.\n"
            "</reasoning_content>\n"
            "\n"
            "<content>\n"
            "```python\n"
            "class FinalJobQueue:\n"
            "    pass\n"
            "```\n"
            "</content>"
        )
        # Without unwrap: extracts DraftJobQueue (first ```python in reasoning)
        # With unwrap: extracts FinalJobQueue (from <content>)
        result = extract_code(wrapped)
        assert result == "class FinalJobQueue:\n    pass"
        assert "Draft" not in result, "must not return reasoning-trace draft"

    def test_unwraps_handles_empty_content_block(self) -> None:
        """When <content> is empty, fall through gracefully (return empty)."""
        from gpumod.benchmarks.coding.code_extraction import extract_code

        wrapped = "<reasoning_content>\nThinking...\n</reasoning_content>\n\n<content>\n</content>"
        # No code block in content → extract_code returns empty/whitespace
        result = extract_code(wrapped)
        assert result == ""

    def test_no_unwrap_when_response_is_not_wrapped(self) -> None:
        """Plain responses (validation path) must work unchanged."""
        from gpumod.benchmarks.coding.code_extraction import extract_code

        response = "```python\nclass Foo:\n    pass\n```"
        assert extract_code(response) == "class Foo:\n    pass"
