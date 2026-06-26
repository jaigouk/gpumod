"""Tests for shared benchmark response normalizers (gpumod-nor9).

RED-first: these characterize the extraction logic that previously lived
inline in ``scripts/run_qwen36_benchmark.py:471-477`` so the refactor that
moves it into ``ResponseNormalizer`` adapters cannot drift.
"""

from __future__ import annotations

import re
import time

import pytest

from gpumod.benchmarks.normalizers import (
    CodeAnswerNormalizer,
    ResponseNormalizer,
    TextAnswerNormalizer,
    _strip_think_blocks,
)


def _legacy_regex_strip(text: str) -> str:
    """The original backtracking-regex implementation (gpumod-nor9).

    Kept here in the test ONLY, as the reference oracle. The production
    ``_strip_think_blocks`` was rewritten to a linear scanner to remove a
    quadratic ReDoS; this oracle proves the rewrite is byte-for-byte
    behavior-preserving.
    """
    return re.compile(r"<think>.*?</think>", re.DOTALL).sub("", text).strip()


class TestStripThinkBlocks:
    """``_strip_think_blocks`` removes only CLOSED <think> spans."""

    def test_closed_block_removed(self) -> None:
        assert _strip_think_blocks("<think>plan</think>answer") == "answer"

    def test_unclosed_block_left_intact(self) -> None:
        # Budget-exhausted case: an unterminated <think> must survive so the
        # reasoning-fallback path can still mine it for a last-resort draft.
        text = "<think>still thinking when budget ran out"
        assert _strip_think_blocks(text) == text

    def test_multiple_blocks_removed(self) -> None:
        text = "<think>a</think>X<think>b</think>Y"
        assert _strip_think_blocks(text) == "XY"

    def test_no_think_content_unchanged(self) -> None:
        # No-op for models that route thinking to reasoning_content (e.g. Gemma).
        assert _strip_think_blocks("just a plain answer") == "just a plain answer"

    def test_dotall_across_newlines(self) -> None:
        text = "<think>line one\nline two\nline three</think>final"
        assert _strip_think_blocks(text) == "final"

    def test_result_is_stripped(self) -> None:
        assert _strip_think_blocks("  <think>x</think>  answer  ") == "answer"


# Inputs exercising every branch where the linear scanner must agree with the
# original backtracking regex (gpumod-nor9 security rewrite). Each is fed to
# both `_strip_think_blocks` and `_legacy_regex_strip` and asserted equal.
_EQUIVALENCE_INPUTS = [
    "",  # empty string
    "no tags at all",  # no-tags text
    "<think>plan</think>answer",  # closed block
    "<think>still thinking, no close",  # unclosed <think>
    "<think>a</think>X<think>b</think>Y",  # multiple blocks
    "<think>outer<think>inner</think>tail",  # nested opens, one close
    "<think>a<think>b</think>c</think>d",  # nested, two closes
    "<think>line one\nline two\nline three</think>final",  # DOTALL newlines
    "</think>orphan close before any open",  # </think> before any <think>
    "<think>```python\ndraft=0\n```</think>```python\nreal=1\n```",  # draft+real fence
    "  <think>x</think>  answer  ",  # leading/trailing whitespace -> strip
    "</think>" + "<think>" * 50,  # close-then-many-bare-opens tail (ReDoS shape)
    "<think></think>",  # empty closed block
    "text<think>mid</think>more<think>unterminated tail",  # mixed closed + open
]


class TestStripThinkBlocksEquivalence:
    """The linear scanner is byte-for-byte equal to the legacy regex."""

    @pytest.mark.parametrize("text", _EQUIVALENCE_INPUTS)
    def test_matches_legacy_regex(self, text: str) -> None:
        assert _strip_think_blocks(text) == _legacy_regex_strip(text)


class TestStripThinkBlocksDosBound:
    """Degenerate bare-open inputs must NOT trigger quadratic backtracking."""

    def test_many_bare_opens_linear(self) -> None:
        # 30k unterminated <think> with no close: nothing is removed, so the
        # result is the input (already has no surrounding whitespace).
        payload = "<think>" * 30000
        t0 = time.perf_counter()
        result = _strip_think_blocks(payload)
        elapsed = time.perf_counter() - t0
        assert result == payload
        # Generous CI-safe ceiling; the old regex took ~21s here.
        assert elapsed < 1.0

    def test_orphan_close_then_many_opens_linear(self) -> None:
        # A leading </think> with no matching open, then 30k bare opens: the
        # crafted-tail shape that defeats a naive `if "</think>" not in text`
        # guard. Nothing closes after the first orphan, so nothing is removed.
        payload = "</think>" + "<think>" * 30000
        t0 = time.perf_counter()
        result = _strip_think_blocks(payload)
        elapsed = time.perf_counter() - t0
        assert result == payload
        assert elapsed < 1.0


class TestCodeAnswerNormalizer:
    """Reproduces run_qwen36_benchmark.py:471-477 byte-for-byte."""

    def test_fence_in_content_returns_stripped_content(self) -> None:
        content = "Here:\n```python\nx = 1\n```"
        result = CodeAnswerNormalizer().extract_answer(content, "some reasoning")
        assert result == "Here:\n```python\nx = 1\n```"

    def test_no_fence_with_reasoning_concatenates(self) -> None:
        content = "no code here"
        reasoning = "```python\ny = 2\n```"
        result = CodeAnswerNormalizer().extract_answer(content, reasoning)
        assert result == reasoning + "\n\n" + content

    def test_no_fence_empty_reasoning_returns_content(self) -> None:
        result = CodeAnswerNormalizer().extract_answer("no code", "")
        assert result == "no code"

    def test_no_fence_none_reasoning_returns_content(self) -> None:
        result = CodeAnswerNormalizer().extract_answer("no code", None)
        assert result == "no code"

    def test_draft_fence_inside_think_is_stripped_and_not_picked(self) -> None:
        # A draft ```fence inside <think> must be stripped; with no real fence
        # after </think>, the reasoning fallback applies.
        content = "<think>```python\ndraft = 0\n```</think>final prose"
        reasoning = "```python\nreal = 1\n```"
        result = CodeAnswerNormalizer().extract_answer(content, reasoning)
        # stripped content = "final prose" (no fence) -> reasoning + content
        assert result == reasoning + "\n\nfinal prose"

    def test_real_fence_after_think_is_picked(self) -> None:
        content = "<think>```python\ndraft = 0\n```</think>```python\nreal = 1\n```"
        result = CodeAnswerNormalizer().extract_answer(content, "ignored reasoning")
        assert result == "```python\nreal = 1\n```"


class TestTextAnswerNormalizer:
    """Strip <think>; fall back to reasoning only when stripped content empty."""

    def test_content_text_after_strip_returns_stripped(self) -> None:
        content = "<think>plan</think>The capital is Paris."
        result = TextAnswerNormalizer().extract_answer(content, "reasoning text")
        assert result == "The capital is Paris."

    def test_all_think_empty_after_strip_falls_back_to_reasoning(self) -> None:
        content = "<think>everything was thinking</think>"
        reasoning = "the real answer"
        result = TextAnswerNormalizer().extract_answer(content, reasoning)
        assert result == "the real answer"

    def test_empty_content_no_reasoning_returns_empty(self) -> None:
        assert TextAnswerNormalizer().extract_answer("", None) == ""

    def test_empty_content_empty_reasoning_returns_empty(self) -> None:
        assert TextAnswerNormalizer().extract_answer("", "") == ""


# --- Characterization: the literal :471-477 algorithm on captured tuples ---


def _legacy_code_extract(response: str, reasoning: str) -> str:
    """The EXACT pre-refactor algorithm (run_qwen36_benchmark.py:471-477).

    ``reasoning`` is a str; "" is falsy -> treated as absent.
    """
    content_final = _strip_think_blocks(response)
    if "```" in content_final:
        extract_source = content_final
    elif reasoning:
        extract_source = reasoning + "\n\n" + content_final
    else:
        extract_source = content_final
    return extract_source


class TestCodeAnswerNormalizerCharacterization:
    """Byte-for-byte equality with the legacy inline algorithm."""

    # (a) Gemma-style: content carries a fence, reasoning routed separately.
    GEMMA_CONTENT = "Solution:\n```python\ndef add(a, b):\n    return a + b\n```"
    GEMMA_REASONING = "Let me think about addition step by step."

    # (b) VibeThinker-style: a draft fence INSIDE <think>, real fence after.
    VIBE_CONTENT = (
        "<think>Draft:\n```python\ndef f():\n    pass\n```\nrefine it</think>"
        "```python\ndef f():\n    return 42\n```"
    )
    VIBE_REASONING = ""

    # (c) No-fence fallback: content has no fence, reasoning present.
    FALLBACK_CONTENT = "I could not finish the code."
    FALLBACK_REASONING = "```python\nx = 1\n```"

    def test_gemma_reasoning_routing_case(self) -> None:
        result = CodeAnswerNormalizer().extract_answer(self.GEMMA_CONTENT, self.GEMMA_REASONING)
        assert result == _legacy_code_extract(self.GEMMA_CONTENT, self.GEMMA_REASONING)
        # Literal expected output (byte-for-byte):
        assert result == "Solution:\n```python\ndef add(a, b):\n    return a + b\n```"

    def test_vibethinker_think_in_content_case(self) -> None:
        result = CodeAnswerNormalizer().extract_answer(self.VIBE_CONTENT, self.VIBE_REASONING)
        assert result == _legacy_code_extract(self.VIBE_CONTENT, self.VIBE_REASONING)
        # The draft fence inside <think> is stripped; real fence after wins.
        assert result == "```python\ndef f():\n    return 42\n```"

    def test_no_fence_fallback_case(self) -> None:
        result = CodeAnswerNormalizer().extract_answer(
            self.FALLBACK_CONTENT, self.FALLBACK_REASONING
        )
        assert result == _legacy_code_extract(self.FALLBACK_CONTENT, self.FALLBACK_REASONING)
        # Literal: reasoning + "\n\n" + stripped content.
        assert result == "```python\nx = 1\n```\n\nI could not finish the code."


class TestProtocolConformance:
    """Both normalizers structurally satisfy ResponseNormalizer."""

    def test_normalizers_satisfy_protocol(self) -> None:
        def takes_normalizer(n: ResponseNormalizer) -> str:
            return n.extract_answer("x", None)

        assert takes_normalizer(CodeAnswerNormalizer()) == "x"
        assert takes_normalizer(TextAnswerNormalizer()) == "x"
