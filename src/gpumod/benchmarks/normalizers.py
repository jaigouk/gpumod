"""Shared response normalizers for benchmark suites (gpumod-nor9).

A ``ResponseNormalizer`` is the adapter seam between a served model's raw
chat-completion output (``message.content`` + optional
``message.reasoning_content``) and the single answer string a benchmark
scores. Different model architectures stream their final answer in different
places — Qwen3.6/Gemma route chain-of-thought to ``reasoning_content`` while
Qwen2-template reasoning models (VibeThinker-3B) emit ``<think>...</think>``
inline in ``content`` — so each architecture family gets its own normalizer
rather than every suite re-implementing the routing logic.

Adding a NEW architecture = one isolated, unit-tested normalizer here.
Adding a KNOWN-architecture model = one registry entry with the matching
normalizer (or ``None`` for the suite default).
"""

from __future__ import annotations

from typing import Protocol

# gpumod-msy8: VibeThinker-3B (and other Qwen2-template reasoning models) emit
# <think>...</think> reasoning into message.content. Their plain ChatML template
# is a "content-only" chat format in llama.cpp, so the reasoning is NOT split
# into reasoning_content even with --reasoning-format deepseek. The model often
# drafts ```python fences INSIDE <think>, so extract_code (first-fence-wins)
# would validate the draft instead of the post-</think> final answer. We strip
# CLOSED <think> blocks from content before the fence search. An unterminated
# <think> (budget exhausted mid-thought) is left intact so the reasoning-fallback
# path can still mine it for a last-resort draft. No-op for models whose content
# has no <think> (e.g. Gemma, which routes thinking to reasoning_content).
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _strip_think_blocks(text: str) -> str:
    """Remove complete <think>...</think> spans from a model response.

    Linear-time scanner (not a backtracking regex): a response with many
    unterminated <think> opens and no </think> is O(n) here, where the prior
    ``re.sub(r"<think>.*?</think>", ...)`` was O(n^2) (gpumod-nor9 security
    review — a degenerate model emitting bare <think> tokens burned ~20s of CPU
    per call inside the regex). Semantics are identical to that regex: leftmost
    <think>, nearest following </think>, that closed span removed; an unclosed
    <think> and everything after it left intact.
    """
    out: list[str] = []
    i = 0
    while True:
        start = text.find(_THINK_OPEN, i)
        if start == -1:
            out.append(text[i:])
            break
        end = text.find(_THINK_CLOSE, start + len(_THINK_OPEN))
        if end == -1:
            out.append(text[i:])  # unclosed — leave the rest intact
            break
        out.append(text[i:start])
        i = end + len(_THINK_CLOSE)
    return "".join(out).strip()


class ResponseNormalizer(Protocol):
    """Adapter that distills a raw model response into a single answer string."""

    def extract_answer(self, content: str, reasoning_content: str | None) -> str:
        """Return the answer to score from ``content`` + optional reasoning.

        ``reasoning_content`` is treated as absent when it is ``""`` or ``None``.
        """
        ...


class CodeAnswerNormalizer:
    """Coding-suite normalizer (run_qwen36_benchmark.py:471-477, verbatim).

    Prefer the model's final ``content`` (the polished answer). Strip closed
    ``<think>`` blocks first so a draft fence inside the reasoning trace is not
    mistaken for the final fenced code. Only fall back to ``reasoning_content``
    when the stripped content has no code fence at all (the model burned its
    budget thinking without reaching final code).
    """

    def extract_answer(self, content: str, reasoning_content: str | None) -> str:
        content_final = _strip_think_blocks(content)
        if "```" in content_final:
            return content_final
        if reasoning_content:  # "" or None -> absent
            return reasoning_content + "\n\n" + content_final
        return content_final


class TextAnswerNormalizer:
    """Free-text normalizer (e.g. AgentWorldBench world-model predictions).

    Strip closed ``<think>`` blocks; if the stripped content is empty AND
    reasoning is present, score the reasoning instead (the model spent its
    whole budget thinking). Otherwise score the stripped content.
    """

    def extract_answer(self, content: str, reasoning_content: str | None) -> str:
        content_final = _strip_think_blocks(content)
        if not content_final and reasoning_content:  # "" or None -> absent
            return reasoning_content
        return content_final
