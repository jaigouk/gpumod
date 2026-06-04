"""Code extraction from model responses (gpumod-9ial).

Two fixes vs the script-embedded predecessor in scripts/run_qwen36_benchmark.py:

1. Unwrap the legacy artifact wrapper. When the benchmark runner stores level
   artifacts, it wraps reasoning + content as
       <reasoning_content>{thinking-trace}</reasoning_content>\\n\\n<content>{final}</content>
   so post-hoc debugging can see both fields. But re-extraction from this
   wrapper picks up the FIRST fence — which is in reasoning (a draft) instead
   of in content (the polished final code that was actually validated). The
   extractor now unwraps and uses the <content> portion. Validation path is
   unaffected because it passes the raw response/extract_source (not wrapped).

2. Dedent the extracted code block. Gemma 4 (and other chat-template-driven
   models) frequently emit ```python fences inside numbered list items at
   column 4+. Without dedent, the extracted code has every line indented at
   column 4 and top-level imports/classes raise IndentationError.

Pure function, no I/O. Safe to use everywhere the script previously called
`_extract_code`.
"""

from __future__ import annotations

import textwrap

_REASONING_OPEN = "<reasoning_content>"
_CONTENT_OPEN = "<content>"
_CONTENT_CLOSE = "</content>"


def _unwrap_artifact_response(response: str) -> str:
    """If response is the legacy <reasoning_content>...<content> wrapper,
    return just the <content> portion. Otherwise return response unchanged.
    """
    if not response.startswith(_REASONING_OPEN):
        return response
    idx = response.find(_CONTENT_OPEN)
    if idx < 0:
        return response
    inner = response[idx + len(_CONTENT_OPEN) :]
    close = inner.rfind(_CONTENT_CLOSE)
    if close >= 0:
        inner = inner[:close]
    return inner


def extract_code(response: str) -> str:
    """Extract Python code from a model response.

    Handles three response shapes (in order):

    1. Legacy artifact wrapper ``<reasoning_content>...</reasoning_content>``
       ``<content>...</content>`` — unwrap to ``<content>`` first so re-extraction
       does not pick up a draft from reasoning.
    2. Fenced ``\\`\\`\\`python ... \\`\\`\\``` — extract the first python block.
    3. Bare fenced ``\\`\\`\\` ... \\`\\`\\``` — extract the block; drop a non-Python
       header line if present.
    4. Unfenced — return the stripped response.

    Always applies ``textwrap.dedent`` to the extracted code so fences emitted
    inside indented contexts (e.g. numbered list items in a chat template)
    parse at column 0.
    """
    if not response:
        return ""

    response = _unwrap_artifact_response(response)

    if "```python" in response:
        parts = response.split("```python")
        if len(parts) > 1:
            code_part = parts[1].split("```")[0]
            return textwrap.dedent(code_part).strip()

    if "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            code_part = parts[1]
            # Dedent first while indentation is intact across all lines.
            # `code_part.strip()` before dedent would erase the leading
            # whitespace on line 1 and break dedent's common-prefix search.
            dedented = textwrap.dedent(code_part)
            lines = dedented.strip().split("\n")
            if lines and not lines[0].startswith(("def ", "class ", "import ", "from ")):
                lines = lines[1:]
            return "\n".join(lines).strip()

    return response.strip()
