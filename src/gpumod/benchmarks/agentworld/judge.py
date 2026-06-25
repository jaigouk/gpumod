"""Reference-grounded Claude judge for AgentWorldBench (gpumod-kpmq.1).

Scores a predicted environment observation against the ground truth on five
dimensions (1-5) by calling **`claude -p`** — the Claude Code CLI in headless mode,
which authenticates with the Claude Max subscription (no API key, no proxy). See
``.notes/claude-judge-no-apikey-poc.md``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile

from gpumod.benchmarks.agentworld.scorer import DIMENSIONS, DimensionScores

# `claude -p --model` value (CLI alias resolved against the subscription default).
DEFAULT_JUDGE_MODEL = "opus"

_RUBRIC = """You are a strict, reference-grounded judge for an environment world model \
in the "{domain}" domain. Compare the PREDICTED next environment observation against the \
GROUND-TRUTH observation and score each dimension 1-5 (1=very poor, 5=excellent):
- format: structural compliance with the expected {domain} output shape
- factuality: correctness of stated facts vs the ground truth
- consistency: internal and cross-turn coherence
- realism: behavioural alignment with a real {domain} environment
- quality: completeness without needless verbosity
Deterministic content must match the ground truth; runtime noise (timestamps, PIDs) \
needs only a plausible format. Return ONLY a JSON object, no prose, no markdown fences:
{{"format":N,"factuality":N,"consistency":N,"realism":N,"quality":N,"rationale":"<=1 sentence"}}

[GROUND TRUTH]
{truth}

[PREDICTED]
{pred}
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_judge_prompt(domain: str, prediction: str, ground_truth: str) -> str:
    return _RUBRIC.format(domain=domain, truth=ground_truth, pred=prediction)


def parse_judge_response(text: str) -> DimensionScores:
    """Parse a judge response (tolerating prose/markdown fences) into DimensionScores.

    Raises ValueError if no JSON object is found or any of the five dimensions is missing.
    Out-of-range values are clamped to 1-5.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    match = _JSON_RE.search(cleaned)
    if match is None:
        raise ValueError(f"no JSON object in judge response: {text[:200]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge response is not valid JSON: {exc}") from exc

    scores: dict[str, int] = {}
    for dim in DIMENSIONS:
        if dim not in data:
            raise ValueError(f"judge response missing dimension {dim!r}: {data}")
        value = round(float(data[dim]))
        scores[dim] = max(1, min(5, value))
    return DimensionScores(**scores)


class ClaudeJudge:
    """Scores one prediction by invoking ``claude -p`` (subscription, no API key)."""

    def __init__(self, model: str | None = None, timeout: float = 180.0) -> None:
        self._model = model or os.environ.get("CLAUDE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        self._timeout = timeout
        # Run from a neutral cwd so the CLI does not load this project's context.
        self._cwd = tempfile.gettempdir()

    def score(self, domain: str, prediction: str, ground_truth: str) -> DimensionScores:
        prompt = build_judge_prompt(domain, prediction, ground_truth)
        cmd = ["claude", "-p", prompt]
        if self._model:
            cmd += ["--model", self._model]
        proc = subprocess.run(  # noqa: S603 - trusted local CLI, subscription auth
            cmd,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            cwd=self._cwd,
            timeout=self._timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"`claude -p` failed (rc={proc.returncode}): {proc.stderr[:300]}")
        return parse_judge_response(proc.stdout)
