"""RLM orchestrator — manages the consult lifecycle.

Creates a GpumodConsultEnv, wires it to an RLM client/handler, and
runs the iterative code-execution loop. Returns a structured
ConsultResult.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field
from rlm.clients import BaseLM, get_client
from rlm.core.lm_handler import LMHandler
from rlm.utils.parsing import find_code_blocks, find_final_answer

if TYPE_CHECKING:
    from collections.abc import Callable

    from rlm.core.types import ClientBackend

from gpumod.rlm.environment import GpumodConsultEnv
from gpumod.rlm.prompts import GPUMOD_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Absolute cap on iterations regardless of caller request.
MAX_ITERATIONS_HARD_LIMIT = 10
DEFAULT_MAX_ITERATIONS = 5


class ConsultResult(BaseModel):
    """Structured result of an RLM consultation."""

    can_run: bool | None = None
    recommendation: str = ""
    reasoning_steps: list[str] = Field(default_factory=list)
    suggested_commands: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    turns_used: int = 0
    incomplete: bool = False


def _parse_final_answer(raw: str) -> ConsultResult:
    """Best-effort parse of the FINAL(...) payload into ConsultResult."""
    # Try JSON parse first.
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return ConsultResult(
                can_run=data.get("can_run"),
                recommendation=data.get("recommendation", raw),
                reasoning_steps=data.get("reasoning_steps", []),
                suggested_commands=data.get("suggested_commands", []),
                sources=data.get("sources", []),
            )
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting JSON from within the string (FINAL may include prose).
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, dict):
                return ConsultResult(
                    can_run=data.get("can_run"),
                    recommendation=data.get("recommendation", raw),
                    reasoning_steps=data.get("reasoning_steps", []),
                    suggested_commands=data.get("suggested_commands", []),
                    sources=data.get("sources", []),
                )
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: treat the whole answer as the recommendation.
    return ConsultResult(recommendation=raw)


class RLMOrchestrator:
    """Manages the RLM consulting lifecycle.

    Creates a :class:`GpumodConsultEnv`, wires it to an LLM client via
    RLM's ``LMHandler``, and runs the iterative REPL loop. The caller
    provides *tool_wrappers* — synchronous callables that mirror MCP
    tool signatures.

    Parameters
    ----------
    tool_wrappers:
        Mapping of tool name → callable. Only whitelisted tools are
        injected into the environment.
    backend:
        LLM backend identifier (``"anthropic"`` or ``"openai"``).
    model:
        Model name override. Falls back to env-var defaults.
    """

    _DEFAULT_MODELS: ClassVar[dict[str, str]] = {
        "anthropic": "claude-sonnet-4-5-20250929",
        "openai": "gpt-4o-mini",
    }

    def __init__(
        self,
        tool_wrappers: dict[str, Callable[..., Any]],
        backend: ClientBackend = "anthropic",
        model: str | None = None,
    ) -> None:
        self._tool_wrappers = tool_wrappers
        self._backend = backend
        self._model = model or self._resolve_model(backend)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consult(
        self,
        query: str,
        *,
        max_turns: int = DEFAULT_MAX_ITERATIONS,
        context: dict[str, Any] | None = None,
    ) -> ConsultResult:
        """Run an RLM consultation and return a structured result.

        Parameters
        ----------
        query:
            User question (e.g. "Can I run Qwen3-235B on 24 GB?").
        max_turns:
            Maximum REPL iterations before forcing a final answer.
        context:
            Optional pre-fetched data to inject as ``context`` variable.
        """
        max_turns = min(max_turns, MAX_ITERATIONS_HARD_LIMIT)

        # Build context payload.
        payload: dict[str, Any] = {"query": query}
        if context:
            payload["data"] = context

        # Create environment.
        env = GpumodConsultEnv(
            tool_wrappers=self._tool_wrappers,
            context_payload=payload,
        )

        # Create LLM client + handler.
        backend_kwargs = {"model_name": self._model}
        client: BaseLM = get_client(self._backend, backend_kwargs)
        handler = LMHandler(client)
        handler.start()

        try:
            result = self._run_loop(env, handler, query, max_turns)
        finally:
            handler.stop()
            env.cleanup()

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(
        self,
        env: GpumodConsultEnv,
        handler: LMHandler,
        query: str,
        max_turns: int,
    ) -> ConsultResult:
        """Core REPL loop mirroring RLM.completion() logic."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": GPUMOD_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Answer this query: {query}\n\n"
                    "Start by examining the `context` variable, then call "
                    "tool functions to gather data."
                ),
            },
        ]

        for turn in range(max_turns):
            response: str = handler.completion(messages)

            # Check for final answer.
            final_raw = find_final_answer(response, environment=env)
            if final_raw is not None:
                result = _parse_final_answer(final_raw)
                result.turns_used = turn + 1
                return result

            # Parse and execute code blocks.
            code_blocks = find_code_blocks(response)
            messages.append({"role": "assistant", "content": response})

            for code_str in code_blocks:
                repl_result = env.execute_code(code_str)
                output_parts: list[str] = []
                if repl_result.stdout:
                    output_parts.append(repl_result.stdout)
                if repl_result.stderr:
                    output_parts.append(f"STDERR: {repl_result.stderr}")
                output = "\n".join(output_parts) or "No output"
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Code executed:\n```python\n{code_str}\n```\n\nREPL output:\n{output}"
                        ),
                    }
                )

            # If no code blocks were found, nudge the model.
            if not code_blocks:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue exploring via ```repl``` code blocks, "
                            "or provide your FINAL(...) answer."
                        ),
                    }
                )

        # Exhausted turns — ask for a forced final answer.
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have used all available turns. Please provide your "
                    "FINAL(...) answer now based on what you have gathered."
                ),
            }
        )
        response = handler.completion(messages)
        final_raw = find_final_answer(response, environment=env)
        if final_raw is not None:
            result = _parse_final_answer(final_raw)
            result.turns_used = max_turns + 1
            return result

        # Truly could not extract a structured answer.
        return ConsultResult(
            recommendation=response,
            turns_used=max_turns + 1,
            incomplete=True,
        )

    @classmethod
    def _resolve_model(cls, backend: ClientBackend) -> str:
        env_key = f"GPUMOD_RLM_{backend.upper()}_MODEL"
        return os.environ.get(env_key, cls._DEFAULT_MODELS.get(backend, ""))
