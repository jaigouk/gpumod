"""Assemble the chat request for one AgentWorldBench sample.

The model-under-test is asked to predict the next environment observation:
``system_str`` as the system message, the prior turns interleaved as
user(action)/assistant(observation) history, then the current action as the final
user message. The ground-truth observation (``sample.ground_truth``) is deliberately
NOT included — it is what the model must predict and what the judge scores against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpumod.benchmarks.agentworld.models import AgentWorldSample


def build_messages(sample: AgentWorldSample) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": sample.system_str}]
    for action, observation in sample.history:
        messages.append({"role": "user", "content": action})
        messages.append({"role": "assistant", "content": observation})
    messages.append({"role": "user", "content": sample.current_prompt})
    return messages
