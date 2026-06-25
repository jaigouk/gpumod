"""Data model for one AgentWorldBench sample.

Schema (Qwen/AgentWorldBench, verified 2026-06-25): task, id, prompt[], response[],
current_prompt, system_str, turn_idx, total_turns. ``turn_idx`` is 1-indexed; the
``prompt``/``response`` lists cover turns 1..turn_idx, so the turn under evaluation is
at index ``turn_idx - 1`` and the prior turns are the interaction history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentWorldSample:
    task: str
    id: Any
    prompt: list[str]
    response: list[str]
    current_prompt: str
    system_str: str
    turn_idx: int
    total_turns: int

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> AgentWorldSample:
        return cls(
            task=row["task"],
            id=row.get("id"),
            prompt=list(row["prompt"]),
            response=list(row["response"]),
            current_prompt=row["current_prompt"],
            system_str=row["system_str"],
            turn_idx=int(row["turn_idx"]),
            total_turns=int(row["total_turns"]),
        )

    @property
    def ground_truth(self) -> str:
        """The real environment observation for the current turn (what we score against)."""
        return self.response[self.turn_idx - 1]

    @property
    def history(self) -> list[tuple[str, str]]:
        """(action, observation) pairs for turns 1..turn_idx-1 (excludes the current turn)."""
        n = self.turn_idx - 1
        return list(zip(self.prompt[:n], self.response[:n], strict=False))
