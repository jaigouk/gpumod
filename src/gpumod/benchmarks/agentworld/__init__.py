"""AgentWorldBench world-modeling eval (gpumod-kpmq.1).

Measures how well a model predicts the next environment observation given a
system prompt, interaction history, and a current action — the AgentWorld task —
scored by a reference-grounded LLM judge on five dimensions (Format, Factuality,
Consistency, Realism, Quality). Runs alongside the L1-L5 coding suite.

Dataset: ``Qwen/AgentWorldBench`` (2170 samples, 7 domains).
"""

from __future__ import annotations
