"""Consistency-based hallucination detection (SelfCheckGPT approach).

Detects potential hallucinations by measuring response consistency
across multiple runs. If a model knows a fact, it will consistently
produce it; if hallucinating, responses will diverge.

Usage:
    from gpumod.benchmark.consistency import ConsistencyChecker

    checker = ConsistencyChecker(threshold=0.5)

    async def generate(prompt: str) -> str:
        return await call_model(prompt)

    result = await checker.check_with_generator(
        prompt="Who created Python?",
        generator=generate,
        runs=5,
    )

    if result.consistency_score < 0.7:
        print(f"Potential hallucinations: {result.inconsistent_facts}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass
class ConsistencyResult:
    """Result of consistency checking across multiple runs.

    Attributes
    ----------
    runs:
        Number of times the prompt was run.
    responses:
        List of raw responses from each run.
    facts_extracted:
        Set of facts extracted from each response.
    consistency_score:
        Score from 0.0 to 1.0 indicating overall consistency.
        1.0 means all runs produced identical facts.
    inconsistent_facts:
        Facts that appeared in less than threshold% of runs.
    """

    runs: int
    responses: list[str]
    facts_extracted: list[set[str]] = field(default_factory=list)
    consistency_score: float = 1.0
    inconsistent_facts: list[str] = field(default_factory=list)


def extract_facts(response: str) -> set[str]:
    """Extract atomic facts from a response.

    Uses simple sentence splitting and normalization to extract
    comparable fact units from a response.

    Parameters
    ----------
    response:
        The model response to extract facts from.

    Returns
    -------
    set[str]:
        Set of normalized fact strings.
    """
    if not response or not response.strip():
        return set()

    # Normalize whitespace
    text = " ".join(response.split())

    # Split into sentences (simple approach)
    # Handle: periods, question marks, exclamation marks, newlines
    # Also handle bullet points
    sentences = re.split(r"[.!?\n]|\s*[-•]\s*", text)

    facts = set()
    for sentence in sentences:
        # Clean and normalize
        sentence = sentence.strip()
        if not sentence:
            continue

        # Normalize to lowercase for comparison
        normalized = sentence.lower()

        # Remove leading articles/conjunctions for better matching
        normalized = re.sub(r"^(the|a|an|and|but|or|so)\s+", "", normalized)

        # Only keep sentences with substantive content (at least 2 words)
        words = normalized.split()
        if len(words) >= 2:
            facts.add(normalized)

    return facts


def compute_consistency(
    responses: list[str],
    threshold: float = 0.5,
) -> ConsistencyResult:
    """Compute consistency score across multiple responses.

    Parameters
    ----------
    responses:
        List of responses to compare.
    threshold:
        Minimum fraction of runs a fact must appear in to be
        considered consistent. Default 0.5 (50%).

    Returns
    -------
    ConsistencyResult:
        Aggregated consistency metrics.
    """
    runs = len(responses)

    if runs == 0:
        return ConsistencyResult(
            runs=0,
            responses=[],
            facts_extracted=[],
            consistency_score=1.0,
            inconsistent_facts=[],
        )

    # Extract facts from each response
    facts_per_response = [extract_facts(r) for r in responses]

    # Single run = perfect consistency
    if runs == 1:
        return ConsistencyResult(
            runs=1,
            responses=responses,
            facts_extracted=facts_per_response,
            consistency_score=1.0,
            inconsistent_facts=[],
        )

    # Count how often each fact appears
    all_facts: set[str] = set()
    for facts in facts_per_response:
        all_facts.update(facts)

    # If no facts extracted, return neutral result
    if not all_facts:
        return ConsistencyResult(
            runs=runs,
            responses=responses,
            facts_extracted=facts_per_response,
            consistency_score=1.0,  # Nothing to be inconsistent about
            inconsistent_facts=[],
        )

    fact_counts: dict[str, int] = {}
    for fact in all_facts:
        fact_counts[fact] = sum(1 for facts in facts_per_response if fact in facts)

    # Identify inconsistent facts (below threshold)
    inconsistent = [
        fact for fact, count in fact_counts.items() if count / runs < threshold
    ]

    # Compute overall consistency score
    # Score = average appearance rate of all facts
    if fact_counts:
        total_rate = sum(count / runs for count in fact_counts.values())
        consistency_score = total_rate / len(fact_counts)
    else:
        consistency_score = 1.0

    return ConsistencyResult(
        runs=runs,
        responses=responses,
        facts_extracted=facts_per_response,
        consistency_score=round(consistency_score, 4),
        inconsistent_facts=sorted(inconsistent),
    )


class ConsistencyChecker:
    """Checks response consistency across multiple runs.

    Uses the SelfCheckGPT approach: run the same prompt multiple
    times and measure how consistent the extracted facts are.
    Inconsistent facts are potential hallucinations.

    Parameters
    ----------
    threshold:
        Minimum fraction of runs a fact must appear in to be
        considered consistent. Default 0.5 (50%).
    default_runs:
        Default number of runs when not specified. Default 5.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        default_runs: int = 5,
    ) -> None:
        self.threshold = threshold
        self.default_runs = default_runs

    async def check_with_generator(
        self,
        prompt: str,
        generator: Callable[[str], Awaitable[str]],
        runs: int | None = None,
    ) -> ConsistencyResult:
        """Check consistency using a response generator.

        Parameters
        ----------
        prompt:
            The prompt to send to the model.
        generator:
            Async function that takes a prompt and returns a response.
        runs:
            Number of times to run the prompt. Defaults to default_runs.

        Returns
        -------
        ConsistencyResult:
            Consistency metrics across all runs.
        """
        if runs is None:
            runs = self.default_runs

        # Collect responses
        responses = []
        for _ in range(runs):
            response = await generator(prompt)
            responses.append(response)

        return compute_consistency(responses, threshold=self.threshold)
