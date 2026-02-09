"""Tests for benchmark consistency checking (SelfCheckGPT approach).

Follows TDD: tests written first (Red), then implementation (Green).

Phase 1: Consistency-based hallucination detection
- Run prompts multiple times
- Extract facts from responses
- Measure consistency across runs
- Flag inconsistent facts as potential hallucinations
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# ConsistencyResult dataclass tests
# ---------------------------------------------------------------------------


class TestConsistencyResult:
    """Tests for the ConsistencyResult dataclass."""

    def test_creates_with_required_fields(self) -> None:
        """ConsistencyResult captures runs, responses, and scores."""
        from gpumod.benchmark.consistency import ConsistencyResult

        result = ConsistencyResult(
            runs=5,
            responses=["resp1", "resp2", "resp3", "resp4", "resp5"],
            facts_extracted=[
                {"fact1", "fact2"},
                {"fact1"},
                {"fact1", "fact3"},
                {"fact1"},
                {"fact1"},
            ],
            consistency_score=0.8,
            inconsistent_facts=["fact2", "fact3"],
        )

        assert result.runs == 5
        assert len(result.responses) == 5
        assert result.consistency_score == 0.8
        assert "fact2" in result.inconsistent_facts

    def test_perfect_consistency_score(self) -> None:
        """Score is 1.0 when all runs produce identical facts."""
        from gpumod.benchmark.consistency import ConsistencyResult

        result = ConsistencyResult(
            runs=3,
            responses=["same", "same", "same"],
            facts_extracted=[{"fact1", "fact2"}, {"fact1", "fact2"}, {"fact1", "fact2"}],
            consistency_score=1.0,
            inconsistent_facts=[],
        )

        assert result.consistency_score == 1.0
        assert result.inconsistent_facts == []

    def test_serializes_to_dict(self) -> None:
        """ConsistencyResult can be serialized for JSON output."""
        from dataclasses import asdict

        from gpumod.benchmark.consistency import ConsistencyResult

        result = ConsistencyResult(
            runs=3,
            responses=["a", "b", "c"],
            facts_extracted=[{"f1"}, {"f1", "f2"}, {"f1"}],
            consistency_score=0.67,
            inconsistent_facts=["f2"],
        )

        d = asdict(result)
        assert d["runs"] == 3
        assert d["consistency_score"] == 0.67


# ---------------------------------------------------------------------------
# Fact extraction tests
# ---------------------------------------------------------------------------


class TestExtractFacts:
    """Tests for extracting atomic facts from responses."""

    def test_extracts_simple_statements(self) -> None:
        """Extracts sentence-like facts from a response."""
        from gpumod.benchmark.consistency import extract_facts

        response = "Python was created by Guido van Rossum. It was first released in 1991."
        facts = extract_facts(response)

        assert isinstance(facts, set)
        assert len(facts) >= 2
        # Should contain normalized versions of key claims
        assert any("guido" in f.lower() for f in facts)
        assert any("1991" in f for f in facts)

    def test_handles_empty_response(self) -> None:
        """Returns empty set for empty response."""
        from gpumod.benchmark.consistency import extract_facts

        facts = extract_facts("")
        assert facts == set()

    def test_handles_whitespace_only(self) -> None:
        """Returns empty set for whitespace-only response."""
        from gpumod.benchmark.consistency import extract_facts

        facts = extract_facts("   \n\t  ")
        assert facts == set()

    def test_normalizes_facts(self) -> None:
        """Facts are normalized for comparison (lowercase, stripped)."""
        from gpumod.benchmark.consistency import extract_facts

        response1 = "Python was created by Guido."
        response2 = "python was created by guido."

        facts1 = extract_facts(response1)
        facts2 = extract_facts(response2)

        # Normalized facts should be comparable
        assert facts1 == facts2

    def test_extracts_from_bullet_points(self) -> None:
        """Extracts facts from bullet-point formatted responses."""
        from gpumod.benchmark.consistency import extract_facts

        response = """Here are the facts:
- Python was released in 1991
- Java was released in 1995
- Rust was released in 2010
"""
        facts = extract_facts(response)

        assert len(facts) >= 3
        assert any("1991" in f for f in facts)
        assert any("1995" in f for f in facts)

    def test_extracts_numbers_as_facts(self) -> None:
        """Numbers and dates are preserved as part of facts."""
        from gpumod.benchmark.consistency import extract_facts

        response = "The population is 331,000,000. The area is 9.8 million km2."
        facts = extract_facts(response)

        # Should preserve numeric information
        assert any("331" in f or "million" in f.lower() for f in facts)


# ---------------------------------------------------------------------------
# Consistency computation tests
# ---------------------------------------------------------------------------


class TestComputeConsistency:
    """Tests for computing consistency across multiple responses."""

    def test_perfect_consistency(self) -> None:
        """Returns 1.0 when all responses have identical facts."""
        from gpumod.benchmark.consistency import compute_consistency

        responses = [
            "Python was created in 1991 by Guido.",
            "Python was created in 1991 by Guido.",
            "Python was created in 1991 by Guido.",
        ]

        result = compute_consistency(responses)

        assert result.consistency_score == 1.0
        assert result.inconsistent_facts == []

    def test_partial_consistency(self) -> None:
        """Returns score < 1.0 when facts vary across responses."""
        from gpumod.benchmark.consistency import compute_consistency

        responses = [
            "Python was created in 1991 by Guido van Rossum.",
            "Python was created in 1991 by Guido.",
            "Python was created in 1989 by Guido.",  # Wrong year
        ]

        result = compute_consistency(responses)

        assert 0.0 < result.consistency_score < 1.0
        # The inconsistent year should be flagged
        assert len(result.inconsistent_facts) > 0

    def test_zero_consistency(self) -> None:
        """Returns 0.0 when no facts are shared across responses."""
        from gpumod.benchmark.consistency import compute_consistency

        responses = [
            "Apples are red.",
            "Bananas are yellow.",
            "Oranges are orange.",
        ]

        result = compute_consistency(responses)

        # No shared facts = low consistency
        assert result.consistency_score < 0.5

    def test_flags_facts_below_threshold(self) -> None:
        """Facts appearing in <50% of runs are flagged as inconsistent."""
        from gpumod.benchmark.consistency import compute_consistency

        responses = [
            "The capital of France is Paris. The Eiffel Tower is 330m tall.",
            "The capital of France is Paris.",
            "The capital of France is Paris.",
            "The capital of France is Paris.",
            "The capital of France is Paris.",
        ]

        result = compute_consistency(responses)

        # "Paris" appears in all 5 (consistent)
        # "Eiffel Tower" / "330m" appears in 1/5 (20%, inconsistent)
        assert any("eiffel" in f.lower() or "330" in f for f in result.inconsistent_facts)

    def test_handles_single_response(self) -> None:
        """Single response has perfect consistency (nothing to compare)."""
        from gpumod.benchmark.consistency import compute_consistency

        responses = ["Single response with facts."]

        result = compute_consistency(responses)

        # Single run = perfect consistency by definition
        assert result.consistency_score == 1.0
        assert result.runs == 1

    def test_handles_empty_responses(self) -> None:
        """Empty responses return zero consistency."""
        from gpumod.benchmark.consistency import compute_consistency

        responses = ["", "", ""]

        result = compute_consistency(responses)

        # No facts = undefined, treat as 1.0 (nothing to be inconsistent about)
        assert result.runs == 3


# ---------------------------------------------------------------------------
# ConsistencyChecker integration tests
# ---------------------------------------------------------------------------


class TestConsistencyChecker:
    """Tests for the ConsistencyChecker class."""

    def test_has_default_runs(self) -> None:
        """ConsistencyChecker has a sensible default number of runs."""
        from gpumod.benchmark.consistency import ConsistencyChecker

        checker = ConsistencyChecker()
        assert checker.default_runs >= 3

    def test_check_returns_consistency_result(self) -> None:
        """check() method returns a ConsistencyResult."""
        from gpumod.benchmark.consistency import ConsistencyChecker, ConsistencyResult

        # Mock the response generator
        checker = ConsistencyChecker()

        # Create a mock that returns consistent responses
        async def mock_generate(prompt: str) -> str:
            return "Python was created by Guido in 1991."

        result = checker.check_with_generator(
            prompt="Who created Python?",
            generator=mock_generate,
            runs=3,
        )

        # Should return immediately with a coroutine
        import asyncio

        actual_result = asyncio.run(result)

        assert isinstance(actual_result, ConsistencyResult)
        assert actual_result.runs == 3

    @pytest.mark.asyncio
    async def test_check_runs_generator_n_times(self) -> None:
        """check() runs the generator the specified number of times."""
        from gpumod.benchmark.consistency import ConsistencyChecker

        call_count = 0

        async def counting_generator(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Response {call_count}"

        checker = ConsistencyChecker()
        await checker.check_with_generator(
            prompt="Test prompt",
            generator=counting_generator,
            runs=5,
        )

        assert call_count == 5

    @pytest.mark.asyncio
    async def test_check_collects_all_responses(self) -> None:
        """check() collects all responses in the result."""
        from gpumod.benchmark.consistency import ConsistencyChecker

        responses_returned = ["First", "Second", "Third"]
        idx = 0

        async def sequential_generator(prompt: str) -> str:
            nonlocal idx
            resp = responses_returned[idx]
            idx += 1
            return resp

        checker = ConsistencyChecker()
        result = await checker.check_with_generator(
            prompt="Test",
            generator=sequential_generator,
            runs=3,
        )

        assert result.responses == responses_returned


# ---------------------------------------------------------------------------
# Threshold configuration tests
# ---------------------------------------------------------------------------


class TestConsistencyThreshold:
    """Tests for configurable consistency thresholds."""

    def test_default_threshold_is_50_percent(self) -> None:
        """Default threshold for flagging inconsistent facts is 50%."""
        from gpumod.benchmark.consistency import ConsistencyChecker

        checker = ConsistencyChecker()
        assert checker.threshold == 0.5

    def test_custom_threshold(self) -> None:
        """Threshold can be customized."""
        from gpumod.benchmark.consistency import ConsistencyChecker

        checker = ConsistencyChecker(threshold=0.7)
        assert checker.threshold == 0.7

    def test_threshold_affects_inconsistent_facts(self) -> None:
        """Higher threshold flags more facts as inconsistent."""
        from gpumod.benchmark.consistency import compute_consistency

        responses = [
            "Fact A. Fact B. Fact C.",
            "Fact A. Fact B.",
            "Fact A. Fact B.",
            "Fact A.",
        ]

        # With 50% threshold: A appears in 4/4 (100%), B in 3/4 (75%), C in 1/4 (25%)
        # Only C should be flagged
        result_50 = compute_consistency(responses, threshold=0.5)

        # With 80% threshold: A (100%) ok, B (75%) flagged, C (25%) flagged
        result_80 = compute_consistency(responses, threshold=0.8)

        assert len(result_80.inconsistent_facts) >= len(result_50.inconsistent_facts)
