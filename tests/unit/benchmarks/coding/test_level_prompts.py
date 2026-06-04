"""Tests that lock in key properties of level prompts (gpumod-7vy8).

These tests guard against accidental prompt regressions that would reintroduce
distractor language. They do NOT assert exact prompt text — only the presence
or absence of specific tokens that affect model behavior on the 12B-class
dense models (which have no parameter headroom to absorb prompt noise).
"""

from __future__ import annotations


class TestLevel2PromptStripsDistractors:
    """L2 prompt must not contain distractor language found to derail 12B dense models.

    Diagnosed in gpumod-h6gs run 2 artifacts:
      - `requests` / `requests.get` in the example → iter_06 degenerated into
        "If __ename__ import requests" repetition loop
      - `url` in the example data shape → models infer wrong wrappers around
        `data`, leading to extra parens in `processor(self.jobs[job_id]["data"]))`
      - "retry up to 3 times" alone is ambiguous → models spend tokens deriving
        whether that means 3 attempts or 3 retries (4 attempts)
    """

    def test_l2_prompt_does_not_mention_requests(self) -> None:
        from gpumod.benchmarks.coding.levels import get_level

        prompt = get_level(2).prompt
        assert "requests" not in prompt, (
            "L2 prompt must not mention `requests` — gpumod-h6gs iter_06 "
            "showed the 12B Q8 degenerating into 'import requests' repetition"
        )
        assert "requests.get" not in prompt
        assert "fetch_url" not in prompt

    def test_l2_prompt_does_not_use_url_data_shape(self) -> None:
        """The tests use {'value': N} and {}, not {'url': '...'} — match that."""
        from gpumod.benchmarks.coding.levels import get_level

        prompt = get_level(2).prompt
        assert '"url"' not in prompt
        assert "'url'" not in prompt

    def test_l2_prompt_disambiguates_retry_count(self) -> None:
        """The prompt must specify "4 total attempts" (initial + 3 retries) to
        eliminate the ambiguity in "retry up to 3 times" that triggered
        backoff-math digressions in iter_02 (~30% of token budget burned)."""
        from gpumod.benchmarks.coding.levels import get_level

        prompt = get_level(2).prompt
        # Either "4 total attempts" or "4 attempts" or "initial call plus 3 retries"
        # — accept any of these explicit disambiguations
        disambiguators = [
            "4 total attempts",
            "4 attempts",
            "initial call plus 3 retries",
            "(initial + 3 retries)",
            "plus 3 retries",
        ]
        assert any(d in prompt for d in disambiguators), (
            f"L2 prompt must contain one of {disambiguators} to disambiguate the retry count"
        )

    def test_l2_prompt_forbids_external_imports(self) -> None:
        """Explicitly close the rabbit hole of importing requests/aiohttp/etc."""
        from gpumod.benchmarks.coding.levels import get_level

        prompt = get_level(2).prompt
        forbidders = [
            "do not import external",
            "do not import",
            "only the standard library",
            "no external libraries",
            "no external packages",
            "no external imports",
        ]
        prompt_lower = prompt.lower()
        assert any(f in prompt_lower for f in forbidders), (
            f"L2 prompt must explicitly forbid external imports (any of {forbidders})"
        )

    def test_l2_prompt_still_requires_process_job_signature(self) -> None:
        """Existing-behavior regression — must still ask for the method."""
        from gpumod.benchmarks.coding.levels import get_level

        prompt = get_level(2).prompt
        assert "process_job" in prompt
        assert "Callable" in prompt
        assert "bool" in prompt


class TestLevel5PromptRequiresComposition:
    """L5 prompt must ask for a meaningful composition task, not a trivial one.

    Diagnosed in gpumod-pdtn: the prior L5 asked for multi-file refactor but
    the PytestValidator only writes a single solution.py, and the L5 tests
    only checked `from solution import JobQueue/Job`. Trivially-passable.

    New L5 asks for three composable classes (Job dataclass, RetryPolicy
    class, JobQueue orchestrator) within a single file.
    """

    def test_l5_prompt_requires_three_classes(self) -> None:
        from gpumod.benchmarks.coding.levels import get_level

        prompt = get_level(5).prompt
        assert "Job" in prompt
        assert "RetryPolicy" in prompt
        assert "JobQueue" in prompt

    def test_l5_prompt_specifies_single_file_target(self) -> None:
        """The validator only saves one file — prompt must match."""
        from gpumod.benchmarks.coding.levels import get_level

        prompt = get_level(5).prompt
        prompt_lower = prompt.lower()
        # Either "single file" or an explicit "solution.py" reference
        assert any(phrase in prompt_lower for phrase in ["single file", "solution.py"])

    def test_l5_prompt_does_not_ask_for_multi_file(self) -> None:
        """Prior 'queue/core.py + queue/retry.py + queue/priority.py' framing
        was incompatible with the single-file validator. Must not return."""
        from gpumod.benchmarks.coding.levels import get_level

        prompt = get_level(5).prompt
        forbidden = ["queue/__init__.py", "queue/core.py", "queue/retry.py", "queue/priority.py"]
        for f in forbidden:
            assert f not in prompt, f"L5 prompt must not request multi-file path {f!r}"

    def test_l5_prompt_forbids_external_imports(self) -> None:
        from gpumod.benchmarks.coding.levels import get_level

        prompt = get_level(5).prompt
        forbidders = ["only the standard library", "no external libraries", "no external packages"]
        prompt_lower = prompt.lower()
        assert any(f in prompt_lower for f in forbidders)
