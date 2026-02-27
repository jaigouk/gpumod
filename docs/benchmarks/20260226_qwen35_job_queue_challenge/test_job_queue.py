"""Test suite for Job Queue Challenge benchmark.

This test suite is used to evaluate LLM-generated code across 5 difficulty levels.
The tests are designed to discriminate between model capabilities.

Scoring:
- L1: 25 pts (5 tests × 5) - Basic queue operations
- L2: 25 pts (5 tests × 5) - Retry with backoff
- L3: 25 pts (5 tests × 5) - Priority scheduling
- L4: 15 pts (1 test × 15) - Concurrency bug fix
- L5: 10 pts (2 tests × 5) - Multi-file refactor

Total: 100 points
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def queue_module(tmp_path: Path) -> Any:
    """Load the generated job queue module.

    Priority:
    1. JOB_QUEUE_FILE env var (set by benchmark runner)
    2. tmp_path/job_queue.py (for manual testing)

    Note: We use 'job_queue.py' to avoid conflict with Python's stdlib 'queue' module.
    """
    # Check environment variable first (set by benchmark runner)
    queue_file_env = os.environ.get("JOB_QUEUE_FILE")
    if queue_file_env:
        queue_file = Path(queue_file_env)
    else:
        queue_file = tmp_path / "job_queue.py"

    if not queue_file.exists():
        pytest.skip(f"job_queue.py not found at {queue_file}")

    spec = importlib.util.spec_from_file_location("job_queue_impl", queue_file)
    if spec is None or spec.loader is None:
        pytest.fail("Could not load job_queue.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules["job_queue_impl"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def job_queue(queue_module: Any) -> Any:
    """Get a fresh JobQueue instance."""
    if not hasattr(queue_module, "JobQueue"):
        pytest.fail("JobQueue class not found in queue.py")
    return queue_module.JobQueue()


# ---------------------------------------------------------------------------
# L1: Basic Queue Operations (25 points)
# ---------------------------------------------------------------------------


class TestL1BasicQueue:
    """Level 1: Basic job queue with add/get operations."""

    def test_add_job_returns_job_id(self, job_queue: Any) -> None:
        """add_job should return a job_id."""
        job_id = job_queue.add_job(lambda: 42)
        assert job_id is not None
        assert isinstance(job_id, (str, int))

    def test_get_result_returns_value(self, job_queue: Any) -> None:
        """get_result should return the job's return value."""
        job_id = job_queue.add_job(lambda: 42)
        # Give it time to execute
        time.sleep(0.1)
        result = job_queue.get_result(job_id)
        assert result == 42

    def test_multiple_jobs_execute(self, job_queue: Any) -> None:
        """Multiple jobs should all execute and return results."""
        results_expected = [1, 2, 3, 4, 5]
        job_ids = [job_queue.add_job(lambda x=i: x) for i in results_expected]

        time.sleep(0.5)

        results = [job_queue.get_result(jid) for jid in job_ids]
        assert results == results_expected

    def test_fifo_ordering(self, job_queue: Any) -> None:
        """Jobs should execute in FIFO order."""
        execution_order: list[int] = []

        def record(n: int) -> int:
            execution_order.append(n)
            return n

        for i in range(5):
            job_queue.add_job(record, i)

        time.sleep(0.5)

        assert execution_order == [0, 1, 2, 3, 4]

    def test_get_result_nonexistent_job(self, job_queue: Any) -> None:
        """get_result for nonexistent job should raise or return None."""
        result = job_queue.get_result("nonexistent-job-id-12345")
        # Accept either None or KeyError
        assert result is None or isinstance(result, Exception)


# ---------------------------------------------------------------------------
# L2: Retry with Exponential Backoff (25 points)
# ---------------------------------------------------------------------------


class TestL2RetryLogic:
    """Level 2: Jobs that fail should retry with exponential backoff."""

    def test_retry_on_exception(self, job_queue: Any) -> None:
        """Job that fails then succeeds should eventually return result."""
        attempts = [0]

        def flaky_job() -> str:
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("Temporary failure")
            return "success"

        job_id = job_queue.add_job(flaky_job)
        time.sleep(5)  # Allow time for retries with backoff

        result = job_queue.get_result(job_id)
        assert result == "success"
        assert attempts[0] == 3

    def test_max_retries_exhausted(self, job_queue: Any) -> None:
        """Job that always fails should be marked as failed after 3 retries."""
        attempts = [0]

        def always_fails() -> None:
            attempts[0] += 1
            raise ValueError("Permanent failure")

        job_id = job_queue.add_job(always_fails)
        time.sleep(10)  # Allow time for all retries

        # Should have attempted 4 times (1 initial + 3 retries)
        assert attempts[0] == 4

        # Result should indicate failure (None, exception, or "failed" status)
        result = job_queue.get_result(job_id)
        assert result is None or isinstance(result, Exception) or result == "failed"

    def test_exponential_backoff_timing(self, job_queue: Any) -> None:
        """Backoff should be approximately 1s, 2s, 4s."""
        timestamps: list[float] = []

        def record_time() -> None:
            timestamps.append(time.time())
            if len(timestamps) < 4:
                raise ValueError("Retry me")

        job_id = job_queue.add_job(record_time)
        time.sleep(10)

        assert len(timestamps) >= 4

        # Check backoff intervals (with 0.5s tolerance)
        if len(timestamps) >= 2:
            interval1 = timestamps[1] - timestamps[0]
            assert 0.5 <= interval1 <= 1.5, f"First backoff should be ~1s, got {interval1}"

        if len(timestamps) >= 3:
            interval2 = timestamps[2] - timestamps[1]
            assert 1.5 <= interval2 <= 2.5, f"Second backoff should be ~2s, got {interval2}"

        if len(timestamps) >= 4:
            interval3 = timestamps[3] - timestamps[2]
            assert 3.5 <= interval3 <= 4.5, f"Third backoff should be ~4s, got {interval3}"

    def test_successful_job_no_retry(self, job_queue: Any) -> None:
        """Successful job should not trigger retries."""
        attempts = [0]

        def succeeds() -> str:
            attempts[0] += 1
            return "done"

        job_id = job_queue.add_job(succeeds)
        time.sleep(1)

        result = job_queue.get_result(job_id)
        assert result == "done"
        assert attempts[0] == 1

    def test_partial_failure_recovery(self, job_queue: Any) -> None:
        """Mix of passing and failing jobs should handle correctly."""
        results: dict[str, Any] = {}

        def job_a() -> str:
            return "a_ok"

        def job_b() -> str:
            raise ValueError("b_fails")

        def job_c() -> str:
            return "c_ok"

        id_a = job_queue.add_job(job_a)
        id_b = job_queue.add_job(job_b)
        id_c = job_queue.add_job(job_c)

        time.sleep(10)

        results["a"] = job_queue.get_result(id_a)
        results["b"] = job_queue.get_result(id_b)
        results["c"] = job_queue.get_result(id_c)

        assert results["a"] == "a_ok"
        assert results["c"] == "c_ok"
        # b should have failed
        assert results["b"] is None or isinstance(results["b"], Exception)


# ---------------------------------------------------------------------------
# L3: Priority Queue (25 points)
# ---------------------------------------------------------------------------


class TestL3PriorityQueue:
    """Level 3: Jobs with priority should execute highest priority first."""

    def test_higher_priority_first(self, job_queue: Any) -> None:
        """Priority 10 job should execute before priority 1."""
        execution_order: list[str] = []

        def record(name: str) -> str:
            execution_order.append(name)
            return name

        # Add low priority first, high priority second
        job_queue.add_job(record, "low", priority=1)
        job_queue.add_job(record, "high", priority=10)

        time.sleep(0.5)

        # High priority should execute first despite being added second
        assert execution_order[0] == "high"

    def test_same_priority_fifo(self, job_queue: Any) -> None:
        """Jobs with same priority should execute in FIFO order."""
        execution_order: list[int] = []

        def record(n: int) -> int:
            execution_order.append(n)
            return n

        for i in range(5):
            job_queue.add_job(record, i, priority=5)

        time.sleep(0.5)

        assert execution_order == [0, 1, 2, 3, 4]

    def test_mixed_priorities(self, job_queue: Any) -> None:
        """Mixed priorities should sort correctly."""
        execution_order: list[str] = []

        def record(name: str) -> str:
            execution_order.append(name)
            return name

        job_queue.add_job(record, "p3", priority=3)
        job_queue.add_job(record, "p7", priority=7)
        job_queue.add_job(record, "p1", priority=1)
        job_queue.add_job(record, "p10", priority=10)
        job_queue.add_job(record, "p5", priority=5)

        time.sleep(1)

        # Should execute in priority order: 10, 7, 5, 3, 1
        assert execution_order == ["p10", "p7", "p5", "p3", "p1"]

    def test_default_priority(self, job_queue: Any) -> None:
        """Jobs without priority should use default (middle) priority."""
        execution_order: list[str] = []

        def record(name: str) -> str:
            execution_order.append(name)
            return name

        job_queue.add_job(record, "low", priority=1)
        job_queue.add_job(record, "default")  # No priority specified
        job_queue.add_job(record, "high", priority=10)

        time.sleep(0.5)

        # Default should be between high and low (typically priority 5)
        assert execution_order[0] == "high"
        assert execution_order[-1] == "low"

    def test_priority_with_args_kwargs(self, job_queue: Any) -> None:
        """Priority should work with positional and keyword arguments."""
        results: dict[str, int] = {}

        def compute(a: int, b: int, multiplier: int = 1) -> int:
            return (a + b) * multiplier

        id1 = job_queue.add_job(compute, 1, 2, multiplier=3, priority=5)
        id2 = job_queue.add_job(compute, 10, 20, priority=10)

        time.sleep(0.5)

        results["job1"] = job_queue.get_result(id1)
        results["job2"] = job_queue.get_result(id2)

        assert results["job1"] == 9  # (1+2)*3
        assert results["job2"] == 30  # (10+20)*1


# ---------------------------------------------------------------------------
# L4: Concurrency Bug Fix (15 points)
# ---------------------------------------------------------------------------


class TestL4ConcurrencyBug:
    """Level 4: Fix a race condition in concurrent job completion."""

    def test_concurrent_completion_no_lost_results(self, job_queue: Any) -> None:
        """Concurrent job completions should not lose any results.

        This test runs many jobs in parallel and verifies all results
        are captured. A race condition in the results dict would cause
        some results to be lost.
        """
        num_jobs = 100
        job_ids: list[Any] = []

        def quick_job(n: int) -> int:
            time.sleep(0.01)  # Small delay to encourage race conditions
            return n * 2

        # Submit all jobs
        for i in range(num_jobs):
            job_ids.append(job_queue.add_job(quick_job, i))

        # Wait for completion
        time.sleep(5)

        # Collect results
        results = []
        for jid in job_ids:
            result = job_queue.get_result(jid)
            if result is not None:
                results.append(result)

        # All 100 jobs should have results
        assert len(results) == num_jobs, f"Lost {num_jobs - len(results)} results"

        # Verify correctness
        expected = {i * 2 for i in range(num_jobs)}
        actual = set(results)
        assert actual == expected


# ---------------------------------------------------------------------------
# L5: Multi-file Refactor (10 points)
# ---------------------------------------------------------------------------


class TestL5Refactor:
    """Level 5: Refactor monolithic queue.py into a package."""

    def _get_queue_dir(self, tmp_path: Path) -> Path:
        """Get the queue package directory."""
        queue_dir_env = os.environ.get("QUEUE_DIR")
        if queue_dir_env:
            return Path(queue_dir_env) / "queue"
        return tmp_path / "queue"

    def test_package_structure(self, tmp_path: Path) -> None:
        """Verify the package has correct structure."""
        queue_dir = self._get_queue_dir(tmp_path)

        # Check directory exists
        assert queue_dir.is_dir(), f"queue/ directory not found at {queue_dir}"

        # Check required files
        required_files = ["__init__.py", "core.py", "retry.py", "priority.py"]
        for fname in required_files:
            fpath = queue_dir / fname
            assert fpath.exists(), f"Missing {fname} in queue/"

    def test_imports_work(self, tmp_path: Path) -> None:
        """Verify imports from the package work correctly."""
        queue_dir = self._get_queue_dir(tmp_path)
        if not queue_dir.is_dir():
            pytest.skip("queue/ package not found")

        # Add parent directory to sys.path for imports
        parent_dir = queue_dir.parent
        sys.path.insert(0, str(parent_dir))
        try:
            # Should be able to import JobQueue from package
            from queue import JobQueue  # type: ignore

            # Create instance and verify basic operation
            q = JobQueue()
            job_id = q.add_job(lambda: 123)
            time.sleep(0.2)
            result = q.get_result(job_id)
            assert result == 123
        finally:
            sys.path.remove(str(parent_dir))


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------


def calculate_score(results: dict[str, bool]) -> dict[str, Any]:
    """Calculate score from test results.

    Args:
        results: Dict mapping test name to pass/fail boolean

    Returns:
        Score breakdown by level and total
    """
    scores = {
        "L1": 0,
        "L2": 0,
        "L3": 0,
        "L4": 0,
        "L5": 0,
        "total": 0,
        "max": 100,
        "percentage": 0.0,
    }

    for test_name, passed in results.items():
        if not passed:
            continue

        if "L1" in test_name or "BasicQueue" in test_name:
            scores["L1"] += 5
        elif "L2" in test_name or "RetryLogic" in test_name:
            scores["L2"] += 5
        elif "L3" in test_name or "PriorityQueue" in test_name:
            scores["L3"] += 5
        elif "L4" in test_name or "ConcurrencyBug" in test_name:
            scores["L4"] += 15
        elif "L5" in test_name or "Refactor" in test_name:
            scores["L5"] += 5

    # Cap each level
    scores["L1"] = min(scores["L1"], 25)
    scores["L2"] = min(scores["L2"], 25)
    scores["L3"] = min(scores["L3"], 25)
    scores["L4"] = min(scores["L4"], 15)
    scores["L5"] = min(scores["L5"], 10)

    scores["total"] = sum(scores[f"L{i}"] for i in range(1, 6))
    scores["percentage"] = (scores["total"] / scores["max"]) * 100

    return scores
