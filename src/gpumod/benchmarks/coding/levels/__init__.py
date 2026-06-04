"""Level definitions and pytest-based validators for Qwen3.5 benchmark."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LevelDefinition:
    """Definition of a benchmark level with prompt and test code."""

    level: int
    name: str
    points: int
    prompt: str
    test_code: str


@dataclass
class ValidationResult:
    """Result of validating generated code against tests."""

    passed: bool
    pass_rate: float
    tests_passed: int
    tests_total: int
    error: str | None = None


class PytestValidator:
    """Validates generated code by running pytest tests."""

    def __init__(self, timeout_seconds: int = 30) -> None:
        self.timeout_seconds = timeout_seconds

    def validate(self, code: str, test_code: str) -> ValidationResult:
        """Validate generated code against test code.

        Args:
            code: The generated Python code (solution)
            test_code: The pytest test code

        Returns:
            ValidationResult with pass/fail status and metrics
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Write solution code
            solution_file = tmp_path / "solution.py"
            try:
                solution_file.write_text(code)
            except Exception as e:
                return ValidationResult(
                    passed=False,
                    pass_rate=0.0,
                    tests_passed=0,
                    tests_total=0,
                    error=f"Failed to write solution: {e}",
                )

            # Write test code
            test_file = tmp_path / "test_solution.py"
            test_file.write_text(test_code)

            # Run pytest
            try:
                result = subprocess.run(  # noqa: S603
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        str(test_file),
                        "-v",
                        "--tb=short",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    cwd=tmpdir,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return ValidationResult(
                    passed=False,
                    pass_rate=0.0,
                    tests_passed=0,
                    tests_total=0,
                    error="Timeout: code execution exceeded time limit",
                )
            except Exception as e:
                return ValidationResult(
                    passed=False,
                    pass_rate=0.0,
                    tests_passed=0,
                    tests_total=0,
                    error=str(e),
                )

            # Parse pytest output
            return self._parse_pytest_output(result)

    def _parse_pytest_output(self, result: subprocess.CompletedProcess[str]) -> ValidationResult:
        """Parse pytest output to extract pass/fail counts."""
        output = result.stdout + result.stderr

        # Check for syntax errors
        if "SyntaxError" in output:
            return ValidationResult(
                passed=False,
                pass_rate=0.0,
                tests_passed=0,
                tests_total=0,
                error="SyntaxError in generated code",
            )

        # Parse test results from pytest output
        # Look for patterns like "1 passed" or "2 passed, 1 failed"
        tests_passed = 0
        tests_failed = 0

        import re

        passed_match = re.search(r"(\d+) passed", output)
        failed_match = re.search(r"(\d+) failed", output)

        if passed_match:
            tests_passed = int(passed_match.group(1))
        if failed_match:
            tests_failed = int(failed_match.group(1))

        tests_total = tests_passed + tests_failed

        if tests_total == 0:
            # No tests ran - likely an import error
            return ValidationResult(
                passed=False,
                pass_rate=0.0,
                tests_passed=0,
                tests_total=0,
                error="No tests executed - likely import error",
            )

        pass_rate = tests_passed / tests_total
        passed = result.returncode == 0

        return ValidationResult(
            passed=passed,
            pass_rate=pass_rate,
            tests_passed=tests_passed,
            tests_total=tests_total,
            error=None if passed else "Some tests failed",
        )


# ---------------------------------------------------------------------------
# Level Registry
# ---------------------------------------------------------------------------


LEVEL_REGISTRY: dict[int, LevelDefinition] = {}


def register_level(level: LevelDefinition) -> None:
    """Register a level definition.

    Args:
        level: The LevelDefinition to register
    """
    LEVEL_REGISTRY[level.level] = level


def get_level(level_num: int) -> LevelDefinition:
    """Get a level definition by number.

    Args:
        level_num: The level number (1-5)

    Returns:
        The LevelDefinition

    Raises:
        KeyError: If level not found
    """
    return LEVEL_REGISTRY[level_num]


# ---------------------------------------------------------------------------
# Default Level Definitions
# ---------------------------------------------------------------------------


_LEVEL_1_PROMPT = """Implement a basic job queue in Python.

Requirements:
1. Create a `JobQueue` class with the following methods:
   - `add_job(job_id: str, data: dict) -> str`: Add a job to the queue, return job_id
   - `get_result(job_id: str) -> dict | None`: Get the result of a completed job

2. Jobs should be processed in FIFO (First-In-First-Out) order
3. The queue should store jobs internally until they are processed
4. Each job has a unique job_id that is returned when added

Example usage:
```python
queue = JobQueue()
job_id = queue.add_job("job1", {"task": "process_data"})
# After processing...
result = queue.get_result("job1")
```

Write only the Python code, no explanations.
"""

_LEVEL_1_TESTS = """
def test_add_job_returns_job_id():
    from solution import JobQueue
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "test"})
    assert job_id == "job1"

def test_add_multiple_jobs():
    from solution import JobQueue
    queue = JobQueue()
    queue.add_job("job1", {"a": 1})
    queue.add_job("job2", {"b": 2})
    # Should not raise

def test_get_result_returns_none_for_unknown():
    from solution import JobQueue
    queue = JobQueue()
    result = queue.get_result("unknown")
    assert result is None
"""

_LEVEL_2_PROMPT = """Extend the JobQueue class with a retry method.

Add `process_job(job_id: str, processor: Callable) -> bool`. Behavior:

1. Call `processor(data)` where `data` is the dict passed to `add_job(job_id, data)`.
2. On any exception from the processor, retry. Make up to 4 total attempts (initial + 3 retries).
3. Track the retry count per job. Use exponential backoff (delays of 1s, 2s, 4s).
4. Do NOT actually sleep — record the delays as data.
5. Return True on the first successful attempt. Return False if all 4 attempts raise.

Use only the standard library — do not import external packages.
Write only the Python code, no explanations.
"""

_LEVEL_2_TESTS = """
def test_process_job_success():
    from solution import JobQueue
    queue = JobQueue()
    queue.add_job("job1", {"value": 42})

    def processor(data):
        return data["value"] * 2

    success = queue.process_job("job1", processor)
    assert success is True

def test_process_job_retries_on_failure():
    from solution import JobQueue
    queue = JobQueue()
    queue.add_job("job1", {})

    call_count = [0]
    def failing_processor(data):
        call_count[0] += 1
        if call_count[0] < 3:
            raise Exception("Temporary failure")
        return "success"

    success = queue.process_job("job1", failing_processor)
    assert success is True
    assert call_count[0] == 3

def test_process_job_fails_after_max_retries():
    from solution import JobQueue
    queue = JobQueue()
    queue.add_job("job1", {})

    def always_failing(data):
        raise Exception("Always fails")

    success = queue.process_job("job1", always_failing)
    assert success is False
"""

_LEVEL_3_PROMPT = """Implement priority-based job scheduling for the queue.

Requirements:
1. Modify `add_job` to accept an optional priority parameter (default=0)
2. Higher priority jobs should be processed before lower priority jobs
3. Jobs with the same priority should maintain FIFO order
4. Add `get_next_job() -> tuple[str, dict] | None` to get the highest priority job

Priority levels:
- 0: Normal (default)
- 1: High
- 2: Critical

Example:
```python
queue = JobQueue()
queue.add_job("normal", {"type": "normal"}, priority=0)
queue.add_job("critical", {"type": "critical"}, priority=2)
queue.add_job("high", {"type": "high"}, priority=1)

job = queue.get_next_job()  # Returns critical job first
```

Write only the Python code, no explanations.
"""

_LEVEL_3_TESTS = """
def test_priority_order():
    from solution import JobQueue
    queue = JobQueue()
    queue.add_job("low", {"p": 0}, priority=0)
    queue.add_job("high", {"p": 2}, priority=2)
    queue.add_job("mid", {"p": 1}, priority=1)

    job_id, _ = queue.get_next_job()
    assert job_id == "high"

def test_fifo_within_same_priority():
    from solution import JobQueue
    queue = JobQueue()
    queue.add_job("first", {}, priority=1)
    queue.add_job("second", {}, priority=1)

    job_id, _ = queue.get_next_job()
    assert job_id == "first"

def test_default_priority_is_zero():
    from solution import JobQueue
    queue = JobQueue()
    queue.add_job("default", {})
    queue.add_job("explicit_zero", {}, priority=0)
    # Both should have same priority
"""

_LEVEL_4_PROMPT = """Fix the concurrency bug in this job queue implementation.

The following code has a race condition. Find and fix it:

```python
import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}

    def add_job(self, job_id, data):
        self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        result = processor(data)

        # BUG: Race condition here - multiple threads can write simultaneously
        self.results[job_id] = result

        del self.jobs[job_id]
        return True

    def get_result(self, job_id):
        return self.results.get(job_id)
```

Fix the race condition using proper locking mechanisms.
Write the complete fixed implementation.
"""

_LEVEL_4_TESTS = """
import threading
import time

def test_thread_safe_process():
    from solution import JobQueue
    queue = JobQueue()

    for i in range(10):
        queue.add_job(f"job{i}", {"value": i})

    results = []
    def process_jobs():
        for i in range(10):
            def processor(data):
                return data["value"] * 2
            queue.process_job(f"job{i}", processor)
            result = queue.get_result(f"job{i}")
            if result is not None:
                results.append(result)

    threads = [threading.Thread(target=process_jobs) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have processed all jobs without errors
    assert len(results) > 0

def test_uses_lock():
    from solution import JobQueue
    import inspect
    source = inspect.getsource(JobQueue)
    assert "Lock" in source or "lock" in source
"""

_LEVEL_5_PROMPT = """Refactor the job queue into three composable classes in solution.py.

Define these three classes:

1. `Job` — dataclass with fields `id: str`, `data: dict`, `priority: int = 0`,
   `retries: int = 0`. Use `@dataclass` from the standard library.
2. `RetryPolicy` — encapsulates retry-with-backoff:
   - `__init__(self, max_attempts: int = 4)`
   - `run(self, fn: Callable, data) -> tuple[bool, int]` — calls `fn(data)`, retries
     on any exception until success or `max_attempts` reached, returns
     `(success, attempts_made)`. Do NOT actually sleep.
3. `JobQueue` — orchestrates Jobs using RetryPolicy:
   - `add_job(self, job_id: str, data: dict, priority: int = 0) -> None`
   - `process_job(self, job_id: str, processor: Callable) -> bool` — must USE
     RetryPolicy (compose, don't reimplement retry logic).
   - `get_next_job(self) -> tuple[str, dict] | None` — return the highest-priority
     job's (id, data); FIFO order within the same priority.

Use only the standard library — do not import external packages.
Write only the Python code, no explanations.
"""

_LEVEL_5_TESTS = """
import inspect

def test_job_dataclass_has_required_fields():
    from solution import Job
    j = Job(id="t1", data={"x": 1})
    assert j.id == "t1"
    assert j.data == {"x": 1}
    assert j.priority == 0
    assert j.retries == 0

def test_retry_policy_succeeds_within_max_attempts():
    from solution import RetryPolicy
    policy = RetryPolicy()
    call_count = [0]
    def fail_first_two(data):
        call_count[0] += 1
        if call_count[0] < 3:
            raise Exception("fail")
        return "ok"
    success, attempts = policy.run(fail_first_two, {})
    assert success is True
    assert attempts == 3

def test_retry_policy_returns_false_after_max_attempts():
    from solution import RetryPolicy
    policy = RetryPolicy()
    def always_fail(data):
        raise Exception("always")
    success, attempts = policy.run(always_fail, {})
    assert success is False
    assert attempts == 4

def test_job_queue_process_job_composes_retry_policy():
    from solution import JobQueue, RetryPolicy
    queue = JobQueue()
    queue.add_job("j1", {"val": 1})
    call_count = [0]
    def fail_once(data):
        call_count[0] += 1
        if call_count[0] < 2:
            raise Exception("once")
        return "ok"
    assert queue.process_job("j1", fail_once) is True
    assert call_count[0] == 2
    # Compose, don't reimplement: JobQueue source must reference RetryPolicy
    src = inspect.getsource(JobQueue)
    assert "RetryPolicy" in src

def test_job_queue_priority_ordering():
    from solution import JobQueue
    queue = JobQueue()
    queue.add_job("normal", {"k": 1}, priority=0)
    queue.add_job("critical", {"k": 2}, priority=2)
    queue.add_job("high", {"k": 3}, priority=1)
    nxt = queue.get_next_job()
    assert nxt is not None and nxt[0] == "critical"
"""


# Register default levels
register_level(
    LevelDefinition(
        level=1,
        name="Basic Queue",
        points=25,
        prompt=_LEVEL_1_PROMPT,
        test_code=_LEVEL_1_TESTS,
    )
)

register_level(
    LevelDefinition(
        level=2,
        name="Retry with Backoff",
        points=25,
        prompt=_LEVEL_2_PROMPT,
        test_code=_LEVEL_2_TESTS,
    )
)

register_level(
    LevelDefinition(
        level=3,
        name="Priority Queue",
        points=25,
        prompt=_LEVEL_3_PROMPT,
        test_code=_LEVEL_3_TESTS,
    )
)

register_level(
    LevelDefinition(
        level=4,
        name="Concurrency Bug Fix",
        points=15,
        prompt=_LEVEL_4_PROMPT,
        test_code=_LEVEL_4_TESTS,
    )
)

register_level(
    LevelDefinition(
        level=5,
        name="Compose Job + RetryPolicy + JobQueue",
        points=10,
        prompt=_LEVEL_5_PROMPT,
        test_code=_LEVEL_5_TESTS,
    )
)
