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

_LEVEL_2_PROMPT = """Extend the job queue with retry logic and exponential backoff.

Requirements:
1. Add a `process_job(job_id: str, processor: Callable) -> bool` method
2. If the processor raises an exception, retry up to 3 times
3. Use exponential backoff between retries: 1s, 2s, 4s (can be simulated)
4. Track retry count for each job
5. Return True if job succeeded, False if all retries exhausted

Example:
```python
queue = JobQueue()
queue.add_job("job1", {"url": "https://example.com"})

def fetch_url(data):
    # May raise on network error
    return requests.get(data["url"])

success = queue.process_job("job1", fetch_url)
```

The backoff delays can be stored/tracked rather than actually sleeping.
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

_LEVEL_5_PROMPT = """Refactor the monolithic job queue into multiple files.

Take this single-file implementation and split it into a proper package structure:

Current structure (single file):
```python
# queue.py - everything in one file
class Job: ...
class JobQueue: ...
class PriorityQueue: ...
def process_with_retry(): ...
```

Target structure:
```
queue/
├── __init__.py      # Exports: JobQueue, Job
├── core.py          # Job dataclass, basic queue operations
├── retry.py         # Retry logic with exponential backoff
└── priority.py      # Priority queue implementation
```

Requirements:
1. Maintain all existing functionality
2. Proper imports between modules
3. Clean public API in __init__.py
4. Each module has a single responsibility

Provide the content of each file.
"""

_LEVEL_5_TESTS = """
def test_can_import_from_package():
    from solution import JobQueue
    queue = JobQueue()
    assert queue is not None

def test_job_class_exported():
    from solution import Job
    job = Job("test", {})
    assert job is not None
"""


# Register default levels
register_level(LevelDefinition(
    level=1,
    name="Basic Queue",
    points=25,
    prompt=_LEVEL_1_PROMPT,
    test_code=_LEVEL_1_TESTS,
))

register_level(LevelDefinition(
    level=2,
    name="Retry with Backoff",
    points=25,
    prompt=_LEVEL_2_PROMPT,
    test_code=_LEVEL_2_TESTS,
))

register_level(LevelDefinition(
    level=3,
    name="Priority Queue",
    points=25,
    prompt=_LEVEL_3_PROMPT,
    test_code=_LEVEL_3_TESTS,
))

register_level(LevelDefinition(
    level=4,
    name="Concurrency Bug Fix",
    points=15,
    prompt=_LEVEL_4_PROMPT,
    test_code=_LEVEL_4_TESTS,
))

register_level(LevelDefinition(
    level=5,
    name="Multi-file Refactor",
    points=10,
    prompt=_LEVEL_5_PROMPT,
    test_code=_LEVEL_5_TESTS,
))
