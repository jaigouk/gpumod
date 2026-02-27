#!/usr/bin/env python3
"""Qwen3.5 Job Queue Challenge Benchmark Runner.

Runs the graduated difficulty benchmark against an LLM endpoint and scores results.

Usage:
    uv run python docs/benchmarks/20260226_qwen35_job_queue_challenge/benchmark_runner.py \
        --model qwen35-27b-q3 --port 7093 \
        --output docs/benchmarks/20260226_qwen35_job_queue_challenge/
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Prompts for each level
# ---------------------------------------------------------------------------

PROMPTS = {
    "L1": """Implement a JobQueue class in Python with the following requirements:

1. `add_job(fn, *args, **kwargs) -> job_id`: Add a job to the queue. Returns a unique job_id.
2. `get_result(job_id) -> result`: Get the result of a completed job. Return None if not found.
3. Jobs should execute in FIFO order.
4. Jobs should execute asynchronously (use threading or asyncio).

Provide a complete, working implementation in a single Python file.
Only output the Python code, no explanations.""",
    "L2": """Extend the JobQueue to add retry logic with exponential backoff:

1. If a job raises an exception, retry it up to 3 times.
2. Use exponential backoff: wait 1s before first retry, 2s before second, 4s before third.
3. After 3 failed retries (4 total attempts), mark the job as failed.
4. `get_result()` should return None or raise for failed jobs.

Keep all L1 functionality. Provide complete code.
Only output the Python code, no explanations.""",
    "L3": """Extend the JobQueue to add priority scheduling:

1. `add_job(fn, *args, priority=5, **kwargs)`: Accept optional priority 1-10 (10 = highest).
2. Higher priority jobs should execute before lower priority jobs.
3. Jobs with the same priority should execute in FIFO order.
4. Default priority is 5.

Keep all L1 and L2 functionality. Provide complete code.
Only output the Python code, no explanations.""",
    "L4": """The following JobQueue implementation has a concurrency bug that causes lost results
when many jobs complete simultaneously. Find and fix the bug.

```python
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable

class JobQueue:
    def __init__(self):
        self.jobs = deque()
        self.results = {}
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def add_job(self, fn: Callable, *args, priority: int = 5, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        with self.lock:
            self.jobs.append((priority, time.time(), job_id, fn, args, kwargs))
            self.jobs = deque(sorted(self.jobs, key=lambda x: (-x[0], x[1])))
        return job_id

    def get_result(self, job_id: str) -> Any:
        return self.results.get(job_id)

    def _worker(self):
        while True:
            job = None
            with self.lock:
                if self.jobs:
                    job = self.jobs.popleft()

            if job:
                priority, ts, job_id, fn, args, kwargs = job
                try:
                    result = self._execute_with_retry(fn, args, kwargs)
                    # BUG: This write is not protected!
                    self.results[job_id] = result
                except Exception as e:
                    self.results[job_id] = None
            else:
                time.sleep(0.01)

    def _execute_with_retry(self, fn, args, kwargs, max_retries=3):
        backoff = 1
        for attempt in range(max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(backoff)
                backoff *= 2
```

Identify the bug and provide a fixed implementation.
Only output the fixed Python code, no explanations.""",
    "L5": """Refactor the monolithic queue.py into a Python package with this structure:

```
queue/
  __init__.py      # Exports JobQueue
  core.py          # Base JobQueue class
  retry.py         # RetryPolicy mixin/decorator
  priority.py      # PriorityMixin
```

Requirements:
1. `from queue import JobQueue` should work
2. All existing functionality must be preserved
3. Each module should have a single responsibility
4. Use mixins or composition for retry and priority features

Provide all four files with their complete contents.
Format as:
```python
# queue/__init__.py
<code>

# queue/core.py
<code>

# queue/retry.py
<code>

# queue/priority.py
<code>
```""",
}


@dataclass
class LevelResult:
    """Result from running one level."""

    level: str
    prompt: str
    response: str
    code: str
    tests_passed: int
    tests_total: int
    test_output: str
    duration_ms: float
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""

    model_id: str
    model_name: str
    port: int
    timestamp: str
    total_duration_ms: float
    levels: list[LevelResult] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    # Try to find code blocks
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if code_blocks:
        return "\n\n".join(code_blocks)

    # If no code blocks, assume entire response is code
    # Remove common non-code patterns
    lines = response.strip().split("\n")
    code_lines = []
    for line in lines:
        # Skip explanation lines
        if line.startswith(("#", "Here", "This", "The", "I ", "Note")):
            if not line.startswith("# "):  # Keep comments
                continue
        code_lines.append(line)

    return "\n".join(code_lines)


def extract_multifile_code(response: str) -> dict[str, str]:
    """Extract multiple files from L5 response."""
    files: dict[str, str] = {}

    # Pattern to match file headers
    patterns = [
        r"#\s*(queue/\w+\.py)\n(.*?)(?=#\s*queue/|\Z)",
        r"```python\n#\s*(queue/\w+\.py)\n(.*?)```",
        r"(queue/\w+\.py)[:\n]+(.*?)(?=queue/\w+\.py|\Z)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            for filename, code in matches:
                filename = filename.strip()
                code = code.strip()
                if code:
                    files[filename] = code
            break

    return files


def call_llm(
    prompt: str,
    port: int,
    model: str = "default",
    temperature: float = 0.1,
    max_tokens: int = 8192,
    timeout: float = 600.0,
) -> tuple[str, float]:
    """Call the LLM endpoint and return response + duration.

    For thinking models (Qwen3.5), we prepend /no_think to skip
    the reasoning phase and get direct output.
    """
    start = time.perf_counter()

    # Prepend /no_think for thinking models to get direct output
    full_prompt = f"/no_think\n{prompt}"

    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()

    duration_ms = (time.perf_counter() - start) * 1000
    content = data["choices"][0]["message"]["content"]

    return content, duration_ms


def run_tests(
    code: str,
    tmp_dir: Path,
    test_classes: list[str],
    is_multifile: bool = False,
) -> tuple[int, int, str]:
    """Run pytest on generated code and return pass count, total, output."""
    # Use job_queue.py to avoid conflict with stdlib 'queue' module
    queue_file = tmp_dir / "job_queue.py"

    # Write the code for single-file tests
    if not is_multifile and code:
        queue_file.write_text(code)
        print(f"  [DEBUG] Wrote {len(code)} bytes to {queue_file}")

    # Build pytest command for specific test classes
    # Use absolute path since we're running from tmp_dir
    test_file = Path(__file__).parent.resolve() / "test_job_queue.py"
    test_args = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),
        "-v",
        "--tb=short",
    ]

    # Use OR expression for test class filter
    if test_classes:
        filter_expr = " or ".join(test_classes)
        test_args.extend(["-k", filter_expr])

    # Set environment variables for test fixtures
    env = {
        **subprocess.os.environ,
        "PYTHONPATH": str(tmp_dir),
        "JOB_QUEUE_FILE": str(queue_file),  # Use job_queue to avoid stdlib conflict
        "QUEUE_DIR": str(tmp_dir),
    }

    # Don't change cwd - it confuses pytest's rootdir detection
    result = subprocess.run(
        test_args,
        capture_output=True,
        text=True,
        env=env,
    )

    output = result.stdout + result.stderr

    # Parse results
    passed = len(re.findall(r"PASSED", output))
    failed = len(re.findall(r"FAILED", output))
    errors = len(re.findall(r"ERROR", output))
    total = passed + failed

    # Debug output for failures
    if errors > 0 or (passed == 0 and failed == 0):
        print(f"  [DEBUG] Test run had {errors} errors, {passed} passed, {failed} failed")
        # Show first error if any
        error_match = re.search(r"E\s+(\w+Error:.*?)(?:\n(?!E\s)|$)", output, re.DOTALL)
        if error_match:
            print(f"  [DEBUG] First error: {error_match.group(1)[:200]}")

    return passed, total, output


def run_level(
    level: str,
    prompt: str,
    port: int,
    model_name: str,
    accumulated_code: str = "",
) -> LevelResult:
    """Run one level of the benchmark."""
    # For L2+ levels, include previous code in context
    full_prompt = prompt
    if accumulated_code and level not in ("L1", "L4", "L5"):
        full_prompt = f"Current implementation:\n```python\n{accumulated_code}\n```\n\n{prompt}"

    try:
        response, duration_ms = call_llm(full_prompt, port, model_name)
    except Exception as e:
        print(f"  [ERROR] LLM call failed: {e}")
        return LevelResult(
            level=level,
            prompt=full_prompt,
            response="",
            code="",
            tests_passed=0,
            tests_total=0,
            test_output="",
            duration_ms=0,
            error=str(e),
        )

    # Debug: show response stats
    print(f"  [DEBUG] Response length: {len(response)} chars")
    if not response.strip():
        print("  [WARNING] Empty response from model!")

    # Extract code
    if level == "L5":
        # Multi-file extraction handled separately
        code = response  # Keep raw for now
    else:
        code = extract_code(response)
        print(f"  [DEBUG] Extracted code: {len(code)} chars")

    # Map levels to test classes
    test_class_map = {
        "L1": ["TestL1BasicQueue"],
        "L2": ["TestL1BasicQueue", "TestL2RetryLogic"],
        "L3": ["TestL1BasicQueue", "TestL2RetryLogic", "TestL3PriorityQueue"],
        "L4": ["TestL4ConcurrencyBug"],
        "L5": ["TestL5Refactor"],
    }

    # Run tests
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        if level == "L5":
            # Handle multi-file
            files = extract_multifile_code(response)
            if files:
                queue_dir = tmp_path / "queue"
                queue_dir.mkdir()
                for fname, fcode in files.items():
                    fpath = tmp_path / fname
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    fpath.write_text(fcode)
            passed, total, test_output = run_tests(
                "", tmp_path, test_class_map[level], is_multifile=True
            )
        else:
            passed, total, test_output = run_tests(
                code, tmp_path, test_class_map[level], is_multifile=False
            )

    return LevelResult(
        level=level,
        prompt=full_prompt,
        response=response,
        code=code,
        tests_passed=passed,
        tests_total=total,
        test_output=test_output,
        duration_ms=duration_ms,
    )


def run_benchmark(
    model_id: str,
    model_name: str,
    port: int,
    levels: list[str] | None = None,
) -> BenchmarkResult:
    """Run the complete benchmark."""
    if levels is None:
        levels = ["L1", "L2", "L3", "L4", "L5"]

    start = time.perf_counter()
    results = BenchmarkResult(
        model_id=model_id,
        model_name=model_name,
        port=port,
        timestamp=datetime.now(UTC).isoformat(),
        total_duration_ms=0,
    )

    accumulated_code = ""

    for level in levels:
        print(f"\n{'='*60}")
        print(f"Running {level}...")
        print("=" * 60)

        prompt = PROMPTS[level]
        level_result = run_level(level, prompt, port, model_name, accumulated_code)
        results.levels.append(level_result)

        print(f"  Duration: {level_result.duration_ms:.0f}ms")
        print(f"  Tests: {level_result.tests_passed}/{level_result.tests_total}")

        if level_result.error:
            print(f"  Error: {level_result.error}")

        # Accumulate code for next level (except L4/L5 which are standalone)
        if level in ("L1", "L2", "L3") and level_result.code:
            accumulated_code = level_result.code

    results.total_duration_ms = (time.perf_counter() - start) * 1000

    # Calculate scores
    test_results = {}
    for lr in results.levels:
        for i in range(lr.tests_passed):
            test_results[f"{lr.level}_test_{i}"] = True
        for i in range(lr.tests_total - lr.tests_passed):
            test_results[f"{lr.level}_fail_{i}"] = False

    # Import scoring function
    from test_job_queue import calculate_score

    results.scores = calculate_score(test_results)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Job Queue Challenge Benchmark")
    parser.add_argument("--model", required=True, help="Model preset ID")
    parser.add_argument("--port", type=int, default=7081, help="LLM port")
    parser.add_argument("--name", help="Model display name (default: use --model)")
    parser.add_argument("--output", type=Path, default=Path("."), help="Output directory")
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=["L1", "L2", "L3", "L4", "L5"],
        help="Run specific levels only",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run (default: 1, recommended: 5 for reliable results)",
    )

    args = parser.parse_args()

    model_name = args.name or args.model

    print(f"Job Queue Challenge Benchmark")
    print(f"Model: {model_name}")
    print(f"Port: {args.port}")
    print(f"Levels: {args.levels or 'all'}")
    print(f"Iterations: {args.iterations}")

    # Run multiple iterations
    all_results: list[BenchmarkResult] = []
    all_scores: list[int] = []
    total_start = time.perf_counter()

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'#'*60}")
        print(f"# ITERATION {iteration}/{args.iterations}")
        print("#" * 60)

        result = run_benchmark(
            model_id=args.model,
            model_name=model_name,
            port=args.port,
            levels=args.levels,
        )
        all_results.append(result)
        all_scores.append(result.scores["total"])

        print(f"\n  Iteration {iteration} score: {result.scores['total']}/100")

    total_duration_ms = (time.perf_counter() - total_start) * 1000

    # Find best result (highest score)
    best_idx = all_scores.index(max(all_scores))
    best_result = all_results[best_idx]

    # Calculate statistics
    avg_score = sum(all_scores) / len(all_scores)
    min_score = min(all_scores)
    max_score = max(all_scores)

    # Create artifacts directory for this model
    artifacts_dir = args.output / "artifacts" / args.model
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save generated code from BEST result only
    for lr in best_result.levels:
        if lr.code:
            code_file = artifacts_dir / f"{lr.level}_job_queue.py"
            code_file.write_text(lr.code)
            print(f"  Artifact saved: {code_file}")

    # Build aggregated results
    aggregated = {
        "model_id": best_result.model_id,
        "model_name": best_result.model_name,
        "port": best_result.port,
        "timestamp": datetime.now(UTC).isoformat(),
        "iterations": args.iterations,
        "total_duration_ms": total_duration_ms,
        "statistics": {
            "best_score": max_score,
            "avg_score": round(avg_score, 1),
            "min_score": min_score,
            "max_score": max_score,
            "all_scores": all_scores,
            "best_iteration": best_idx + 1,
        },
        "scores": best_result.scores,
        "levels": [
            {
                "level": lr.level,
                "tests_passed": lr.tests_passed,
                "tests_total": lr.tests_total,
                "duration_ms": lr.duration_ms,
                "error": lr.error,
            }
            for lr in best_result.levels
        ],
    }

    # Save results
    output_file = args.output / f"result_{args.model}.json"
    with open(output_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL SCORES (Best of {} iterations)".format(args.iterations))
    print("=" * 60)
    print(f"L1 (Basic Queue):     {best_result.scores['L1']}/25")
    print(f"L2 (Retry Logic):     {best_result.scores['L2']}/25")
    print(f"L3 (Priority Queue):  {best_result.scores['L3']}/25")
    print(f"L4 (Concurrency Fix): {best_result.scores['L4']}/15")
    print(f"L5 (Refactor):        {best_result.scores['L5']}/10")
    print("-" * 40)
    pct = best_result.scores['percentage']
    print(f"TOTAL:                {best_result.scores['total']}/100 ({pct:.1f}%)")
    if args.iterations > 1:
        print(f"\nStatistics ({args.iterations} iterations):")
        print(f"  Best:  {max_score}/100 (iteration {best_idx + 1})")
        print(f"  Avg:   {avg_score:.1f}/100")
        print(f"  Min:   {min_score}/100")
        print(f"  Max:   {max_score}/100")
        print(f"  All:   {all_scores}")
    print(f"\nTotal time: {total_duration_ms/1000:.1f}s")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
