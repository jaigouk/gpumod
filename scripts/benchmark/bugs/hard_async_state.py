#!/usr/bin/env python3
"""Bug Scenario: HARD - Subtle state corruption in async task processor.

The bug is a race condition where shared mutable state is modified across
await boundaries without proper synchronization. This is subtle because:
1. It only manifests under concurrent load
2. The code "looks correct" - no obvious threading
3. asyncio's cooperative multitasking hides the danger

To test:
    python hard_async_state.py          # Shows buggy behavior (intermittent failures)
    python hard_async_state.py --fixed  # Shows correct behavior
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field

# ============================================================================
# BUGGY CODE - presented to agents
# ============================================================================

BUGGY_CODE = '''
import asyncio
from dataclasses import dataclass, field

@dataclass
class TaskContext:
    """Shared context for task processing."""
    task_id: str
    retries: int = 0
    status: str = "pending"
    result: dict = field(default_factory=dict)


class TaskProcessor:
    def __init__(self):
        self.current_context: TaskContext | None = None
        self.completed_tasks: list[TaskContext] = []

    async def process_task(self, task_id: str, payload: dict) -> dict:
        """Process a single task with retry logic."""
        self.current_context = TaskContext(task_id=task_id)

        try:
            self.current_context.status = "processing"

            # Simulate async work (e.g., API call)
            result = await self._do_work(payload)

            self.current_context.status = "completed"
            self.current_context.result = result
            self.completed_tasks.append(self.current_context)

            return result

        except Exception as e:
            self.current_context.status = "failed"
            self.current_context.retries += 1

            if self.current_context.retries < 3:
                return await self.process_task(task_id, payload)

            raise

    async def _do_work(self, payload: dict) -> dict:
        """Simulate async work."""
        await asyncio.sleep(0.01)  # Simulate I/O
        if payload.get("fail_first", False) and self.current_context.retries == 0:
            raise ValueError("Simulated failure")
        return {"processed": payload.get("data", ""), "task_id": self.current_context.task_id}

    async def process_batch(self, tasks: list[tuple[str, dict]]) -> list[dict]:
        """Process multiple tasks concurrently."""
        coros = [self.process_task(tid, payload) for tid, payload in tasks]
        return await asyncio.gather(*coros, return_exceptions=True)
'''


@dataclass
class TaskContext:
    """Shared context for task processing."""

    task_id: str
    retries: int = 0
    status: str = "pending"
    result: dict = field(default_factory=dict)


class TaskProcessorBuggy:
    def __init__(self):
        # BUG: Single shared mutable state across all concurrent tasks
        self.current_context: TaskContext | None = None
        self.completed_tasks: list[TaskContext] = []

    async def process_task(self, task_id: str, payload: dict) -> dict:
        """Process a single task with retry logic."""
        # BUG: This overwrites context that another concurrent task may be using
        self.current_context = TaskContext(task_id=task_id)

        try:
            self.current_context.status = "processing"

            # During this await, another task can modify self.current_context!
            result = await self._do_work(payload)

            # BUG: By now, self.current_context may point to a DIFFERENT task
            self.current_context.status = "completed"
            self.current_context.result = result
            self.completed_tasks.append(self.current_context)

            return result

        except Exception:
            self.current_context.status = "failed"
            self.current_context.retries += 1

            if self.current_context.retries < 3:
                return await self.process_task(task_id, payload)

            raise

    async def _do_work(self, payload: dict) -> dict:
        """Simulate async work."""
        await asyncio.sleep(0.01)  # Simulate I/O - this is where context can be stolen
        if payload.get("fail_first", False) and self.current_context.retries == 0:
            raise ValueError("Simulated failure")
        # BUG: self.current_context.task_id may now be wrong!
        return {"processed": payload.get("data", ""), "task_id": self.current_context.task_id}

    async def process_batch(self, tasks: list[tuple[str, dict]]) -> list[dict]:
        """Process multiple tasks concurrently."""
        coros = [self.process_task(tid, payload) for tid, payload in tasks]
        return await asyncio.gather(*coros, return_exceptions=True)


# ============================================================================
# FIXED CODE - expected solution
# ============================================================================

FIXED_CODE = '''
import asyncio
from dataclasses import dataclass, field

@dataclass
class TaskContext:
    """Context for a single task."""
    task_id: str
    retries: int = 0
    status: str = "pending"
    result: dict = field(default_factory=dict)


class TaskProcessor:
    def __init__(self):
        self.completed_tasks: list[TaskContext] = []
        self._lock = asyncio.Lock()  # Protect completed_tasks list

    async def process_task(self, task_id: str, payload: dict) -> dict:
        """Process a single task with retry logic."""
        # FIX: Local context per task, not shared instance variable
        context = TaskContext(task_id=task_id)

        try:
            context.status = "processing"
            result = await self._do_work(context, payload)

            context.status = "completed"
            context.result = result

            async with self._lock:  # FIX: Protect shared list
                self.completed_tasks.append(context)

            return result

        except Exception as e:
            context.status = "failed"
            context.retries += 1

            if context.retries < 3:
                # FIX: Pass context through instead of using self.current_context
                return await self._retry_task(context, payload)

            raise

    async def _retry_task(self, context: TaskContext, payload: dict) -> dict:
        """Retry with existing context."""
        context.status = "processing"
        result = await self._do_work(context, payload)
        context.status = "completed"
        context.result = result

        async with self._lock:
            self.completed_tasks.append(context)

        return result

    async def _do_work(self, context: TaskContext, payload: dict) -> dict:
        """Simulate async work."""
        await asyncio.sleep(0.01)
        if payload.get("fail_first", False) and context.retries == 0:
            raise ValueError("Simulated failure")
        return {"processed": payload.get("data", ""), "task_id": context.task_id}

    async def process_batch(self, tasks: list[tuple[str, dict]]) -> list[dict]:
        """Process multiple tasks concurrently."""
        coros = [self.process_task(tid, payload) for tid, payload in tasks]
        return await asyncio.gather(*coros, return_exceptions=True)
'''


class TaskProcessorFixed:
    def __init__(self):
        self.completed_tasks: list[TaskContext] = []
        self._lock = asyncio.Lock()

    async def process_task(self, task_id: str, payload: dict) -> dict:
        """Process a single task with retry logic."""
        # FIX: Local context per task
        context = TaskContext(task_id=task_id)

        try:
            context.status = "processing"
            result = await self._do_work(context, payload)

            context.status = "completed"
            context.result = result

            async with self._lock:
                self.completed_tasks.append(context)

            return result

        except Exception:
            context.status = "failed"
            context.retries += 1

            if context.retries < 3:
                return await self._retry_task(context, payload)

            raise

    async def _retry_task(self, context: TaskContext, payload: dict) -> dict:
        """Retry with existing context."""
        context.status = "processing"
        result = await self._do_work(context, payload)
        context.status = "completed"
        context.result = result

        async with self._lock:
            self.completed_tasks.append(context)

        return result

    async def _do_work(self, context: TaskContext, payload: dict) -> dict:
        """Simulate async work."""
        await asyncio.sleep(0.01)
        if payload.get("fail_first", False) and context.retries == 0:
            raise ValueError("Simulated failure")
        return {"processed": payload.get("data", ""), "task_id": context.task_id}

    async def process_batch(self, tasks: list[tuple[str, dict]]) -> list[dict]:
        """Process multiple tasks concurrently."""
        coros = [self.process_task(tid, payload) for tid, payload in tasks]
        return await asyncio.gather(*coros, return_exceptions=True)


# ============================================================================
# TEST HARNESS
# ============================================================================


async def test_async_state(use_fixed: bool = False) -> tuple[bool, str]:
    """Test async processor for state corruption.

    Returns (passed, message) tuple.
    """
    errors = []

    processor = TaskProcessorFixed() if use_fixed else TaskProcessorBuggy()

    # Test 1: Sequential processing should always work
    result = await processor.process_task("task-1", {"data": "hello"})
    if result.get("task_id") != "task-1":
        errors.append(
            f"Test 1 FAIL: Sequential task returned wrong ID: "
            f"expected 'task-1', got '{result.get('task_id')}'"
        )

    # Test 2: Concurrent processing - this is where the bug manifests
    tasks = [(f"task-{i}", {"data": f"data-{i}"}) for i in range(10)]
    results = await processor.process_batch(tasks)

    # Check that each result contains the correct task_id
    task_id_mismatches = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            continue
        expected_id = f"task-{i}"
        actual_id = result.get("task_id")
        if actual_id != expected_id:
            task_id_mismatches.append(f"task-{i} returned as '{actual_id}'")

    if task_id_mismatches:
        errors.append(
            f"Test 2 FAIL: Task ID corruption detected!\n"
            f"  Mismatches: {task_id_mismatches[:5]}{'...' if len(task_id_mismatches) > 5 else ''}"
        )

    # Test 3: Verify completed_tasks list integrity
    # Reset processor for clean test
    processor = TaskProcessorFixed() if use_fixed else TaskProcessorBuggy()
    processor.completed_tasks.clear()

    tasks = [(f"batch-{i}", {"data": f"value-{i}"}) for i in range(5)]
    await processor.process_batch(tasks)

    # Check completed_tasks has unique task_ids
    completed_ids = [ctx.task_id for ctx in processor.completed_tasks]
    unique_ids = set(completed_ids)

    if len(completed_ids) != len(unique_ids):
        errors.append(
            f"Test 3 FAIL: Duplicate task IDs in completed_tasks!\n"
            f"  Total: {len(completed_ids)}, Unique: {len(unique_ids)}\n"
            f"  IDs: {completed_ids}"
        )

    # Test 4: Retry logic with concurrent tasks
    processor = TaskProcessorFixed() if use_fixed else TaskProcessorBuggy()

    # Some tasks fail on first try
    tasks = [
        ("retry-0", {"data": "a", "fail_first": True}),
        ("retry-1", {"data": "b", "fail_first": False}),
        ("retry-2", {"data": "c", "fail_first": True}),
        ("retry-3", {"data": "d", "fail_first": False}),
    ]

    results = await processor.process_batch(tasks)

    # Check results match expected task IDs
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            continue
        expected_id = f"retry-{i}"
        actual_id = result.get("task_id")
        if actual_id != expected_id:
            errors.append(
                f"Test 4 FAIL: Retry corrupted task ID! "
                f"Expected '{expected_id}', got '{actual_id}'"
            )

    if errors:
        return False, "\n".join(errors)
    return True, "All tests passed! No async state corruption detected."


def test_sync_wrapper(use_fixed: bool = False) -> tuple[bool, str]:
    """Synchronous wrapper for async test."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already in an event loop - create a new one in a thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, test_async_state(use_fixed))
            return future.result()
    else:
        return asyncio.run(test_async_state(use_fixed))


# ============================================================================
# SCORING HELPERS
# ============================================================================

BUG_KEYWORDS = [
    "race condition",
    "shared state",
    "shared mutable",
    "concurrent access",
    "current_context",
    "instance variable",
    "self.current",
    "await boundary",
    "context switching",
    "interleaving",
    "overwritten",
    "clobbered",
    "task corruption",
    "asyncio",
    "cooperative multitasking",
]

FIX_KEYWORDS = [
    "local variable",
    "local context",
    "pass context",
    "asyncio.lock",
    "async with",
    "remove self.current",
    "thread-local",
    "contextvars",
    "per-task",
    "task-local",
    "lock",
]


def score_response(response: str) -> dict:
    """Score an agent's response."""
    response_lower = response.lower()

    bug_found = any(kw in response_lower for kw in BUG_KEYWORDS)
    fix_proposed = any(kw in response_lower for kw in FIX_KEYWORDS)

    return {
        "bug_identified": bug_found,
        "fix_proposed": fix_proposed,
        "score": (2 if bug_found else 0) + (1 if fix_proposed else 0),
        "max_score": 3,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    use_fixed = "--fixed" in sys.argv

    mode = "FIXED" if use_fixed else "BUGGY"
    print(f"Running {mode} version...")
    print("-" * 40)

    passed, msg = test_sync_wrapper(use_fixed)
    print(msg)
    print("-" * 40)

    if passed:
        print("RESULT: PASS")
        sys.exit(0)
    else:
        print("RESULT: FAIL")
        sys.exit(1)
