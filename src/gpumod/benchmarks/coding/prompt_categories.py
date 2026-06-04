"""Prompt categories for benchmark input sizes."""

from __future__ import annotations

from enum import Enum


class PromptCategory(Enum):
    """Prompt size categories for benchmarking."""

    SHORT = ("short", 100)
    MEDIUM = ("medium", 500)
    LONG = ("long", 2000)
    MULTI_TURN = ("multi_turn", 1000)

    def __init__(self, name: str, target_tokens: int) -> None:
        self._name = name
        self._target_tokens = target_tokens

    @property
    def target_tokens(self) -> int:
        """Target token count for this category."""
        return self._target_tokens


# Prompt templates for coding benchmarks
_SHORT_PROMPT = """Write a Python function that checks if a number is prime.

Requirements:
- Function name: `is_prime`
- Input: An integer `n`
- Output: `True` if prime, `False` otherwise
- Handle edge cases: negative numbers and 0/1 should return False
- Include a docstring explaining the algorithm
- Add type hints for the function signature"""

_MEDIUM_PROMPT = """Implement a job queue system in Python with the following requirements:

1. A `JobQueue` class with these methods:
   - `add_job(job_fn, *args, **kwargs)` - Add a job to the queue, return job_id
   - `get_result(job_id)` - Get the result of a completed job
   - `process_next()` - Process the next job in FIFO order

2. Jobs should be processed in the order they were added (FIFO)
3. Results should be stored and retrievable by job_id
4. Handle exceptions gracefully - store the exception as the result

Example usage:
```python
queue = JobQueue()
job_id = queue.add_job(lambda x: x * 2, 5)
queue.process_next()
result = queue.get_result(job_id)  # Should return 10
```

Implement this with proper type hints and docstrings."""

_LONG_PROMPT = """Design and implement a complete task scheduler system in Python.

## Core Requirements

### 1. Task Management
- `Task` dataclass with: id, name, priority (0-10), dependencies, status,
  created_at, started_at, completed_at
- Tasks can depend on other tasks (DAG structure)
- Tasks cannot start until all dependencies are completed

### 2. Scheduler Class
```python
class TaskScheduler:
    def add_task(self, name: str, priority: int = 5, dependencies: list[str] = None) -> str
    def get_task(self, task_id: str) -> Task
    def start_task(self, task_id: str) -> bool
    def complete_task(self, task_id: str, result: Any = None) -> bool
    def fail_task(self, task_id: str, error: str) -> bool
    def get_ready_tasks(self) -> list[Task]  # Tasks with all deps completed
    def get_next_task(self) -> Task | None  # Highest priority ready task
```

### 3. Priority Queue
- Higher priority tasks (lower number) should be scheduled first
- Within same priority, use FIFO ordering
- `get_next_task()` returns the highest priority task that is ready

### 4. Dependency Resolution
- Detect circular dependencies when adding tasks
- Raise `CircularDependencyError` if detected
- `get_ready_tasks()` only returns tasks whose dependencies are all completed

### 5. Persistence (Optional)
- `save(path: str)` - Save scheduler state to JSON
- `load(path: str)` - Load scheduler state from JSON

## Example Usage

```python
scheduler = TaskScheduler()

# Add tasks with dependencies
task_a = scheduler.add_task("Parse input", priority=1)
task_b = scheduler.add_task("Validate data", priority=2, dependencies=[task_a])
task_c = scheduler.add_task("Process data", priority=2, dependencies=[task_b])
task_d = scheduler.add_task("Generate report", priority=3, dependencies=[task_c])

# Get ready tasks (only task_a initially)
ready = scheduler.get_ready_tasks()  # [task_a]

# Complete task_a
scheduler.start_task(task_a)
scheduler.complete_task(task_a)

# Now task_b is ready
ready = scheduler.get_ready_tasks()  # [task_b]
```

## Implementation Notes

1. Use `dataclasses` for Task
2. Use `heapq` for priority queue (or implement your own)
3. Use `graphlib.TopologicalSorter` for dependency resolution (Python 3.9+)
4. Include comprehensive type hints
5. Add docstrings for all public methods
6. Handle edge cases: empty scheduler, invalid task_id, already completed tasks

Implement the complete system with all features above."""

_MULTI_TURN_PROMPTS = [
    "What is the best way to implement a singleton pattern in Python?",
    "Can you show me an example with thread safety?",
    "How would I add lazy initialization to this?",
    "Now make it work with type hints and generics.",
]


def generate_prompt(category: PromptCategory) -> str | list[str]:
    """Generate a prompt for the given category.

    Args:
        category: The prompt category

    Returns:
        A prompt string, or list of strings for MULTI_TURN
    """
    if category == PromptCategory.SHORT:
        return _SHORT_PROMPT
    if category == PromptCategory.MEDIUM:
        return _MEDIUM_PROMPT
    if category == PromptCategory.LONG:
        return _LONG_PROMPT
    if category == PromptCategory.MULTI_TURN:
        return _MULTI_TURN_PROMPTS.copy()
    msg = f"Unknown category: {category}"
    raise ValueError(msg)
