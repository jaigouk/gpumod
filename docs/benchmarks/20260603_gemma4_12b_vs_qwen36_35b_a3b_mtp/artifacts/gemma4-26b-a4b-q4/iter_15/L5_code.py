from dataclasses import dataclass, field
        from typing import Any
        from collections import deque

        @dataclass(order=False) # Handled manually or via priority
        class Job:
            id: str
            payload: Any
            priority: int = 0

            def __lt__(self, other):
                # For PriorityQueue: lower number = higher priority
                return self.priority < other.priority

        class JobQueue:
            def __init__(self):
                self._queue = deque()
            def push(self, job: Job): self._queue.append(job)
            def pop(self) -> Job: return self._queue.popleft()
            def is_empty(self) -> bool: return len(self._queue) == 0