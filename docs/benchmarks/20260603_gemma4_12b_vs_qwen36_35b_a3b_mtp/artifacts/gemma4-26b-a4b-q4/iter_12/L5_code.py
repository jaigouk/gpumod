from dataclasses import dataclass, field
        from typing import Any, Deque
        from collections import deque

        @dataclass(order=False) # order=False because we handle priority in priority.py
        class Job:
            name: str
            payload: Any
            priority: int = 0
            retries: int = 0

        class JobQueue:
            def __init__(self):
                self._queue: Deque[Job] = deque()
            def push(self, job: Job): self._queue.append(job)
            def pop(self) -> Job: return self._queue.popleft()
            def is_empty(self) -> bool: return len(self._queue) == 0