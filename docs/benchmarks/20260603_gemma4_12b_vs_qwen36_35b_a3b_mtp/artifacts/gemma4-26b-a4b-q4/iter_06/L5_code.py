from dataclasses import dataclass, field
        from typing import Any, List
        from collections import deque

        @dataclass(order=True)
        class Job:
            priority: int
            id: str = field(compare=False)
            payload: Any = field(compare=False)
            retries: int = field(default=0, compare=False)

        class JobQueue:
            def __init__(self):
                self._queue = deque()
            def enqueue(self, job: Job): self._queue.append(job)
            def dequeue(self) -> Job: return self._queue.popleft()
            def is_empty(self) -> bool: return len(self._queue) == 0