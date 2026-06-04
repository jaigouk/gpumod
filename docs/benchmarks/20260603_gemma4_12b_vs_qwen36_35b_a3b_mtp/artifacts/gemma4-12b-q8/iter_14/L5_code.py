from dataclasses import dataclass, field
        from typing import Any
        from collections import deque

        @dataclass(order=False)
        class Job:
            id: str
            task: callable
            priority: int = 0
            retries: int = 0
            max_retries: int = 3

        class JobQueue:
            def __init__(self):
                self._queue = deque()
            def enqueue(self, job: Job):
                self._queue.append(job)
            def dequeue(self) -> Job:
                return self._queue.popleft()
            def __len__(self):
                return len(self._queue)