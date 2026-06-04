from dataclasses import dataclass, field
        from typing import Any, Deque
        from collections import deque

        @dataclass(order=False) # order=False because we handle priority separately
        class Job:
            id: str
            payload: Any

        class JobQueue:
            def __init__(self):
                self._queue: Deque[Job] = deque()

            def enqueue(self, job: Job):
                self._queue.append(job)

            def dequeue(self) -> Job:
                return self._queue.popleft()

            def is_empty(self) -> bool:
                return len(self._queue) == 0