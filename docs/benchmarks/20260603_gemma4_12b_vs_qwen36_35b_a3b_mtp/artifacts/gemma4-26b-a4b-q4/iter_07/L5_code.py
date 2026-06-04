from dataclasses import dataclass, field
        from typing import Any, Deque
        from collections import deque

        @dataclass(order=True)
        class Job:
            priority: int = 10  # Default priority
            id: str = field(compare=False)
            payload: Any = field(compare=False)

        class JobQueue:
            def __init__(self):
                self._queue: Deque[Job] = deque()

            def push(self, job: Job):
                self._queue.append(job)

            def pop(self) -> Job:
                return self._queue.popleft()

            def __len__(self):
                return len(self._queue)