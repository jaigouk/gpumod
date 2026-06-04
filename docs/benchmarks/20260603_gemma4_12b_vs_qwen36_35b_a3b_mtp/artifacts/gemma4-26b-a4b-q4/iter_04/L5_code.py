from dataclasses import dataclass, field
        from typing import Any, deque

        @dataclass(order=True) # order=True helps with PriorityQueue
        class Job:
            priority: int
            id: str = field(compare=False)
            payload: Any = field(compare=False)

        class JobQueue:
            def __init__(self):
                self._queue = deque()
            def push(self, job: Job):
                self._queue.append(job)
            def pop(self) -> Job:
                return self._queue.popleft()
            def is_empty(self) -> bool:
                return len(self._queue) == 0