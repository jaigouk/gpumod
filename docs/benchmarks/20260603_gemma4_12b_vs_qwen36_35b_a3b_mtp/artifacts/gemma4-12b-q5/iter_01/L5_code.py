from dataclasses import dataclass, field
        from typing import Any
        from collections import deque

        @dataclass(order=False)
        class Job:
            id: str
            payload: Any
            priority: int = 0

        class JobQueue:
            def __init__(self):
                self._queue = deque()
            def enqueue(self, job: Job): ...
            def dequeue(self) -> Job: ...