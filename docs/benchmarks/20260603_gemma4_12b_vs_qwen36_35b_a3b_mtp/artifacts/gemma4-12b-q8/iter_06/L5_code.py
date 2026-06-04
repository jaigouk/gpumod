from dataclasses import dataclass, field
        from typing import Any
        from collections import deque

        @dataclass(order=False) # Order handled by priority.py
        class Job:
            id: str
            payload: Any
            priority: int = 0
            retries_left: int = 3

        class JobQueue:
            def __init__(self):
                self.queue = deque()
            def push(self, job: Job):
                self.queue.append(job)
            def pop(self) -> Job:
                return self.queue.popleft() if self.queue else None