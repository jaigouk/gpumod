from dataclasses import dataclass
        from typing import Any, Callable
        from collections import deque

        @dataclass
        class Job:
            id: str
            payload: Any
            priority: int = 0

        class JobQueue:
            def __init__(self):
                self.items = deque()
            def push(self, job: Job):
                self.items.append(job)
            def pop(self) -> Job:
                return self.items.popleft()