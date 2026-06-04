from dataclasses import dataclass, field
        from typing import Callable, Any, List
        from collections import deque

        @dataclass
        class Job:
            task: Callable
            args: tuple = ()
            kwargs: dict = field(default_factory=dict)
            priority: int = 10
            retries: int = 0
            max_retries: int = 3
            id: str = ""

        class JobQueue:
            def __init__(self):
                self._queue = deque()
            def push(self, job: Job):
                self._queue.append(job)
            def pop(self) -> Job:
                return self._queue.popleft()
            def __len__(self):
                return len(self._queue)