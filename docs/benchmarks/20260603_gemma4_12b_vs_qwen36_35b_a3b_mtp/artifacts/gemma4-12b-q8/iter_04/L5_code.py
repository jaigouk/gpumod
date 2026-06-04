from dataclasses import dataclass, field
        from typing import Callable, Any, List
        from collections import deque

        @dataclass
        class Job:
            id: str
            task: Callable
            args: tuple = field(default_factory=tuple)
            kwargs: dict = field(default_factory=dict)
            priority: int = 0

        class JobQueue:
            def __init__(self):
                self._queue = deque()

            def push(self, job: Job):
                self._queue.append(job)

            def pop(self) -> Job:
                return self._queue.popleft() if self._queue else None

            def __len__(self):
                return len(self._queue)