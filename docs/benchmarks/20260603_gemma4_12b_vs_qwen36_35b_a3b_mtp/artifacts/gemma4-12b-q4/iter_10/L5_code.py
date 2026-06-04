from dataclasses import dataclass
        from typing import Any, List
        from collections import deque

        @dataclass
        class Job:
            id: str
            data: Any
            priority: int = 0
            retries: int = 0
            max_retries: int = 3

        class JobQueue:
            def __init__(self):
                self._queue = deque()
            def push(self, job: Job):
                self._queue.append(job)
            def pop(self) -> Job:
                return self._queue.popleft()
            def __len__(self):
                return len(self._queue)