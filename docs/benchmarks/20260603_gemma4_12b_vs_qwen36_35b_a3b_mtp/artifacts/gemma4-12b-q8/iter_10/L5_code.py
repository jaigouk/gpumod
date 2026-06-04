from dataclasses import dataclass
        from typing import Any, List
        from collections import deque

        @dataclass
        class Job:
            id: str
            payload: Any
            priority: int = 0
            retries: int = 0

        class JobQueue:
            def __init__(self):
                self._queue = deque()
            def push(self, job: Job):
                self._queue.append(job)
            def pop(self) -> Job:
                return self._queue.popleft()
            def __len__(self):
                return len(self._queue)