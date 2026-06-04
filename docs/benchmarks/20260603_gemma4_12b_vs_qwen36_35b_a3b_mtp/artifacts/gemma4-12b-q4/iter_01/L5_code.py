from dataclasses import dataclass, field
        from typing import Any, List
        from collections import deque

        @dataclass
        class Job:
            id: str
            task: callable
            priority: int = 0
            retries: int = 0
            max_retries: int = 3

        class JobQueue:
            def __init__(self):
                self._jobs = deque()

            def push(self, job: Job):
                self._jobs.append(job)

            def pop(self) -> Job:
                return self._jobs.popleft() if self._jobs else None

            def is_empty(self) -> bool:
                return len(self._jobs) == 0