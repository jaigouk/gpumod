from dataclasses import dataclass
        from typing import Any, List
        from collections import deque

        @dataclass
        class Job:
            id: str
            data: Any
            priority: int = 0
            retries: int = 0

        class JobQueue:
            def __init__(self):
                self._queue = deque()
            def push(self, job: Job): ...
            def pop(self) -> Job: ...