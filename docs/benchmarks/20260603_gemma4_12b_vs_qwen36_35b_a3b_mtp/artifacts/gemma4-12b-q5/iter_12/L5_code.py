from dataclasses import dataclass, field
        from typing import Any, List
        from collections import deque

        @dataclass
        class Job:
            id: str
            payload: Any
            priority: int = 0
            retries: int = 0
            max_retries: int = 3

        class JobQueue:
            def __init__(self):
                self._queue = deque()
            def push(self, job: Job): ...
            def pop(self) -> Job: ...