from dataclasses import dataclass
        from typing import Any, List

        @dataclass
        class Job:
            id: str
            payload: Any
            priority: int = 0
            retries: int = 0
            max_retries: int = 3

        class JobQueue:
            def __init__(self):
                self._items = []
            def push(self, job: Job):
                self._items.append(job)
            def pop(self) -> Job:
                return self._items.pop(0) if self._items else None