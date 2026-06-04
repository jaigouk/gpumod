from dataclasses import dataclass, field
        from typing import Any

        @dataclass(order=False)
        class Job:
            id: str
            payload: Any
            priority: int = 10  # Lower is higher priority

        class JobQueue:
            def __init__(self):
                self._items = []
            def push(self, job: Job):
                self._items.append(job)
            def pop(self) -> Job:
                return self._items.pop(0)
            def is_empty(self) -> bool:
                return len(self._items) == 0