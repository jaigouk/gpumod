from dataclasses import dataclass, field
        from typing import Any, List

        @dataclass(order=True)
        class Job:
            priority: int
            id: str = field(compare=False)
            data: Any = field(compare=False)
            retries: int = field(default=0, compare=False)

        class JobQueue:
            def __init__(self):
                self._jobs = []
            def push(self, job: Job): ...
            def pop(self) -> Job: ...
            def is_empty(self) -> bool: ...