from dataclasses import dataclass, field
        from typing import Any, List

        @dataclass
        class Job:
            id: str
            data: Any
            priority: int = 0

        class JobQueue:
            def __init__(self):
                self._jobs = []
            def push(self, job: Job):
                self._jobs.append(job)
            def pop(self) -> Job:
                return self._jobs.pop(0)
            def is_empty(self) -> bool:
                return len(self._jobs) == 0