from dataclasses import dataclass, field
        from typing import Any

        @dataclass(order=True) # order=True helps with priority later
        class Job:
            priority: int
            id: str = field(compare=False)
            payload: Any = field(compare=False)
            retries: int = field(default=0, compare=False)

        class JobQueue:
            def __init__(self):
                self._jobs = []
            def push(self, job: Job):
                self._jobs.append(job)
            def pop(self) -> Job:
                return self._jobs.pop(0)
            def is_empty(self):
                return len(self._jobs) == 0