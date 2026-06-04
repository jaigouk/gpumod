from dataclasses import dataclass, field
        from typing import Any, List

        @dataclass(order=True) # order=True allows comparison for priority queue
        class Job:
            priority: int
            id: str = field(compare=False)
            payload: Any = field(compare=False)
            attempts: int = field(default=0, compare=False)

        class JobQueue:
            def __init__(self):
                self._jobs: List[Job] = []

            def push(self, job: Job):
                self._jobs.append(job)

            def pop(self) -> Job:
                return self._jobs.pop(0)

            def is_empty(self) -> bool:
                return len(self._jobs) == 0