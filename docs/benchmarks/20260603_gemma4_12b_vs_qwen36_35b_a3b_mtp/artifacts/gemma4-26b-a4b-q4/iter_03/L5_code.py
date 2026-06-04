from dataclasses import dataclass, field
        from typing import Any, List

        @dataclass(order=False)
        class Job:
            id: str
            payload: Any
            priority: int = 0

            def __lt__(self, other):
                # For PriorityQueue: lower number = higher priority?
                # Or higher number = higher priority?
                # Usually, in heaps, smaller is higher priority.
                return self.priority < other.priority

        class JobQueue:
            def __init__(self):
                self._jobs: List[Job] = []
            def push(self, job: Job):
                self._jobs.append(job)
            def pop(self) -> Job:
                return self._jobs.pop(0)
            def is_empty(self) -> bool:
                return len(self._jobs) == 0