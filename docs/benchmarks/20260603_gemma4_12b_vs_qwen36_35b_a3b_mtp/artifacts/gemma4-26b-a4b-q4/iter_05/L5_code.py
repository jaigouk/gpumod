from dataclasses import dataclass, field
        from typing import Any, List

        @dataclass(order=True) # order=True helps with priority comparison if needed
        class Job:
            priority: int
            job_id: str = field(compare=False)
            payload: Any = field(compare=False)
            retries: int = field(default=0, compare=False)

        class JobQueue:
            def __init__(self):
                self.jobs: List[Job] = []

            def push(self, job: Job):
                self.jobs.append(job)

            def pop(self) -> Job:
                return self.jobs.pop(0)

            def is_empty(self) -> bool:
                return len(self.jobs) == 0