from dataclasses import dataclass, field
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
                self.jobs: List[Job] = []
            def push(self, job: Job):
                self.jobs.append(job)
            def pop(self) -> Job:
                return self.jobs.pop(0) if self.jobs else None