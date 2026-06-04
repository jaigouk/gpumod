from dataclasses import dataclass
        from typing import Any, List

        @dataclass
        class Job:
            id: str
            payload: Any
            retries: int = 0
            priority: int = 0

        class JobQueue:
            def __init__(self):
                self.jobs: List[Job] = []
            def add(self, job: Job):
                self.jobs.append(job)
            def get(self) -> Job:
                return self.jobs.pop(0) if self.jobs else None