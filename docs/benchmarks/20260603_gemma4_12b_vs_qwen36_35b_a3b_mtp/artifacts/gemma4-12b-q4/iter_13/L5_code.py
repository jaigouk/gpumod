from dataclasses import dataclass
        from typing import Any, List

        @dataclass
        class Job:
            id: str
            data: Any
            priority: int = 0

        class JobQueue:
            def __init__(self):
                self.jobs: List[Job] = []
            def enqueue(self, job: Job):
                self.jobs.append(job)
            def dequeue(self) -> Job:
                return self.jobs.pop(0) if self.jobs else None