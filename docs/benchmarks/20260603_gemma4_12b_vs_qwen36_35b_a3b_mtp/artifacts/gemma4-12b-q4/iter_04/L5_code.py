from dataclasses import dataclass
        from typing import Any, List

        @dataclass
        class Job:
            id: str
            payload: Any
            priority: int = 0

        class JobQueue:
            def __init__(self):
                self.jobs: List[Job] = []
            def push(self, job: Job): self.jobs.append(job)
            def pop(self) -> Job: return self.jobs.pop(0)