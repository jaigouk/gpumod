from dataclasses import dataclass
        from typing import List, Optional

        @dataclass
        class Job:
            id: str
            task: str
            priority: int = 0
            retries: int = 0
            max_retries: int = 3

        class JobQueue:
            def __init__(self):
                self.jobs: List[Job] = []
            def push(self, job: Job): self.jobs.append(job)
            def pop(self) -> Optional[Job]: ...