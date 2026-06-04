from dataclasses import dataclass
        from typing import List
        @dataclass
        class Job:
            id: str
            task: str
            priority: int = 0
        class JobQueue:
            def __init__(self): self.jobs = []
            def enqueue(self, job): self.jobs.append(job)
            def dequeue(self): return self.jobs.pop(0) if self.jobs else None