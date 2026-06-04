from dataclasses import dataclass
        from typing import Any, List

        @dataclass
        class Job:
            id: str
            data: Any
            priority: int = 0

        class JobQueue:
            def __init__(self):
                self.queue: List[Job] = []
            def push(self, job: Job):
                self.queue.append(job)
            def pop(self) -> Job:
                return self.queue.pop(0) if self.queue else None