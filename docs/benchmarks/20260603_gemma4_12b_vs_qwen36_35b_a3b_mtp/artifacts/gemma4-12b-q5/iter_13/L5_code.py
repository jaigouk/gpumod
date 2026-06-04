from dataclasses import dataclass
        from typing import Any

        @dataclass
        class Job:
            id: str
            payload: Any
            priority: int = 0

        class JobQueue:
            def __init__(self):
                self.queue = []
            def push(self, job: Job):
                self.queue.append(job)
            def pop(self) -> Job:
                return self.queue.pop(0) if self.queue else None