from dataclasses import dataclass
        from typing import Any, List

        @dataclass
        class Job:
            id: str
            payload: Any
            priority: int = 0

        class JobQueue:
            def __init__(self):
                self.items: List[Job] = []
            def push(self, job: Job):
                self.items.append(job)
            def pop(self) -> Job:
                return self.items.pop(0) if self.items else None