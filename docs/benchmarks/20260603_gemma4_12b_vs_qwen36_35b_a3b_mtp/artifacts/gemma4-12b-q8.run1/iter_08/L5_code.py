from dataclasses import dataclass, field
        from typing import Any, List

        @dataclass(order=False)
        class Job:
            id: str
            payload: Any
            priority: int = 0

            def __lt__(self, other):
                return self.priority < other.priority

        class JobQueue:
            def __init__(self):
                self.items: List[Job] = []
            def enqueue(self, job: Job):
                self.items.append(job)
            def dequeue(self) -> Job:
                return self.items.pop(0) if self.items else None