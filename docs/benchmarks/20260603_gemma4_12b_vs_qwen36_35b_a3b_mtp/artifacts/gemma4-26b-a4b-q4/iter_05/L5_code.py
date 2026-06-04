from dataclasses import dataclass, field
        from typing import Any, List

        @dataclass(order=True)
        class Job:
            priority: int
            id: str = field(compare=False)
            payload: Any = field(compare=False)

        class JobQueue:
            def __init__(self):
                self.jobs: List[Job] = []
            def enqueue(self, job: Job): ...
            def dequeue(self) -> Job: ...