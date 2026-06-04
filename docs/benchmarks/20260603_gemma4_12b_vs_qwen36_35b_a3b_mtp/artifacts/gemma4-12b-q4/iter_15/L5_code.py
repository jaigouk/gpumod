from dataclasses import dataclass, field
        from typing import Any, List

        @dataclass
        class Job:
            id: str
            data: Any
            priority: int = 0

        class JobQueue:
            def __init__(self):
                self.items: List[Job] = []
            def push(self, job: Job): ...
            def pop(self) -> Job: ...