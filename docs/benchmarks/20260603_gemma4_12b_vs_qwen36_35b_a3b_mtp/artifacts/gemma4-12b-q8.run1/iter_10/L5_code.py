from dataclasses import dataclass, field
        from typing import Any, List

        @dataclass(order=False)
        class Job:
            id: str
            payload: Any
            priority: int = 0
            retries: int = 0
            max_retries: int = 3

        class JobQueue:
            def __init__(self):
                self.items: List[Job] = []
            def push(self, job: Job): self.items.append(job)
            def pop(self) -> Job: return self.items.pop(0)
            def __len__(self): return len(self.items)