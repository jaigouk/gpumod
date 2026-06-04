from dataclasses import dataclass, field
        from typing import Callable, Any, List

        @dataclass
        class Job:
            id: str
            func: Callable
            args: tuple = field(default_factory=tuple)
            retries: int = 0
            max_retries: int = 3
            priority: int = 0

        class JobQueue:
            def __init__(self):
                self.items: List[Job] = []
            def enqueue(self, job: Job):
                self.items.append(job)
            def dequeue(self) -> Job:
                return self.items.pop(0) if self.items else None
            def __len__(self):
                return len(self.items)