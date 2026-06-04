from dataclasses import dataclass
        from typing import List
        @dataclass
        class Job:
            id: str
            payload: dict
            priority: int = 0
        class JobQueue:
            def __init__(self): self.items = []
            def push(self, job): self.items.append(job)
            def pop(self): return self.items.pop(0) if self.items else None