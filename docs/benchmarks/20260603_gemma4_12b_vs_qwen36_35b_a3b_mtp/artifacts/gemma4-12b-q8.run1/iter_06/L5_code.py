from dataclasses import dataclass, field
        from typing import Any, List
        import uuid

        @dataclass
        class Job:
            payload: Any
            priority: int = 0
            retries: int = 0
            max_retries: int = 3
            id: str = field(default_factory=lambda: str(uuid.uuid4()))

        class JobQueue:
            def __init__(self):
                self.queue = []
            def push(self, job: Job):
                self.queue.append(job)
            def pop(self) -> Job:
                return self.queue.pop(0) if self.queue else None