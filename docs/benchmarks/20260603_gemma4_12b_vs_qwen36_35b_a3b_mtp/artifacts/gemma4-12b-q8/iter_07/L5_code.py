from dataclasses import dataclass, field
        from typing import Callable, Any
        import uuid

        @dataclass(order=False)
        class Job:
            task: Callable
            args: tuple = field(default_factory=tuple)
            priority: int = 0
            retries: int = 0
            max_retries: int = 3
            id: str = field(default_factory=lambda: str(uuid.uuid4()))

            def execute(self) -> Any:
                return self.task(*self.args)

        class JobQueue:
            def __init__(self):
                self.queue = []

            def push(self, job: Job):
                self.queue.append(job)

            def pop(self) -> Job:
                return self.queue.pop(0) if self.queue else None