from dataclasses import dataclass
        from typing import Callable, Any

        @dataclass
        class Job:
            id: str
            func: Callable
            args: tuple = ()
            kwargs: dict = None

        class JobQueue:
            def __init__(self):
                self.items = []
            def push(self, job: Job):
                self.items.append(job)
            def pop(self) -> Job:
                return self.items.pop(0)