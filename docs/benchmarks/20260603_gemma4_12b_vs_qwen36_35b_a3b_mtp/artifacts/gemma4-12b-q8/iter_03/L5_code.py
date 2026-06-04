from dataclasses import dataclass, field
        from typing import Callable, Any, Optional

        @dataclass
        class Job:
            id: str
            action: Callable[[], Any]
            args: tuple = field(default_factory=tuple)
            kwargs: dict = field(default_factory=dict)
            priority: int = 10 # Default
            retries: int = 0
            max_retries: int = 3

        class JobQueue:
            def __init__(self):
                self.jobs = []
            def add_job(self, job: Job):
                self.jobs.append(job)
            def get_next_job(self) -> Optional[Job]:
                return self.jobs.pop(0) if self.jobs else None