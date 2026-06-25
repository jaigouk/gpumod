from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0


class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data) -> Tuple[bool, int]:
        for attempt in range(1, self.max_attempts + 1):
            try:
                fn(data)
                return True, attempt
            except Exception:
                pass
        return False, self.max_attempts


class JobQueue:
    def __init__(self):
        self.jobs = []
        self.policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self.jobs.append(Job(id=job_id, data=data, priority=priority))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        for job in self.jobs:
            if job.id == job_id:
                success, _ = self.policy.run(processor, job.data)
                return success
        return False

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.jobs:
            return None
        max_priority = max(job.priority for job in self.jobs)
        for job in self.jobs:
            if job.priority == max_priority:
                return job.id, job.data
        return None