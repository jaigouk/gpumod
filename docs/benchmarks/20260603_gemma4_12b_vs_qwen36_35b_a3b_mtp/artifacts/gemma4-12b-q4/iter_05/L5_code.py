from dataclasses import dataclass
from typing import Callable

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: dict) -> tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                fn(data)
                return True, attempts + 1
            except Exception:
                attempts += 1
        return False, attempts

class JobQueue:
    def __init__(self):
        self.jobs: list[Job] = []
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self.jobs.append(Job(id=job_id, data=data, priority=priority)))

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        job = next((j for j in self.jobs if j.id == job_id), None)
        if job is None:
            return False

        success, attempts = self.retry_policy.run(Processor, job.data)
        job.retries = attempts
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.jobs:
            return None

        # max() returns the first occurrence of the maximum value.
        # Since jobs are appended, the first occurrence satisfies FIFO.
        best_job = max(self.jobs, key=lambda j: j.priority)
        self.jobs.remove(best_job)
        return (best_job.id, best_job.data)