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

    def run(self, fn: Callable[[dict]], data: dict) -> tuple[bool, int]:
        attempts_made = 0
        while attempts_made < self.max_attempts:
            try:
                fn(data)
                return True, attempts_made + 1
            except Exception:
                attempts_made += 1
        return False, attempts_made

class JobQueue:
    def __init__(self):
        self.jobs = []
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        new_job = Job(id=job_id, data=data, priority=priority)
        self.jobs.append(new_job)

    def process_job(self, job_id: str, Processor: Callable[[dict]]) -> bool:
        # Locate the specific job by ID
        job = next((j for j in self.jobs if j.id == job_id), None)
        if not job:
            return False

        # Execute using the RetryPolicy
        success, attempts = self.retry_policy.run(Processor, job.data)
        job.retries = attempts

        # Remove from queue if successfully processed
        if success:
            self.jobs.remove(job)

        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.jobs:
            return None

        # highest-priority (max value), then FIFO (first occurrence in list)
        best_job = max(self.jobs, key=lambda j: j.priority)
        return (best_job.id, best_job.data)