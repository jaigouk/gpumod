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

    def run(self, fn: Callable, data) -> tuple[bool, int]:
        attempts = 0
        for attempt in range(self.max_attempts):
            attempts += 1
            try:
                fn(data)
                return True, attempts
            except Exception:
                if attempt < self.max_attempts - 1:
                    # Backoff calculation encapsulated but not executed per instructions
                    # backoff = 2 ** attempt
                    pass
        return False, attempts

class JobQueue:
    def __init__(self):
        self.jobs = []
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self.jobs.append(Job(id=job_id, data=data, priority=priority))

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.jobs:
            return None
        best_job = None
        best_priority = -1
        best_idx = float('inf')
        for i, job in enumerate(self.jobs):
            if job.priority > best_priority or \
               (job.priority == best_priority and i < best_idx):
                best_job = job
                best_priority = job.priority
                best_idx = i
        return (best_job.id, best_job.data)

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = next((j for j in self.jobs if j.id == job_id), None)
        if not job:
            return False
        success, _ = self.retry_policy.run(processor, job.data)
        return success