from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, List

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: dict) -> Tuple[bool, int]:
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
        self.jobs: List[Job] = []
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

    def get_next_job(self) -> Optional[Tuple[str, dict]] :
        if not self.jobs:
            return None

        # max() returns the first occurrence of the maximum value.
        # Since items are appended, the first occurrence is the oldest (FIFO).
        best_job = max(self.jobs, key=lambda j: j.priority)
        return (best_job.id, best_job.data)