from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Any

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable[[dict], Any], data: dict) -> Tuple[bool, int]:
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

    def get_next_job(self) -> Tuple[str, dict] | None:
        if not self.jobs:
            return None
        # Sort by priority descending. Python's sort is stable,
        # preserving FIFO order for jobs with the same priority.
        self.jobs.sort(key=lambda x: x.priority, reverse=True)
        next_job = self.jobs[0]
        return (next_job.id, next_job.data)

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        job = next((j for j in self.jobs if j.id == job_id), None)
        if not job:
            return False

        success, _ = self.retry_policy.run(Processor, job.data)
        if success:
            self.jobs.remove(job)
            return True
        return False