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

    def run(self, fn: Callable, data: dict) -> tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                fn(data)
                return (True, attempts + 1)
            except Exception:
                attempts += 1
        return (False, attempts)

class JobQueue:
    def __init__(self):
        self.jobs: List[Job] = []
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        new_job = Job(id=job_id, data=data, priority=priority)
        self.jobs.append(new_job)

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.jobs:
            return None

        max_priority = max(job.priority for job in self.jobs)
        for job in self.jobs:
            if job.priority == max_priority:
                return (job.id, job.data)
        return None

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        job = next((j for j in self.jobs if j.id == job_id), None)
        if job is None:
            return False

        success, _ = self.retry_policy.run(Processor, job.data)
        return success