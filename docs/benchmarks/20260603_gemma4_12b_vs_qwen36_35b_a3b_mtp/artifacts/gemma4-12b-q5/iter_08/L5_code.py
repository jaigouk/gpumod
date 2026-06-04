from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, Optional


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
        self.jobs: List[Job] = []
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self.jobs.append(Job(id=job_id, data=data, priority=priority))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = next((j for j in self.jobs if j.id == job_id), None)
        if job is None:
            return False

        success, attempts = self.retry_policy.run(processor, job.data)
        job.retries = attempts
        return success

    def get_next_job(self) -> Tuple[str, dict] | None:
        if not self.jobs:
            return None

        # Python's sort is stable, preserving FIFO order for items with same priority
        self.jobs.sort(key=lambda x: x.priority, reverse=True)
        job = self.jobs.pop(0)
        return (job.id, job.data)