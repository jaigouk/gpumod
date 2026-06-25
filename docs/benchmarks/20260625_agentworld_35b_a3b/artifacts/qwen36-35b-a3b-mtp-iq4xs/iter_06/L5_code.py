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
        while attempts < self.max_attempts:
            attempts += 1
            try:
                fn(data)
                return True, attempts
            except Exception:
                pass
        return False, attempts

class JobQueue:
    def __init__(self):
        self.jobs: list[Job] = []

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self.jobs.append(Job(id=job_id, data=data, priority=priority))

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.jobs:
            return None
        best = max(self.jobs, key=lambda j: j.priority)
        return best.id, best.data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = next((j for j in self.jobs if j.id == job_id), None)
        if not job:
            return False
        policy = RetryPolicy()
        success, _ = policy.run(processor, job.data)
        return success