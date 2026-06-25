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
        for _ in range(self.max_attempts):
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
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self.jobs.append(Job(id=job_id, data=data, priority=priority))

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.jobs:
            return None
        best_idx = 0
        for i in range(1, len(self.jobs)):
            if self.jobs[i].priority > self.jobs[best_idx].priority:
                best_idx = i
            elif self.jobs[i].priority == self.jobs[best_idx].priority and i < best_idx:
                best_idx = i
        job = self.jobs.pop(best_idx)
        return job.id, job.data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = next((j for j in self.jobs if j.id == job_id), None)
        if job is None:
            return False
        success, _ = self.retry_policy.run(processor, job.data)
        if not success:
            job.retries += 1
        return success