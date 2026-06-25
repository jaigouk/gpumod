from dataclasses import dataclass
from typing import Callable, Tuple, Optional

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
        self._jobs = []
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._jobs.append(Job(id=job_id, data=data, priority=priority))

    def get_next_job(self) -> Tuple[str, dict] | None:
        if not self._jobs:
            return None
        self._jobs.sort(key=lambda x: -x.priority)
        job = self._jobs.pop(0)
        return job.id, job.data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = next((j for j in self._jobs if j.id == job_id), None)
        if job is None:
            return False
        success, _ = self.retry_policy.run(processor, job.data)
        return success