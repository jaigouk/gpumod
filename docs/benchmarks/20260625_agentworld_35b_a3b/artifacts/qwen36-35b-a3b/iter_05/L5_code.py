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
        self._jobs = []
        self._counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._jobs.append((Job(id=job_id, data=data, priority=priority), self._counter))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._jobs:
            return None
        self._jobs.sort(key=lambda x: (-x[0].priority, x[1]))
        job, _ = self._jobs.pop(0)
        return job.id, job.data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        for i, (job, _) in enumerate(self._jobs):
            if job.id == job_id:
                self._jobs.pop(i)
                policy = RetryPolicy()
                success, _ = policy.run(processor, job.data)
                return success
        return False