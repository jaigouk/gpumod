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
        for attempt in range(1, self.max_attempts + 1):
            try:
                fn(data)
                return True, attempt
            except Exception:
                pass
        return False, self.max_attempts

class JobQueue:
    def __init__(self):
        self._jobs = []
        self._counter = 0
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._jobs.append((-priority, self._counter, Job(id=job_id, data=data, priority=priority)))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._jobs:
            return None
        self._jobs.sort()
        _, _, job = self._jobs.pop(0)
        return job.id, job.data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        for i, item in enumerate(self._jobs):
            if item[2].id == job_id:
                job = item[2]
                self._jobs.pop(i)
                success, _ = self._retry_policy.run(processor, job.data)
                return success
        return False