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
        self._jobs: list[Job] = []
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._jobs.append(Job(id=job_id, data=data, priority=priority, retries=0))

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._jobs:
            return None
        max_priority = max(j.priority for j in self._jobs)
        for j in self._jobs:
            if j.priority == max_priority:
                return j.id, j.data
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        for j in self._jobs:
            if j.id == job_id:
                success, _ = self._retry_policy.run(processor, j.data)
                if success:
                    self._jobs.remove(j)
                return success
        return False