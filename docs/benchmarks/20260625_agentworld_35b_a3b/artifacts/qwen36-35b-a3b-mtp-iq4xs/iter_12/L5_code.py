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
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._jobs.append({'id': job_id, 'data': data, 'priority': priority, 'order': self._counter})
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._jobs:
            return None
        best = max(self._jobs, key=lambda j: (j['priority'], -j['order']))
        self._jobs.remove(best)
        return best['id'], best['data']

    def process_job(self, job_id: str, processor: Callable) -> bool:
        for job in self._jobs:
            if job['id'] == job_id:
                success, _ = self._retry_policy.run(processor, job['data'])
                return success
        return False