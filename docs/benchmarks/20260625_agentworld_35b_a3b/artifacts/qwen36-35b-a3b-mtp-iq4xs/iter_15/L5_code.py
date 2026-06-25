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
        self._queue: list[tuple[int, Job]] = []
        self._counter = 0
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._counter += 1
        self._queue.append((self._counter, Job(id=job_id, data=data, priority=priority, retries=0)))

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None
        self._queue.sort(key=lambda x: (-x[1].priority, x[0]))
        _, job = self._queue.pop(0)
        return (job.id, job.data)

    def process_job(self, job_id: str, processor: Callable) -> bool:
        for _, job in self._queue:
            if job.id == job_id:
                success, _ = self.retry_policy.run(lambda data: processor(data), job.data)
                return success
        return False