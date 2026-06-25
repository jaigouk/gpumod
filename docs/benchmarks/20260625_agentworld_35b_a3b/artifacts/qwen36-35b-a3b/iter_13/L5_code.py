from dataclasses import dataclass
from typing import Callable
import heapq

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
        self._queue = []
        self._counter = 0
        self._job_data = {}
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        heapq.heappush(self._queue, (-priority, self._counter, job_id, data))
        self._job_data[job_id] = data
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if self._queue:
            _, _, job_id, data = heapq.heappop(self._queue)
            return job_id, data
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self._job_data.get(job_id)
        if data is None:
            return False
        success, _ = self.retry_policy.run(processor, data)
        return success