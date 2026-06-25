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
        for attempt in range(1, self.max_attempts + 1):
            try:
                fn(data)
                return True, attempt
            except Exception:
                pass
        return False, self.max_attempts

class JobQueue:
    def __init__(self):
        self._queue: list = []
        self._counter: int = 0
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        heapq.heappush(self._queue, (-priority, self._counter, job_id, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None
        _, _, job_id, data = heapq.heappop(self._queue)
        return job_id, data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        for _, _, j_id, j_data in self._queue:
            if j_id == job_id:
                success, _ = self._retry_policy.run(processor, j_data)
                return success
        return False