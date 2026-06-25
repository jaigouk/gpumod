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
        attempts_made = 0
        for attempt in range(self.max_attempts):
            attempts_made = attempt + 1
            try:
                fn(data)
                return (True, attempts_made)
            except Exception:
                pass
        return (False, attempts_made)

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0
        self._job_data: dict[str, dict] = {}
        self.retry_policy = RetryPolicy(max_attempts=4)

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._job_data[job_id] = data
        heapq.heappush(self._heap, (-priority, self._counter, job_id))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None
        neg_priority, counter, job_id = heapq.heappop(self._heap)
        data = self._job_data.get(job_id)
        return (job_id, data)

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self._job_data.get(job_id)
        if data is None:
            raise ValueError(f"Job {job_id} not found")
        success, _ = self.retry_policy.run(lambda d: processor(d), data)
        return success