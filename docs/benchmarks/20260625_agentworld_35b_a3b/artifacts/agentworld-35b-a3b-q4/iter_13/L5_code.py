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
        for attempt in range(1, self.max_attempts + 1):
            attempts_made = attempt
            try:
                fn(data)
                return (True, attempts_made)
            except Exception:
                pass
        return (False, attempts_made)


class JobQueue:
    def __init__(self):
        self._job_data: dict[str, dict] = {}
        self.heap: list[tuple[int, int, str]] = []
        self.counter = 0
        self.retry_policy = RetryPolicy(max_attempts=4)

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._job_data[job_id] = data
        heapq.heappush(self.heap, (-priority, self.counter, job_id))
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        while self.heap:
            neg_prio, cnt, job_id = heapq.heappop(self.heap)
            if job_id in self._job_data:
                return (job_id, self._job_data[job_id])
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self._job_data.get(job_id)
        if data is None:
            return False
        success, attempts = self.retry_policy.run(lambda d: processor(d), data)
        if success:
            self._job_data.pop(job_id, None)
        return success