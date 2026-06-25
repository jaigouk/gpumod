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

    def run(self, fn: Callable, data: dict) -> tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            try:
                fn(data)
                return True, attempts
            except Exception:
                continue
        return False, attempts

class JobQueue:
    def __init__(self):
        self._heap = []
        self._in_flight = {}
        self._counter = 0
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._counter += 1
        # heapq is a min-heap. We use -priority for max-priority behavior
        # and self._counter to ensure FIFO order for same-priority jobs.
        heapq.heappush(self._heap, (-priority, self._counter, job))

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            _, _, job = heapq.heappop(self._heap)
            self._in_flight[job.id] = job
            return job.id, job.data
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._in_flight:
            return False

        job = self._in_flight[job_id]
        success, attempts = self._retry_policy.run(processor, job.data)

        if not success:
            job.retries += attempts

        del self._in_flight[job_id]
        return success