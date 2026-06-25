import heapq
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0


class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: Any) -> tuple[bool, int]:
        attempts_made = 0
        while attempts_made < self.max_attempts:
            attempts_made += 1
            try:
                fn(data)
                return True, attempts_made
            except Exception:
                continue
        return False, attempts_made


class JobQueue:
    def __init__(self):
        self._jobs = {}
        self._priority_heap = []
        self._counter = 0
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        self._counter += 1
        # Use negative priority for max-heap behavior using heapq (min-heap)
        # Use self._counter as a tie-breaker to ensure FIFO within the same priority
        heapq.heappush(self._priority_heap, (-priority, self._counter, job_id))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        success, _ = self._retry_policy.run(processor, job.data)

        if success:
            del self._jobs[job_id]

        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._priority_heap:
            _, _, job_id = heapq.heappop(self._priority_heap)
            # Check if the job is still in the queue (not processed or removed)
            if job_id in self._jobs:
                job = self._jobs[job_id]
                return job.id, job.data
        return None