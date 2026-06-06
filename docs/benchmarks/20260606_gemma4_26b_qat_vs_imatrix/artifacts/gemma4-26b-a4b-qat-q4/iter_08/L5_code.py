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
        self._jobs: dict[str, Job] = {}
        self._heap: list[tuple[int, int, str]] = []
        self._counter: int = 0
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._jobs[job_id] = Job(id=job_id, data=data, priority=priority)
        # Use -priority for max-priority (heapq is min-heap)
        # Use self._counter for FIFO order within same priority
        heapq.heappush(self._heap, (-priority, self._counter, job_id))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            # Peek at the highest priority item
            neg_priority, count, job_id = self._heap[0]

            # If job was already processed/removed from the dict, clean up heap
            if job_id in self._jobs:
                return job_id, self._jobs[job_id].data
            else:
                heapq.heappop(self._heap)
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        success, _ = self._retry_policy.run(processor, job.data)

        if success:
            del self._jobs[job_id]

        return success