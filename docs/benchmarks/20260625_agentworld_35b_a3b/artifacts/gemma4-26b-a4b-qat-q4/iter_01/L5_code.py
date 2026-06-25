from dataclasses import dataclass
from typing import Callable, Any
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

    def run(self, fn: Callable, data: Any) -> tuple[bool, int]:
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
        self._jobs = {}
        self._heap = []
        self._counter = 0
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        # Use -priority for max-priority behavior with heapq's min-heap
        # Use self._counter to maintain FIFO order for same priority
        heapq.heappush(self._heap, (-priority, self._counter, job_id))
        self._counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False

        success, attempts = self._retry_policy.run(processor, job.data)

        if not success:
            job.retries += attempts

        # Remove from active jobs so get_next_job can progress
        del self._jobs[job_id]
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            # Peek at the highest priority item
            _, _, job_id = self._heap[0]

            if job_id in self._jobs:
                job = self._jobs[job_id]
                return job.id, job.data
            else:
                # Clean up stale job IDs from the heap
                heapq.heappop(self._heap)
        return None