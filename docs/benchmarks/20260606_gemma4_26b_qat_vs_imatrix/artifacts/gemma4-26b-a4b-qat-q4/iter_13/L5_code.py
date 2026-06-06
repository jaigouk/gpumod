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
        # Use negative priority for max-heap behavior and counter for FIFO within priority
        heapq.heappush(self._heap, (-priority, self._counter, job_id))
        self._counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False

        success, attempts = self._retry_policy.run(processor, job.data)
        if success:
            del self._jobs[job_id]
            return True
        else:
            job.retries = attempts
            # Re-enqueue the job to allow future attempts
            self._counter += 1
            heapq.heappush(self._heap, (-job.priority, self._counter, job_id))
            return False

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            _, _, job_id = heapq.heappop(self._heap)
            # Check if job exists (it might have been processed/removed)
            if job_id in self._jobs:
                job = self._jobs[job_id]
                return job.id, job.data
        return None