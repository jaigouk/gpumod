import heapq
import itertools
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

    def run(self, fn: Callable, data: dict) -> tuple[bool, int]:
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
        self._jobs: dict[str, Job] = {}
        self._heap = []
        self._counter = itertools.count()
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        # heapq is a min-heap; use negative priority to treat higher values as higher priority
        heapq.heappush(self._heap, (-priority, next(self._counter), job_id))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False

        success, attempts = self._retry_policy.run(processor, job.data)

        if success:
            del self._jobs[job_id]
            return True
        else:
            job.retries += attempts
            # Re-queue the job so get_next_job can pick it up again
            heapq.heappush(self._heap, (-job.priority, next(self._counter), job_id))
            return False

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            _, _, job_id = heapq.heappop(self._heap)
            if job_id in self._jobs:
                return job_id, self._jobs[job_id].data
        return None