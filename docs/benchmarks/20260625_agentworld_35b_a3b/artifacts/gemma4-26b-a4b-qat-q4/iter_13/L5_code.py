from dataclasses import dataclass
from typing import Callable, Any
import heapq
import itertools

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
                continue
        return False, attempts

class JobQueue:
    def __init__(self):
        self._heap = []
        self._jobs = {}
        self._counter = itertools.count()
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        # heapq is a min-heap. We use -priority to simulate a max-priority queue.
        # We use self._counter to ensure FIFO order for jobs with the same priority.
        heapq.heappush(self._heap, (-priority, next(self._counter), job_id))

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            _, _, job_id = heapq.heappop(self._heap)
            # Check if the job still exists in our tracking dictionary
            if job_id in self._jobs:
                job = self._jobs[job_id]
                return job.id, job.data
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        success, _ = self._retry_policy.run(processor, job.data)

        if success:
            del self._jobs[job_id]

        return success