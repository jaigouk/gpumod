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
                continue
        return False, self.max_attempts

class JobQueue:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._pq = []
        self._counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        self._counter += 1
        # Use -priority for max-heap behavior and self._counter for FIFO tie-breaking
        heapq.heappush(self._pq, (-priority, self._counter, job_id))

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._pq:
            _, _, job_id = heapq.heappop(self._pq)
            if job_id in self._jobs:
                job = self._jobs[job_id]
                return job.id, job.data
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        policy = RetryPolicy()
        success, attempts = policy.run(processor, job.data)

        if success:
            del self._jobs[job_id]
        else:
            job.retries = attempts

        return success