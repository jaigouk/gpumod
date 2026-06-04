from dataclasses import dataclass
from typing import Callable, Any
import heapq
from itertools import count

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
        for _ in range(self.max_attempts):
            attempts_made += 1
            try:
                fn(data)
                return True, attempts_made
            except Exception:
                continue
        return False, attempts_made

class JobQueue:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._queue = []
        self._counter = count()
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        # Use negative priority for max-heap behavior and counter for FIFO tie-breaking
        heapq.heappush(self._queue, (-priority, next(self._counter), job_id))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False

        success, _ = self._retry_policy.run(processor, job.data)
        if success:
            del self._jobs[job_id]
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._queue:
            _, _, job_id = heapq.heappop(self._queue)
            job = self._jobs.get(job_id)
            if job:
                return job_id, job.data
        return None