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
        attempts_made = 0
        for i in range(1, self.max_attempts + 1):
            attempts_made = i
            try:
                fn(data)
                return True, attempts_made
            except Exception:
                continue
        return False, attempts_made

class JobQueue:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._queue: list[tuple[int, int, str]] = []
        self._counter: int = 0
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        # Using -priority for max-heap behavior and self._counter for FIFO within same priority
        heapq.heappush(self._queue, (-priority, self._counter, job_id))
        self._counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False

        success, _ = self._retry_policy.run(processor, job.data)
        if success:
            del self._jobs[job_id]
            return True
        return False

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._queue:
            _, _, job_id = heapq.heappop(self._queue)
            if job_id in self._jobs:
                job = self._jobs[job_id]
                return job.id, job.data
        return None