from dataclasses import dataclass
from typing import Callable, Tuple, Optional
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
        attempts_made = 0
        while attempts_made < self.max_attempts:
            try:
                fn(data)
                return True, attempts_made + 1
            except Exception:
                attempts_made += 1
        return False, attempts_made

class JobQueue:
    def __init__(self):
        self._heap = []
        self._job_registry = {}
        self._counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._job_registry[job_id] = job
        # Use negative priority for max-heap behavior in heapq (min-heap)
        # Counter ensures FIFO order for jobs with the same priority
        heapq.heappush(self._heap, (-priority, self._counter, job))
        self._counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._job_registry.get(job_id)
        if not job:
            return False

        policy = RetryPolicy()
        success, _ = policy.run(processor, job.data)
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None

        _, _, job = heapq.heappop(self._heap)
        return job.id, job.data