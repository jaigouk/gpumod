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
            try:
                fn(data)
                return True, attempts + 1
            except Exception:
                attempts += 1
        return False, attempts

class JobQueue:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.heap: list[tuple[int, int, str]] = []
        self.counter = 0
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self.jobs[job_id] = job
        # Use negative priority for max-heap behavior with heapq (min-heap)
        # Counter ensures FIFO order for items with the same priority
        heapq.heappush(self.heap, (-priority, self.counter, job_id))
        self.counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False

        success, attempts = self.retry_policy.run(processor, job.data)
        job.retries = attempts
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.heap:
            return None

        _, _, job_id = heapq.heappop(self.heap)
        job = self.jobs.get(job_id)
        if job:
            return (job.id, job.data)
        return None