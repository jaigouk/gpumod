from dataclasses import dataclass
from typing import Callable, Tuple, Dict
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

    def run(self, fn: Callable, data: dict) -> Tuple[bool, int]:
        attempts_made = 0
        while attempts_made < self.max_attempts:
            attempts_made += 1
            try:
                fn(data)
                return True, attempts_made
            except Exception:
                continue
        return False, attempts_made

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.queue = []
        self.counter = 0
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self.jobs[job_id] = job
        # Use negative priority for max-heap behavior and counter for FIFO
        heapq.heappush(self.queue, (-priority, self.counter, job))
        self.counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False

        success, attempts = self.retry_policy.run(processor, job.data)
        job.retries = attempts
        return success

    def get_next_job(self) -> Tuple[str, dict] | None:
        if not self.queue:
            return None
        _, _, job = heapq.heappop(self.queue)
        return job.id, job.data