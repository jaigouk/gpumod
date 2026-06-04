from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
import heapq

@dataclass
class Job:
    id: str
    data: Dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: Dict) -> Tuple[bool, int]:
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
        self.jobs: Dict[str, Job] = {}
        self.pq: list[Tuple[int, int, str]] = []
        self.counter = 0
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: Dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self.jobs[job_id] = job
        # Use negative priority for max-heap behavior with heapq (min-heap)
        # Counter ensures FIFO for same priority
        heapq.heappush(self.pq, (-priority, self.counter, job_id))
        self.counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False

        success, attempts = self.retry_policy.run(processor, job.data)
        job.retries = attempts

        # Remove from active jobs once processed (attempts exhausted)
        if job_id in self.jobs:
            del self.jobs[job_id]

        return success

    def get_next_job(self) -> Tuple[str, Dict] | None:
        while self.pq:
            _, _, job_id = heapq.heappop(self.pq)
            if job_id in self.jobs:
                job = self.jobs[job_id]
                return (job.id, job.data)
        return None