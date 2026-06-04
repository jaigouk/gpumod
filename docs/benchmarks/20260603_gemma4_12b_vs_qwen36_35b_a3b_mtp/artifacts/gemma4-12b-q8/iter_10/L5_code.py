from dataclasses import dataclass
from typing import Callable, Dict, List
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
        self.jobs: Dict[str, Job] = {}
        self.heap: List[tuple[int, int, str]] = []
        self.counter = 0
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self.jobs[job_id] = job
        # Use negative priority for max-heap behavior with heapq (min-heap)
        # Use counter to ensure FIFO order for equal priorities
        heapq.heappush(self.heap, (-priority, self.counter, job_id))
        self.counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        success, attempts_made = self.retry_policy.run(processor, job.data)
        job.retries = attempts_made

        # Remove from registry as it has been processed/attempted
        del self.jobs[job_id]
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        while self.heap:
            _, _, job_id = heapq.heappop(self.heap)
            if job_id in self.jobs:
                job = self.jobs.pop(job_id)
                return (job.id, job.data)
        return None