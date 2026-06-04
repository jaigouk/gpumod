from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
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
        self.jobs: Dict[str, Job] = {}
        self.priority_queue = []
        self.counter = 0
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self.jobs[job_id] = job
        # Use negative priority for max-heap behavior and counter for FIFO
        heapq.heappush(self.priority_queue, (-priority, self.counter, job_id))
        self.counter += 1

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False

        success, _ = self.retry_policy.run(Processor, job.data)
        if success:
            self.jobs.pop(job_id)
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        while self.priority_queue:
            _, _, job_id = heapq.heappop(self.priority_queue)
            job = self.jobs.pop(job_id, None)
            if job:
                return (job.id, job.data)
        return None