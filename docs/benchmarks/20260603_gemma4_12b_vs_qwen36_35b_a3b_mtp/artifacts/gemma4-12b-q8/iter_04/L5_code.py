from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List
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
        self.jobs: List[Tuple[int, int, Job]] = []
        self.counter = 0
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        # Use negative priority for max-heap behavior with heapq (min-heap)
        # counter ensures FIFO order for identical priorities
        heapq.heappush(self.jobs, (-priority, self.counter, job))
        self.counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        for _, _, job in self.jobs:
            if job.id == job_id:
                success, _ = self.retry_policy.run(processor, job.data)
                return success
        return False

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.jobs:
            return None
        _, _, job = heapq.heappop(self.jobs)
        return (job.id, job.data)