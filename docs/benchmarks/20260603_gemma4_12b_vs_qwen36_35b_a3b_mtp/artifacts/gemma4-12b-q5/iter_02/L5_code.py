from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List, Optional
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
            try:
                fn(data)
                return True, attempts_made + 1
            except Exception:
                attempts_made += 1
        return False, attempts_made

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.heap: List[Tuple[int, int, str]] = []
        self.counter = 0
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self.jobs[job_id] = job
        # Use negative priority for max-heap behavior using heapq min-heap
        # Use counter to ensure FIFO for same priority
        heapq.heappush(self.heap, (-priority, self.counter, job_id))
        self.counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False

        success, _ = self.retry_policy.run(processor, job.data)
        if success:
            del self.jobs[job_or_id := job_id]
        return success

    def get_next_job(self) -> Tuple[str, dict] | None:
        while self.heap:
            neg_pri, count, j_id = heapq.heappop(self.heap)
            if j_id in self.jobs:
                job = self.jobs[j_id]
                return (job.id, job.data)
        return None