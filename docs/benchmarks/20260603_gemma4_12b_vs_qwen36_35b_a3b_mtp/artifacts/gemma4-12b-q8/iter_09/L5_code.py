from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Dict, List
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

    def run(self, fn: Callable[[dict]], data: dict) -> Tuple[bool, int]:
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
        self.heap: List[Tuple[int, int, str]] = []
        self.counter = 0
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self.jobs[job_id] = job
        # Use negative priority for max-heap behavior (higher number = higher priority)
        # Use counter to ensure FIFO order for equal priorities
        heapq.heappush(self.heap, (-priority, self.counter, job_id))
        self.counter += 1

    def process_job(self, job_id: str, processor: Callable[[dict]]) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        success, _ = self.retry_policy.run(processor, job.data)
        return success

    def get_next_job(self) -> Optional[Tuple[str, dict]] | None:
        while self.heap:
            _, _, job_id = heapq.heappop(self.heap)
            if job_id in self.jobs:
                job = self.jobs[job_id]
                return job_id, job.data
        return None