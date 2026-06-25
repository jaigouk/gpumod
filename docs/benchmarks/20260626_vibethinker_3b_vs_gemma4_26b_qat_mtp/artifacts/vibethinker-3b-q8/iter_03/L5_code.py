from dataclasses import dataclass
from typing import Callable, Tuple, Any, Dict, Optional, List
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
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            try:
                fn(data)
                return (True, attempts)
            except Exception:
                # Simulate backoff (no actual sleep)
                pass
        return (False, self.max_attempts)

class JobQueue:
    def __init__(self):
        self._heap: List[Tuple[int, int, "Job"]] = []
        self._counter: int = 0
        self._retry_policy = RetryPolicy(max_attempts=4)

    def add_job(self, job_id: str, data: Dict, priority: int = 0) -> None:
        job = Job(job_id, data=data, priority=priority)
        self._counter += 1
        heapq.heappush(self._heap, (priority, self._counter, job))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if not self._heap:
            return False
        priority, counter, job = heapq.heappop(self._heap)
        # Optionally verify job.id == job_id
        success, attempts = self._retry_policy.run(processor, job.data)
        return success

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        if not self._heap:
            return None
        priority, counter, job = self._heap[0]
        return (job.id, job.data)