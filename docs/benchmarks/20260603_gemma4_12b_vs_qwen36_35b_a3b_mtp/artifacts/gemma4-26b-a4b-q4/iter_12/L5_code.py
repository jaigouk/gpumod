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

    def run(self, fn: Callable, data: dict) -> Tuple[bool, int]:
        for attempt in range(1, self.max_attempts + 1):
            try:
                fn(data)
                return True, attempt
            except Exception:
                continue
        return False, self.max_attempts

class JobQueue:
    def __init__(self, max_retries: int = 4):
        self._jobs: Dict[str, Job] = {}
        self._heap: List[Tuple[int, int, str]] = []
        self._counter = 0
        self._retry_policy = RetryPolicy(max_attempts=max_retries)

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        # Use negative priority for max-heap behavior; use counter for FIFO
        heapq.heappush(self._heap, (-priority, self._counter, job_id))
        self._counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False

        success, attempts = self._retry_policy.run(processor, job.data)
        job.retries += attempts

        if success:
            del self._jobs[job_id]
        return success

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        while self._heap:
            _, _, job_id = heapq.heappop(self._heap)
            job = self._jobs.get(job_id)
            if job:
                return job.id, job.data
        return None