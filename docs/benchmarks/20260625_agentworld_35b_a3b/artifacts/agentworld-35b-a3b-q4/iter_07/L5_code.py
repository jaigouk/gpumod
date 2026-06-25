from dataclasses import dataclass
from typing import Callable, Dict, Tuple
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
        attempts_made = 0
        for attempt in range(1, self.max_attempts + 1):
            attempts_made = attempt
            try:
                fn(data)
                return (True, attempts_made)
            except Exception:
                pass
        return (False, attempts_made)

class JobQueue:
    def __init__(self):
        self._heap: list = []
        self._sequence = 0
        self._jobs: Dict[str, Job] = {}
        self.retry_policy = RetryPolicy(max_attempts=4)

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority, retries=0)
        self._jobs[job_id] = job
        entry = (-priority, self._sequence, job_id)
        heapq.heappush(self._heap, entry)
        self._sequence += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            neg_priority, sequence, job_id = heapq.heappop(self._heap)
            if job_id in self._jobs:
                job = self._jobs[job_id]
                return (job.id, job.data)
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False
        job = self._jobs[job_id]
        success, _ = self.retry_policy.run(lambda d: processor(d), job.data)
        return success