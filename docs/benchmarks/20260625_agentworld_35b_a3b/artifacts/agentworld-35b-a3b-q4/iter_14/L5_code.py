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

    def run(self, fn: Callable, data) -> tuple[bool, int]:
        attempts_made = 0
        for _ in range(self.max_attempts):
            attempts_made += 1
            try:
                fn(data)
                return (True, attempts_made)
            except Exception:
                pass
        return (False, attempts_made)


class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._pq: list = []
        self._counter = 0
        self.retry_policy = RetryPolicy(max_attempts=4)

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        if job_id not in self._jobs:
            self._jobs[job_id] = Job(id=job_id, data=data, priority=priority, retries=0)
        else:
            job = self._jobs[job_id]
            job.data = data
            job.priority = priority
            job.retries = 0
        entry = (-priority, self._counter, job_id)
        heapq.heappush(self._pq, entry)
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._pq:
            neg_priority, counter, job_id = heapq.heappop(self._pq)
            if job_id in self._jobs:
                job = self._jobs[job_id]
                return (job.id, job.data)
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False
        job = self._jobs[job_id]
        success, attempts = self.retry_policy.run(processor, job.data)
        if not success:
            job.retries += 1
        return success