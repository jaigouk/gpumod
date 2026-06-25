from dataclasses import dataclass
from typing import Callable, Tuple
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
        self._jobs: dict[str, Job] = {}
        self._heap: list = []
        self._counter = 0
        self.retry_policy = RetryPolicy(max_attempts=4)

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority, retries=0)
        self._jobs[job_id] = job
        entry = (-priority, self._counter, job_id)
        self._counter += 1
        heapq.heappush(self._heap, entry)

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            neg_priority, counter, job_id = heapq.heappop(self._heap)
            job = self._jobs.get(job_id)
            if job is not None:
                return (job.id, job.data)
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        success, attempts = self.retry_policy.run(processor, job.data)
        if success:
            if job_id in self._jobs:
                del self._jobs[job_id]
        return success