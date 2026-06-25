from dataclasses import dataclass
from typing import Callable
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
        attempts = 0
        for _ in range(self.max_attempts):
            attempts += 1
            try:
                fn(data)
                return True, attempts
            except Exception:
                pass
        return False, attempts

class JobQueue:
    def __init__(self):
        self._heap: list = []
        self._counter: int = 0
        self._jobs: dict[str, Job] = {}

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        heapq.heappush(self._heap, (-priority, self._counter, job))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None
        _, _, job = heapq.heappop(self._heap)
        self._jobs.pop(job.id, None)
        return job.id, job.data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False
        job = self._jobs[job_id]
        policy = RetryPolicy()
        success, attempts = policy.run(processor, job.data)
        job.retries = attempts
        if success:
            self._jobs.pop(job_id, None)
        return success