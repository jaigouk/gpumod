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
        for attempt in range(1, self.max_attempts + 1):
            try:
                fn(data)
                return True, attempt
            except Exception:
                if attempt == self.max_attempts:
                    return False, attempt

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0
        self._jobs = {}

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        heapq.heappush(self._heap, (-priority, self._counter, job_id))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None
        _, _, job_id = heapq.heappop(self._heap)
        job = self._jobs.pop(job_id, None)
        return (job.id, job.data) if job else None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        policy = RetryPolicy()
        success, _ = policy.run(processor, job.data)
        return success