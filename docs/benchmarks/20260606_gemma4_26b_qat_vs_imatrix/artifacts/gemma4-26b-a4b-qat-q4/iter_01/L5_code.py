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

    def run(self, fn: Callable, data: dict) -> tuple[bool, int]:
        attempts_made = 0
        for _ in range(self.max_attempts):
            attempts_made += 1
            try:
                fn(data)
                return True, attempts_made
            except Exception:
                continue
        return False, attempts_made

class JobQueue:
    def __init__(self):
        self._jobs = {}
        self._heap = []
        self._counter = 0
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        # Use -priority for max-heap behavior with heapq's min-heap
        # Use self._counter to ensure FIFO order for jobs with same priority
        heapq.heappush(self._heap, (-priority, self._counter, job_id))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            _, _, job_id = heapq.heappop(self._heap)
            if job_id in self._jobs:
                job = self._jobs[job_id]
                return job.id, job.data
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False

        success, attempts = self._retry_policy.run(processor, job.data)
        job.retries = attempts

        if success:
            self._jobs.pop(job_id)
            return True
        return False