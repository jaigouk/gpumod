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
        self.jobs: dict[str, Job] = {}
        self.heap = []
        self._counter = 0
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self.jobs[job_id] = job
        # Push (-priority) to use heapq as a max-priority queue
        # Push self._counter to ensure FIFO for identical priorities
        heapq.heappush(self.heap, (-priority, self._counter, job_id))
        self._counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        success, attempts_made = self._retry_policy.run(processor, job.data)
        job.retries = attempts_made - 1

        # Once processed (successfully or exhausted), remove from active jobs
        del self.jobs[job_id]
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        while self.heap:
            _, _, job_id = heapq.heappop(self.heap)
            if job_id in self.jobs:
                job = self.jobs[job_id]
                return job.id, job.data
        return None