import heapq
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: Any) -> tuple[bool, int]:
        attempts_made = 0
        for i in range(1, self.max_attempts + 1):
            attempts_made = i
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
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        self._counter += 1
        # Use -priority for max-heap behavior with heapq, 
        # and self._counter to ensure FIFO within the same priority.
        heapq.heappush(self._heap, (-priority, self._counter, job_id))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        success, _ = self.retry_policy.run(processor, job.data)

        if success:
            self._jobs.pop(job_id)

        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            _, _, job_id = heapq.heappop(self._heap)
            if job_id in self._jobs:
                job = self._jobs[job_id]
                return job.id, job.data
        return None