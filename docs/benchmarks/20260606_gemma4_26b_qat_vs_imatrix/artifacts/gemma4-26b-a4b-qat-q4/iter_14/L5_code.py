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
        self._jobs: dict[str, Job] = {}
        self._pq: list[tuple[int, int, Job]] = []
        self._counter: int = 0
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        # Use -priority for max-heap behavior (highest priority first)
        # Use self._counter to ensure FIFO for jobs with the same priority
        heapq.heappush(self._pq, (-priority, self._counter, job))
        self._counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False

        success, _ = self._retry_policy.run(processor, job.data)
        if success:
            del self._jobs[job_id]
            return True
        return False

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._pq:
            _, _, job = heapq.heappop(self._pq)
            # Check if job is still in the registry (to handle lazy deletion)
            if job.id in self._jobs and self._jobs[job.id] is job:
                return job.id, job.data
        return None