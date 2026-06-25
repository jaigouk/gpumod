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
        for attempt in range(1, self.max_attempts + 1):
            try:
                fn(data)
                return True, attempt
            except Exception:
                continue
        return False, self.max_attempts

class JobQueue:
    def __init__(self, retry_policy: RetryPolicy | None = None):
        self._jobs: dict[str, Job] = {}
        self._heap = []
        self._counter = 0
        self._retry_policy = retry_policy or RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        # Use negative priority for max-heap behavior and counter for FIFO tie-breaking
        heapq.heappush(self._heap, (-priority, self._counter, job_id))
        self._counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False

        success, _ = self._retry_policy.run(processor, job.data)
        if success:
            self._jobs.pop(job_id)
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            _, _, job_id = heapq.heappop(self._heap)
            if job_id in self._jobs:
                job = self._jobs.pop(job_id)
                return job.id, job.data
        return None