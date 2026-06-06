from dataclasses import dataclass
from typing import Callable, Any
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

    def run(self, fn: Callable, data: Any) -> tuple[bool, int]:
        for attempt in range(1, self.max_attempts + 1):
            try:
                fn(data)
                return True, attempt
            except Exception:
                continue
        return False, self.max_attempts

class JobQueue:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._queue: list[tuple[int, int, str]] = []
        self._counter = 0
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        heapq.heappush(self._queue, (-priority, self._counter, job_id))
        self._counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False

        success, attempts = self._retry_policy.run(processor, job.data)
        if success:
            del self._jobs[job_id]
        else:
            job.retries += attempts
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._queue:
            _, _, job_id = heapq.heappop(self._queue)
            if job_id in self._jobs:
                job = self._jobs[job_id]
                return job.id, job.data
        return None