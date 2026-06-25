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
        while attempts < self.max_attempts:
            attempts += 1
            try:
                fn(data)
                return True, attempts
            except Exception:
                continue
        return False, attempts

class JobQueue:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._heap: list[tuple[int, int, str]] = []
        self._counter: int = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        # Use negative priority for max-heap behavior with heapq
        # Use self._counter to ensure FIFO within the same priority
        heapq.heappush(self._heap, (-priority, self._counter, job_id))
        self._counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False

        policy = RetryPolicy()
        success, _ = policy.run(processor, job.data)

        if success:
            self._jobs.pop(job_id, None)
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            _, _, job_id = heapq.heappop(self._heap)
            if job_id in self._jobs:
                job = self._jobs[job_id]
                return (job.id, job.data)
        return None