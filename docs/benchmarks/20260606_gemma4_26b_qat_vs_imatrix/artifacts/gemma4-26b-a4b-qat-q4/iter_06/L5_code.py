from dataclasses import dataclass
from typing import Callable
import heapq
from itertools import count

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
        self._heap = []
        self._job_map = {}
        self._sequence = count()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._job_map[job_id] = job
        # Use negative priority for max-heap behavior via heapq (min-heap)
        # Use sequence counter to ensure FIFO for identical priorities
        heapq.heappush(self._heap, (-priority, next(self._sequence), job_id))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._job_map.get(job_id)
        if not job:
            return False

        policy = RetryPolicy()
        success, _ = policy.run(processor, job.data)
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            _, _, job_id = heapq.heappop(self._heap)
            if job_id in self._job_map:
                job = self._job_map.pop(job_id)
                return job.id, job.data
        return None