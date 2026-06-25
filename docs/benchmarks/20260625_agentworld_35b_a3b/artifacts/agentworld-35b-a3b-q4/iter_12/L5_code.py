from dataclasses import dataclass
from typing import Callable, Dict, Optional
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
        for attempt in range(1, self.max_attempts + 1):
            attempts_made = attempt
            try:
                fn(data)
                return (True, attempts_made)
            except Exception:
                pass
        return (False, attempts_made)

class JobQueue:
    def __init__(self):
        self._heap: list = []
        self._counter = 0
        self._jobs: Dict[str, dict] = {}
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._jobs[job_id] = data
        heapq.heappush(self._heap, (-priority, self._counter, job_id))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            _, _, job_id = heapq.heappop(self._heap)
            if job_id in self._jobs:
                data = self._jobs[job_id]
                return (job_id, data)
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False
        
        data = self._jobs[job_id]
        success, attempts = self.retry_policy.run(lambda d: processor(d), data)
        if success:
            if job_id in self._jobs:
                del self._jobs[job_id]
        return success