import heapq
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional

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
        self._jobs: Dict[str, Job] = {}
        self._heap: list[tuple[int, int, str]] = []
        self._enqueue_counter = 0
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        if job_id not in self._jobs:
            self._jobs[job_id] = Job(id=job_id, data=data, priority=priority, retries=0)
        else:
            self._jobs[job_id].data = data
            self._jobs[job_id].priority = priority
        self._enqueue_counter += 1
        heapq.heappush(self._heap, (-priority, self._enqueue_counter, job_id))

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            neg_priority, enq_order, job_id = heapq.heappop(self._heap)
            if job_id in self._jobs:
                job = self._jobs[job_id]
                return (job.id, job.data)
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False
        job = self._jobs[job_id]
        success, attempts = self.retry_policy.run(lambda data: processor(data), job.data)
        if success:
            del self._jobs[job_id]
            return True
        else:
            job.retries += 1
            self._enqueue_counter += 1
            heapq.heappush(self._heap, (-job.priority, self._enqueue_counter, job_id))
            return False