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
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            try:
                fn(data)
                return True, attempts
            except Exception:
                pass
        return False, attempts

class JobQueue:
    def __init__(self):
        self._jobs = {}
        self._heap = []
        self._counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        self._counter += 1
        # heapq is a min-heap. We use -priority to simulate a max-priority queue.
        # We use self._counter to maintain FIFO order for equal priorities.
        heapq.heappush(self._heap, (-priority, self._counter, job_id))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        policy = RetryPolicy()
        success, attempts = policy.run(processor, job.data)
        job.retries = attempts

        if success:
            del self._jobs[job_id]
            return True
        return False

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            # Peek at the top of the heap
            _, _, job_id = self._heap[0]

            # If the job still exists in our dictionary, it is valid
            if job_id in self._jobs:
                job = self._jobs[job_id]
                return (job.id, job.data)
            else:
                # Job was processed or removed; clean up the heap
                heapq.heappop(self._heap)
        return None