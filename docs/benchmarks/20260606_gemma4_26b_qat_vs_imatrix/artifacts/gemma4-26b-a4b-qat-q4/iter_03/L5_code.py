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
        attempts_made = 0
        while attempts_made < self.max_attempts:
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
        self._heap: list[tuple[int, int, str]] = []
        self._counter: int = 0
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job
        self._counter += 1
        # Use negative priority for max-heap behavior and counter for FIFO tie-breaking
        heapq.heappush(self._heap, (-priority, self._counter, job_id))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        success, attempts_made = self._retry_policy.run(processor, job.data)

        # Update the job's retry count (retries = total attempts - 1)
        job.retries = max(0, attempts_made - 1)

        # Remove job from the queue after processing attempt
        del self._jobs[job_id]
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._heap:
            # Check the top of the heap
            _, _, job_id = self._heap[0]

            # If the job is still in the registry, it's the next valid job
            if job_id in self._jobs:
                job = self._jobs[job_id]
                return job.id, job.data
            else:
                # If job was removed by process_job, clean up the heap
                heapq.heappop(self._heap)

        return None