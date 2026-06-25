from dataclasses import dataclass
from typing import Callable, List, Tuple, Any

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: Any) -> Tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                result = fn(data)
                return True, attempts + 1
            except Exception as e:
                attempts += 1
                # Simulate backoff by incrementing the attempt counter
                # No actual sleep is performed as per requirement

        return False, self.max_attempts

class JobQueue:
    def __init__(self):
        # Stores (priority, arrival_order, job)
        # arrival_order ensures FIFO within the same priority level.
        self._jobs: List[Tuple[int, int, Job]] = []
        self._next_arrival_order = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs.append((priority, self._next_arrival_order, job))
        self._next_arrival_order += 1
        # Keep sorted to facilitate easy retrieval of highest priority jobs
        self._jobs.sort(key=lambda x: (x[0], x[1]))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        # Find and remove the job (using FIFO logic implicit in iteration/sorting)
        for i, (_, _, job) in enumerate(self._jobs):
            if job.id == job_id:
                # Remove job
                self._jobs.pop(i)
                break

        if not self._jobs:
            return False

        # Extract the job data
        _, _, job = self._jobs[0]

        # Apply RetryPolicy
        policy = RetryPolicy()
        success, attempts = policy.run(processor, job.data)

        if success:
            # If successful, remove the job
            self._jobs.pop(0)
            return True
        else:
            # If failed after max attempts, keep the job in queue (or re-add based on requirements, 
            # but usually if it fails permanently, it's dropped/moved. Here we assume 
            # failure means we don't remove it unless the policy explicitly drops it, 
            # but standard retry logic keeps it pending for future runs/re-processing)
            # For this simple implementation, we just return False and keep the job pending.
            return False

    def get_next_job(self) -> Tuple[str, dict] | None:
        if not self._jobs:
            return None

        # The list is maintained sorted by priority (low number = high priority).
        # The job at index 0 is the highest priority job.
        _, _, job = self._jobs[0]
        return (job.id, job.data)