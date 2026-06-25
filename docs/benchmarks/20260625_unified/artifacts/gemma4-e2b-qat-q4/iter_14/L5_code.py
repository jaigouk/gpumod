from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    """Encapsulates retry-with-backoff logic."""
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: Any) -> Tuple[bool, int]:
        for attempt in range(1, self.max_attempts + 1):
            try:
                fn(data)
                return True, attempt
            except Exception:
                if attempt == self.max_attempts:
                    return False, attempt
        # Should be unreachable if max_attempts >= 1, but kept for safety
        return False, self.max_attempts

class JobQueue:
    """Orchestrates Jobs using RetryPolicy."""
    def __init__(self):
        # Stores job details: {job_id: Job instance}
        self._jobs: Dict[str, Job] = {}
        # Simple counter to maintain FIFO order for jobs with the same priority
        self._counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        """Adds a new job to the queue."""
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """Processes a job using the RetryPolicy."""
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        policy = RetryPolicy()

        # The policy handles the retry logic internally
        success, attempts = policy.run(processor, job.data)

        if success:
            # In a real system, we might update the job state or remove it from the queue
            print(f"Job {job_id} completed successfully after {attempts} attempts.")
        else:
            print(f"Job {job_id} failed permanently after {attempts} attempts.")

        return success

    def get_next_job(self) -> Tuple[str, dict] | None:
        """Returns the highest-priority job's (id, data)."""
        if not self._jobs:
            return None

        # Sort criteria: 
        # 1. Highest priority first (descending: -job.priority)
        # 2. FIFO for ties (ascending: job.retries, or based on insertion order logic)

        # We sort by: (-priority, job_id_sort_key)
        # If we assume lower Job IDs or simply relying on the dictionary iteration 
        # order to handle stable FIFO within the same priority:

        sorted_jobs = sorted(
            self._jobs.values(), 
            key=lambda j: (-j.priority, j.id)
        )

        if not sorted_jobs:
            return None

        return sorted_jobs[0].id, sorted_jobs[0].data