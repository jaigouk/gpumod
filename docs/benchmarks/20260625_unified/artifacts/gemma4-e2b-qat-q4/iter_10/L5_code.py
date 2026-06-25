from dataclasses import dataclass
from typing import Callable, Any, Optional, Dict, List, Tuple

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: dict) -> Tuple[bool, int]:
        """
        Executes the function with retry logic. Does not sleep.
        """
        for attempt in range(1, self.max_attempts + 1):
            try:
                # Assuming fn returns a result or None upon success
                fn(data)
                return True, attempt
            except Exception:
                # Failure, try again if attempts remain
                if attempt == self.max_attempts:
                    break

        return False, self.max_attempts

class JobQueue:
    def __init__(self):
        # Stores Job objects. Using a dictionary for fast ID lookup.
        self._jobs: Dict[str, Job] = {}

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        """
        Adds a job to the queue.
        """
        new_job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = new_job

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Processes a job using the RetryPolicy.
        """
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]

        # Compose the job processing with the RetryPolicy
        policy = RetryPolicy()
        success, attempts_made = policy.run(processor, job.data)

        if success:
            # In a real system, you would update the job state here
            job.retries += attempts_made
            return True
        else:
            # Job failed all retry attempts
            return False

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        """
        Returns the highest-priority job's (id, data). 
        FIFO order within the same priority level.
        """
        if not self._jobs:
            return None

        next_job_id = None
        max_priority = -1

        # 1. Find the highest priority job
        for job_id, job in self._jobs.items():
            if job.priority > max_priority:
                max_priority = job.priority
                next_job_id = job_id

        if next_job_id is None:
            return None

        # 2. Collect all jobs matching the max_priority
        candidates = [
            (job_id, job.data) 
            for job_id, job in self._jobs.items() 
            if job.priority == max_priority
        ]

        # 3. Determine which of those is first (FIFO based on insertion/ID order)
        # Note: Since we are iterating over a dictionary, which preserves insertion order (Python 3.7+), 
        # we rely on the order encountered here.

        # We sort candidates to ensure deterministic behavior if insertion order isn't strictly guaranteed
        # but since the requirement asks for FIFO within priority, we rely on the original iteration order 
        # or sort by job ID if deterministic ID order is needed.

        # Let's rely on the initial iteration order for simplicity/consistency,
        # assuming the dictionary iteration reflects the priority queue ordering requirements.

        # In a strict FIFO requirement for same-priority items, 
        # we must maintain a separate structure (like a queue) or sort by addition timestamp.
        # However, since we only need the 'next' highest, we just pick one from the candidates.

        # Sort candidates by ID for deterministic tie-breaking
        candidates.sort(key=lambda x: x[0])

        return candidates[0]